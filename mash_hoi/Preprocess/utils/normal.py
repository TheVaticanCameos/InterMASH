"""
Point cloud normal estimation utilities.

This module provides three methods for computing point cloud normals with consistent outward orientation:
1. PyTorch3D-based method (GPU-accelerated, batch processing)
2. Open3D-based method (CPU, single point cloud)
3. WNNC-based method (Winding Number Normal Consistency, GPU-accelerated)

Both methods ensure normals point outward from the point cloud surface.
"""

import numpy as np
import torch
import torch.nn.functional as F
import open3d as o3d
from typing import Optional
from pytorch3d.ops import estimate_pointcloud_normals, knn_points
import wn_treecode


# =============================================================================
# PyTorch3D-based methods
# =============================================================================

def orient_normals_consistent(
    points: torch.Tensor,
    normals: torch.Tensor,
    k: int = 30,
    iterations: int = 3,
) -> torch.Tensor:
    """
    Adjust normal vectors for consistency using neighbor voting.
    
    For each point, if the majority of its neighbors have opposite-facing normals,
    flip the normal to match the majority.
    
    Args:
        points: [B, N, 3] point cloud
        normals: [B, N, 3] initial normals
        k: Number of neighbors to consider
        iterations: Number of consistency iterations
        
    Returns:
        normals: [B, N, 3] adjusted normals with consistent orientation
    """
    
    for _ in range(iterations):
        # Find K+1 nearest neighbors (including self)
        knn_result = knn_points(points, points, K=k+1)
        neighbor_indices = knn_result.idx[:, :, 1:]  # Exclude self
        
        # Gather neighbor normals
        B, N, _ = normals.shape
        neighbor_normals = torch.gather(
            normals.unsqueeze(2).expand(-1, -1, k, -1),
            1,
            neighbor_indices.unsqueeze(-1).expand(-1, -1, -1, 3)
        )
        
        # Compute dot products with neighbors
        dots = torch.sum(normals.unsqueeze(2) * neighbor_normals, dim=-1)
        
        # If more than half of neighbors are opposite, flip
        should_flip = (dots < 0).float().mean(dim=-1) > 0.5
        normals = normals.clone()
        normals[should_flip] = -normals[should_flip]
    
    return normals


def orient_normals_outward(
    points: torch.Tensor,
    normals: torch.Tensor,
    k: int = 30,
) -> torch.Tensor:
    """
    Orient normals to point outward from the point cloud.
    
    First ensures local consistency, then uses bounding box exterior point
    as reference to determine global outward direction.
    
    Args:
        points: [B, N, 3] point cloud
        normals: [B, N, 3] initial normals
        k: Number of neighbors for consistency check
        
    Returns:
        normals: [B, N, 3] normals pointing outward
    """
    
    # First ensure local consistency
    normals = orient_normals_consistent(points, normals, k, iterations=3)
    
    # Compute bounding box and external reference point
    bbox_max = points.max(dim=1, keepdim=True)[0]  # [B, 1, 3]
    bbox_min = points.min(dim=1, keepdim=True)[0]  # [B, 1, 3]
    bbox_center = (bbox_max + bbox_min) / 2
    
    # External reference point beyond the bounding box
    external_ref = bbox_max + (bbox_max - bbox_center) * 0.5  # [B, 1, 3]
    
    # Compute outward vectors from each point to external reference
    outward_vecs = external_ref - points  # [B, N, 3]
    outward_vecs = outward_vecs / (outward_vecs.norm(dim=-1, keepdim=True) + 1e-8)
    
    # Check overall alignment
    alignment = (normals * outward_vecs).sum(dim=-1).mean()
    
    # If mostly pointing inward, flip all normals
    if alignment < 0:
        normals = -normals
    
    return normals


def estimate_and_orient_normals_pytorch3d(
    points: torch.Tensor,
    neighborhood_size: int = 50,
    orient_outward: bool = True,
) -> torch.Tensor:
    """
    Estimate point cloud normals using PyTorch3D and orient them consistently.
    
    Args:
        points: [B, N, 3] or [N, 3] point cloud
        neighborhood_size: Number of neighbors for normal estimation
        orient_outward: If True, orient normals to point outward
        
    Returns:
        normals: Same shape as input, unit normals for each point
    """
    
    # Handle 2D input
    squeeze_output = False
    if points.dim() == 2:
        points = points.unsqueeze(0)
        squeeze_output = True
    
    # Estimate normals using PyTorch3D
    normals = estimate_pointcloud_normals(points, neighborhood_size=neighborhood_size)
    
    # Orient normals
    if orient_outward:
        normals = orient_normals_outward(points, normals, k=neighborhood_size)
    else:
        normals = orient_normals_consistent(points, normals, k=neighborhood_size, iterations=3)
    
    if squeeze_output:
        normals = normals.squeeze(0)
    
    return normals


# =============================================================================
# Open3D-based methods
# =============================================================================

def estimate_and_orient_normals_open3d(
    points: torch.Tensor,
    k: int = 30,
    orient_outward: bool = True,
) -> torch.Tensor:
    """
    Estimate point cloud normals using Open3D and orient them consistently.
    
    Args:
        points: [N, 3] or [B, N, 3] torch.Tensor
        k: Number of neighbors for normal estimation
        orient_outward: If True, orient normals to point outward
        
    Returns:
        normals: same shape as input, torch.Tensor of unit normals
    """
    device = points.device
    dtype = points.dtype
    
    # Handle batched input by processing each batch element
    if points.ndim == 3:  # [B, N, 3]
        B = points.shape[0]
        normals_list = []
        for b in range(B):
            normals_b = _estimate_and_orient_normals_open3d_single(
                points[b], k, orient_outward
            )
            normals_list.append(normals_b)
        return torch.stack(normals_list, dim=0).to(device=device, dtype=dtype)
    
    # Single sample case
    return _estimate_and_orient_normals_open3d_single(points, k, orient_outward).to(device=device, dtype=dtype)


def _estimate_and_orient_normals_open3d_single(
    points: torch.Tensor,
    k: int = 30,
    orient_outward: bool = True,
) -> torch.Tensor:
    """
    Estimate normals for a single point cloud using Open3D.
    
    Args:
        points: [N, 3] torch.Tensor
        k: Number of neighbors for normal estimation
        orient_outward: If True, orient normals to point outward
        
    Returns:
        normals: [N, 3] torch.Tensor of unit normals
    """
    device = points.device
    dtype = points.dtype
    
    # Convert to numpy
    points_np = points.detach().cpu().numpy()
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_np)
    
    # Estimate normals
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k)
    )
    
    if orient_outward:
        # First do consistency adjustment using tangent plane propagation
        pcd.orient_normals_consistent_tangent_plane(k=k)
        
        # Compute bounding box and external reference point
        bbox = pcd.get_axis_aligned_bounding_box()
        bbox_max = bbox.get_max_bound()
        bbox_min = bbox.get_min_bound()
        bbox_center = (bbox_max + bbox_min) / 2
        external_ref = bbox_max + (bbox_max - bbox_center) * 0.5
        
        # Check overall alignment with outward direction
        normals = np.asarray(pcd.normals)
        outward_vecs = external_ref - points_np
        outward_vecs = outward_vecs / (np.linalg.norm(outward_vecs, axis=1, keepdims=True) + 1e-8)
        alignment = np.mean(np.sum(normals * outward_vecs, axis=1))
        
        # If pointing inward on average, flip all normals
        if alignment < 0:
            pcd.normals = o3d.utility.Vector3dVector(-normals)
    else:
        # Only do consistency adjustment
        pcd.orient_normals_consistent_tangent_plane(k=k)
    
    normals_np = np.asarray(pcd.normals)
    return torch.from_numpy(normals_np).to(device=device, dtype=dtype)


# =============================================================================
# WNNC-based methods (Winding Number Normal Consistency)
# =============================================================================

def estimate_normals_wnnc(
    points: torch.Tensor,
    width_config: str = 'l1',
    wsmin: float = 0.01,
    wsmax: float = 0.04,
    iterations: int = 40,
) -> torch.Tensor:
    """
    Estimate point cloud normals using WNNC (Winding Number Normal Consistency).
    
    WNNC is particularly robust for noisy or sparse point clouds.
    The computation is performed on the same device as the input tensor.
    
    Args:
        points: [N, 3] or [B, N, 3] torch.Tensor
        width_config: Preset width configuration:
            - 'l0': [0.002, 0.016] - noise free points
            - 'l1': [0.01, 0.04] - real scans (default)
            - 'l2': [0.02, 0.08] - sigma=0.25% noise
            - 'l3': [0.03, 0.12] - sigma=0.5% noise
            - 'l4': [0.04, 0.16] - sigma=1% noise
            - 'l5': [0.05, 0.2] - sparse points
            - 'custom': use wsmin and wsmax parameters
        wsmin: Minimum smoothing width (only for 'custom' config)
        wsmax: Maximum smoothing width (only for 'custom' config)
        iterations: Number of optimization iterations
        
    Returns:
        normals: [N, 3] or [B, N, 3] torch.Tensor of unit normals (same shape and device as input)
    """
    device = points.device
    dtype = points.dtype
    
    # Handle batched input by processing each batch element
    if points.ndim == 3:  # [B, N, 3]
        B = points.shape[0]
        normals_list = []
        for b in range(B):
            normals_b = _estimate_normals_wnnc_single(
                points[b], width_config, wsmin, wsmax, iterations
            )
            normals_list.append(normals_b)
        return torch.stack(normals_list, dim=0).to(dtype=dtype)
    
    # Single sample case
    return _estimate_normals_wnnc_single(points, width_config, wsmin, wsmax, iterations).to(dtype=dtype)


def _estimate_normals_wnnc_single(
    points: torch.Tensor,
    width_config: str = 'l1',
    wsmin: float = 0.01,
    wsmax: float = 0.04,
    iterations: int = 40,
) -> torch.Tensor:
    """
    Estimate normals for a single point cloud using WNNC.
    
    Args:
        points: [N, 3] torch.Tensor
        width_config, wsmin, wsmax, iterations: same as estimate_normals_wnnc
        
    Returns:
        normals: [N, 3] torch.Tensor of unit normals
    """
    device = points.device
    points_input = points.detach()
    
    # Normalize points to [-1, 1] range
    bbox_scale = 1.1
    bbox_min = points_input.min(dim=0)[0]  # [3]
    bbox_max = points_input.max(dim=0)[0]  # [3]
    bbox_center = (bbox_min + bbox_max) / 2.0
    bbox_len = (bbox_max - bbox_min).max()
    points_normalized = (points_input - bbox_center) * (2 / (bbox_len * bbox_scale))
    
    # Ensure contiguous float tensors on the same device
    points_tensor = points_normalized.contiguous().float()
    normals = torch.zeros_like(points_tensor).contiguous()
    b = torch.ones(points_tensor.shape[0], 1, device=device, dtype=torch.float32) * 0.5
    widths = torch.ones(points_tensor.shape[0], device=device, dtype=torch.float32)
    
    # Create WNNC function (uses the device of points_tensor)
    wn_func = wn_treecode.WindingNumberTreecode(points_tensor)
    
    # Preset width configurations
    preset_widths = {
        'l0': [0.002, 0.016],   # noise free
        'l1': [0.01, 0.04],     # real scans
        'l2': [0.02, 0.08],     # sigma=0.25%
        'l3': [0.03, 0.12],     # sigma=0.5%
        'l4': [0.04, 0.16],     # sigma=1%
        'l5': [0.05, 0.2],      # sparse points
        'custom': [wsmin, wsmax],
    }
    
    ws_min, ws_max = preset_widths[width_config]
    
    # Iterative optimization
    with torch.no_grad():
        for i in range(iterations):
            width_scale = ws_min + ((iterations - 1 - i) / (iterations - 1)) * (ws_max - ws_min)
            
            # Gradient step
            A_mu = wn_func.forward_A(normals, widths * width_scale)
            AT_A_mu = wn_func.forward_AT(A_mu, widths * width_scale)
            r = wn_func.forward_AT(b, widths * width_scale) - AT_A_mu
            A_r = wn_func.forward_A(r, widths * width_scale)
            alpha = (r * r).sum() / (A_r * A_r).sum()
            normals = normals + alpha * r
            
            # WNNC step
            out_normals = wn_func.forward_G(normals, widths * width_scale)
            
            # Rescale
            out_normals = F.normalize(out_normals, dim=-1).contiguous()
            normals_len = torch.linalg.norm(normals, dim=-1, keepdim=True)
            normals = out_normals.clone() * normals_len
    
    # Final normalization
    out_normals = F.normalize(normals, dim=-1)
    
    return out_normals


# =============================================================================
# Comparison and testing
# =============================================================================

def compare_normal_estimation_methods(
    points: torch.Tensor,
    neighborhood_size: int = 30,
    expected_normals: Optional[torch.Tensor] = None,
    verbose: bool = True,
) -> dict:
    """
    Compare PyTorch3D and Open3D normal estimation methods.
    
    Args:
        points: [N, 3] point cloud (torch.Tensor)
        neighborhood_size: Number of neighbors for normal estimation
        expected_normals: [N, 3] ground truth normals (optional)
        verbose: If True, print comparison results
        
    Returns:
        dict with comparison metrics
    """
    results = {}
    
    # PyTorch3D estimation
    normals_pt3d = estimate_and_orient_normals_pytorch3d(
        points, neighborhood_size=neighborhood_size, orient_outward=True
    )
    results['pytorch3d'] = normals_pt3d
    
    # Open3D estimation
    normals_o3d_np = estimate_and_orient_normals_open3d(
        points, k=neighborhood_size, orient_outward=False
    )
    normals_o3d = torch.from_numpy(normals_o3d_np).float()
    results['open3d'] = normals_o3d
    
    if verbose:
        print("=" * 60)
        print("Normal Estimation Comparison")
        print("=" * 60)
    
    # Compare with expected normals if provided
    if expected_normals is not None:
        alignment_pt3d = (normals_pt3d * expected_normals).sum(dim=-1).mean()
        results['pytorch3d_alignment'] = alignment_pt3d.item()
        if verbose:
            print(f"\nPyTorch3D vs Expected:")
            print(f"  Direction alignment: {alignment_pt3d.item():.4f}")
        
        alignment_o3d = (normals_o3d * expected_normals).sum(dim=-1).mean()
        results['open3d_alignment'] = alignment_o3d.item()
        if verbose:
            print(f"\nOpen3D vs Expected:")
            print(f"  Direction alignment: {alignment_o3d.item():.4f}")
    
    # Compare the two methods
    diff = torch.abs(normals_pt3d - normals_o3d).mean()
    cosine_sim = (normals_pt3d * normals_o3d).sum(dim=-1)
    
    results['mean_difference'] = diff.item()
    results['cosine_sim_mean'] = cosine_sim.mean().item()
    results['cosine_sim_min'] = cosine_sim.min().item()
    
    if verbose:
        print(f"\nPyTorch3D vs Open3D:")
        print(f"  Mean absolute difference: {diff.item():.6f}")
        print(f"  Cosine similarity (mean): {cosine_sim.mean().item():.4f}")
        print(f"  Cosine similarity (min):  {cosine_sim.min().item():.4f}")
    
    return results

