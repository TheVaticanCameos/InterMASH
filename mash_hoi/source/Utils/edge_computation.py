"""
Edge computation utilities for MASH parameters.

This module provides functions to:
- Compute edge/boundary of point clouds from MASH parameters
- Fit a disk (circle in 3D space) to approximate the edge
- Return disk parameters: center, radius, and normal vector
"""

import torch
import numpy as np
from typing import Tuple, Optional
from .preprocess_utils import (
    split_mash_params,
    compute_sample_points_from_mash_params,
)


def compute_edge_disk_from_mash(
    mash_params: torch.Tensor,
    positions: torch.Tensor,
    mask_dim: int,
    sh_dim: int,
    sample_phi_num: int,
    mask_degree_max: int,
    sh_degree_max: int,
    edge_threshold: float = 0.1,
    target: str = "object",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute edge disk from MASH parameters.
    
    Process:
    1. Generate sample point cloud from MASH parameters
    2. Detect edge/boundary points based on distance from center
    3. Fit a disk (circle in 3D) to the edge points
    4. Return disk parameters: center, radius, normal
    
    Args:
        mash_params: [B, N, D] MASH parameters (mask + sh + ortho)
        positions: [B, N, 3] anchor positions
        mask_dim: Dimension of mask parameters
        sh_dim: Dimension of spherical harmonics parameters
        sample_phi_num: Number of phi samples for point cloud generation
        mask_degree_max: Maximum degree for mask Fourier series
        sh_degree_max: Maximum degree for spherical harmonics
        edge_threshold: Threshold percentile (0-1) for edge detection (default: 0.1 = top 10%)
        target: "hand" or "object"
        
    Returns:
        disk_centers: [B, N, 3] disk center for each anchor
        disk_radii: [B, N] disk radius for each anchor
        disk_normals: [B, N, 3] disk normal vector for each anchor
    """
    B, N, _ = mash_params.shape
    device = mash_params.device
    dtype = mash_params.dtype
    
    # Step 1: Split MASH parameters
    mask_params, sh_params, ortho_poses = split_mash_params(mash_params, mask_dim, sh_dim)
    
    # Step 2: Generate sample point cloud
    # sample_points: [B, N, S, S, 3] where S = sample_phi_num
    sample_points = compute_sample_points_from_mash_params(
        mask_params=mask_params,
        sh_params=sh_params,
        ortho_poses=ortho_poses,
        positions=positions,
        sample_phi_num=sample_phi_num,
        mask_degree_max=mask_degree_max,
        sh_degree_max=sh_degree_max,
        target=target,
    )
    
    # Reshape to [B, N, M, 3] where M = S*S
    S = sample_phi_num
    sample_points_flat = sample_points.reshape(B, N, S * S, 3)
    
    # Step 3: Detect edge points for each anchor
    # Strategy: Points that are farthest from the anchor center are likely edge points
    disk_centers = torch.zeros(B, N, 3, device=device, dtype=dtype)
    disk_radii = torch.zeros(B, N, device=device, dtype=dtype)
    disk_normals = torch.zeros(B, N, 3, device=device, dtype=dtype)
    
    for b in range(B):
        for n in range(N):
            points = sample_points_flat[b, n]  # [M, 3]
            anchor_pos = positions[b, n]  # [3]
            
            # Compute distances from anchor position
            distances = torch.norm(points - anchor_pos.unsqueeze(0), dim=-1)  # [M]
            
            # Find edge points (top percentile of distances)
            threshold_dist = torch.quantile(distances, 1.0 - edge_threshold)
            edge_mask = distances >= threshold_dist
            edge_points = points[edge_mask]  # [K, 3] where K is number of edge points
            
            if edge_points.shape[0] < 3:
                # Not enough points for disk fitting, use all points
                edge_points = points
            
            # Step 4: Fit disk to edge points
            center, radius, normal = fit_disk_to_points(edge_points)
            
            disk_centers[b, n] = center
            disk_radii[b, n] = radius
            disk_normals[b, n] = normal
    
    return disk_centers, disk_radii, disk_normals


def fit_disk_to_points(points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fit a disk (circle in 3D space) to a set of points.
    
    Method:
    1. Use PCA to find the best-fit plane
    2. Project points onto the plane
    3. Compute center as mean of projected points
    4. Compute radius as mean distance from center
    5. Normal is the smallest principal component
    
    Args:
        points: [K, 3] point cloud
        
    Returns:
        center: [3] disk center
        radius: scalar disk radius
        normal: [3] disk normal vector (unit vector)
    """
    device = points.device
    dtype = points.dtype
    K = points.shape[0]
    
    if K < 3:
        # Not enough points, return degenerate disk
        center = points.mean(dim=0) if K > 0 else torch.zeros(3, device=device, dtype=dtype)
        radius = torch.tensor(0.01, device=device, dtype=dtype)
        normal = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)
        return center, radius, normal
    
    # Step 1: Compute center (mean of points)
    center = points.mean(dim=0)  # [3]
    
    # Step 2: Center the points
    centered_points = points - center.unsqueeze(0)  # [K, 3]
    
    # Step 3: Compute covariance matrix for PCA
    cov = torch.matmul(centered_points.T, centered_points) / (K - 1)  # [3, 3]
    
    # Step 4: Eigenvalue decomposition
    try:
        eigenvals, eigenvecs = torch.linalg.eigh(cov)  # eigenvals: [3], eigenvecs: [3, 3]
    except:
        # Fallback if decomposition fails
        radius = torch.norm(centered_points, dim=-1).mean()
        normal = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)
        return center, radius, normal
    
    # Step 5: Normal is the eigenvector with smallest eigenvalue (perpendicular to plane)
    normal = eigenvecs[:, 0]  # [3] - eigenvecs are sorted by eigenvalue in ascending order
    
    # Ensure normal points in positive Z direction (arbitrary choice for consistency)
    if normal[2] < 0:
        normal = -normal
    
    # Step 6: Project points onto the plane defined by normal
    # Distance from center in the plane
    distances_in_plane = torch.norm(centered_points - torch.outer(
        torch.sum(centered_points * normal.unsqueeze(0), dim=-1), normal
    ), dim=-1)  # [K]
    
    # Step 7: Radius is the mean distance in plane
    radius = distances_in_plane.mean()
    
    # Handle edge case where radius is too small
    if radius < 1e-6:
        radius = torch.tensor(0.01, device=device, dtype=dtype)
    
    return center, radius, normal


def compute_edge_disk_from_mash_batched(
    mash_params: torch.Tensor,
    positions: torch.Tensor,
    mask_dim: int,
    sh_dim: int,
    sample_phi_num: int,
    mask_degree_max: int,
    sh_degree_max: int,
    edge_threshold: float = 0.1,
    target: str = "object",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Batched version of edge disk computation (more efficient for large batches).
    
    Same as compute_edge_disk_from_mash but uses batched operations where possible.
    
    Args:
        mash_params: [B, N, D] MASH parameters (mask + sh + ortho)
        positions: [B, N, 3] anchor positions
        mask_dim: Dimension of mask parameters
        sh_dim: Dimension of spherical harmonics parameters
        sample_phi_num: Number of phi samples for point cloud generation
        mask_degree_max: Maximum degree for mask Fourier series
        sh_degree_max: Maximum degree for spherical harmonics
        edge_threshold: Threshold percentile (0-1) for edge detection
        target: "hand" or "object"
        
    Returns:
        disk_centers: [B, N, 3] disk center for each anchor
        disk_radii: [B, N] disk radius for each anchor
        disk_normals: [B, N, 3] disk normal vector for each anchor
    """
    B, N, _ = mash_params.shape
    device = mash_params.device
    dtype = mash_params.dtype
    
    # Step 1: Split MASH parameters
    mask_params, sh_params, ortho_poses = split_mash_params(mash_params, mask_dim, sh_dim)
    
    # Step 2: Generate sample point cloud
    sample_points = compute_sample_points_from_mash_params(
        mask_params=mask_params,
        sh_params=sh_params,
        ortho_poses=ortho_poses,
        positions=positions,
        sample_phi_num=sample_phi_num,
        mask_degree_max=mask_degree_max,
        sh_degree_max=sh_degree_max,
        target=target,
    )
    
    # Reshape to [B, N, M, 3] where M = S*S
    S = sample_phi_num
    M = S * S
    sample_points_flat = sample_points.reshape(B, N, M, 3)
    
    # Step 3: Compute distances from anchor positions (batched)
    # positions: [B, N, 3] -> [B, N, 1, 3]
    # sample_points_flat: [B, N, M, 3]
    distances = torch.norm(
        sample_points_flat - positions.unsqueeze(2), dim=-1
    )  # [B, N, M]
    
    # Step 4: Find edge points (top percentile) - batched
    threshold_dists = torch.quantile(
        distances, 1.0 - edge_threshold, dim=-1, keepdim=True
    )  # [B, N, 1]
    edge_masks = distances >= threshold_dists  # [B, N, M]
    
    # Step 5: Fit disks (batched where possible)
    disk_centers = torch.zeros(B, N, 3, device=device, dtype=dtype)
    disk_radii = torch.zeros(B, N, device=device, dtype=dtype)
    disk_normals = torch.zeros(B, N, 3, device=device, dtype=dtype)
    
    # Batch process by flattening B and N dimensions
    for b in range(B):
        for n in range(N):
            edge_mask = edge_masks[b, n]  # [M]
            edge_points = sample_points_flat[b, n][edge_mask]  # [K, 3]
            
            if edge_points.shape[0] < 3:
                # Use all points if not enough edge points
                edge_points = sample_points_flat[b, n]
            
            center, radius, normal = fit_disk_to_points(edge_points)
            
            disk_centers[b, n] = center
            disk_radii[b, n] = radius
            disk_normals[b, n] = normal
    
    return disk_centers, disk_radii, disk_normals


def visualize_disk_parameters(
    disk_centers: torch.Tensor,
    disk_radii: torch.Tensor,
    disk_normals: torch.Tensor,
    batch_idx: int = 0,
    anchor_idx: int = 0,
) -> dict:
    """
    Extract disk parameters for visualization (helper function).
    
    Args:
        disk_centers: [B, N, 3] disk centers
        disk_radii: [B, N] disk radii
        disk_normals: [B, N, 3] disk normals
        batch_idx: batch index to extract
        anchor_idx: anchor index to extract
        
    Returns:
        dict with 'center', 'radius', 'normal' as numpy arrays
    """
    center = disk_centers[batch_idx, anchor_idx].detach().cpu().numpy()
    radius = disk_radii[batch_idx, anchor_idx].detach().cpu().item()
    normal = disk_normals[batch_idx, anchor_idx].detach().cpu().numpy()
    
    return {
        'center': center,
        'radius': radius,
        'normal': normal,
    }


# Example usage
if __name__ == "__main__":
    # Example: Compute edge disk from MASH parameters
    B, N = 2, 5  # 2 batches, 5 anchors
    mask_dim = 5
    sh_dim = 9
    ortho_dim = 6
    D = mask_dim + sh_dim + ortho_dim
    
    # Create dummy MASH parameters
    mash_params = torch.randn(B, N, D)
    positions = torch.randn(B, N, 3)
    
    # Compute edge disks
    disk_centers, disk_radii, disk_normals = compute_edge_disk_from_mash(
        mash_params=mash_params,
        positions=positions,
        mask_dim=mask_dim,
        sh_dim=sh_dim,
        sample_phi_num=8,
        mask_degree_max=2,
        sh_degree_max=2,
        edge_threshold=0.1,
        target="object",
    )
    
    print("Disk centers shape:", disk_centers.shape)  # [B, N, 3]
    print("Disk radii shape:", disk_radii.shape)      # [B, N]
    print("Disk normals shape:", disk_normals.shape)  # [B, N, 3]
    
    # Visualize first disk
    disk_info = visualize_disk_parameters(
        disk_centers, disk_radii, disk_normals,
        batch_idx=0, anchor_idx=0
    )
    print("\nFirst disk parameters:")
    print(f"  Center: {disk_info['center']}")
    print(f"  Radius: {disk_info['radius']:.4f}")
    print(f"  Normal: {disk_info['normal']}")

