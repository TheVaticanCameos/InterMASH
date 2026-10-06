"""
Edge/disk computation utilities for point clouds.

This module provides functions to:
- Compute disk patches from anchor point clouds
- Calculate IoU matrix between object patches and hand patches
- Use WNNC for point cloud normal estimation

All operations use PyTorch tensors for GPU acceleration and differentiability.
All input data should be pre-computed point clouds.
"""

import math
import torch
from typing import Tuple, List, Dict

from mash_hoi.Preprocess.utils.normal import estimate_normals_wnnc


def compute_point_cloud_normals(
    points: torch.Tensor,
) -> torch.Tensor:
    """
    Compute normals for a point cloud using WNNC (Winding Number Normal Consistency).
    Normals are oriented to point outward (away from the centroid).
    
    Args:
        points: [M, 3] point cloud tensor (should be on CUDA for best performance)
        
    Returns:
        normals: [M, 3] unit normals for each point, pointing outward
    """
    normals = estimate_normals_wnnc(
        points,
        width_config='l1',
        wsmin=0.01,
        wsmax=0.04,
        iterations=40,
    )
    return normals


def fit_disk_with_normals(
    anchor_points: torch.Tensor,
    anchor_normals: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fit a disk to anchor points using precomputed normals.
    
    Args:
        anchor_points: [K, 3] points belonging to this anchor
        anchor_normals: [K, 3] corresponding normals
        
    Returns:
        center: [3] disk center
        radius: scalar tensor disk radius
        normal: [3] disk normal (unit vector)
    """
    device = anchor_points.device
    dtype = anchor_points.dtype
    K = anchor_points.shape[0]
    
    if K == 0:
        return (
            torch.zeros(3, device=device, dtype=dtype),
            torch.tensor(0.01, device=device, dtype=dtype),
            torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype),
        )
    
    # Center = mean of points
    center = anchor_points.mean(dim=0)
    
    # Normal = mean of point normals (normalized)
    mean_normal = anchor_normals.mean(dim=0)
    normal_norm = mean_normal.norm()
    if normal_norm < 1e-6:
        normal = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)
    else:
        normal = mean_normal / normal_norm
    
    # Project points onto the disk plane and compute radius
    centered_points = anchor_points - center
    proj_along_normal = (centered_points @ normal).unsqueeze(-1) * normal
    points_in_plane = centered_points - proj_along_normal
    distances_in_plane = points_in_plane.norm(dim=1)
    radius = distances_in_plane.mean()
    radius = torch.clamp(radius, min=1e-6)
    
    return center, radius, normal


def fit_disks_batched(
    points: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fit disks to anchor points using PCA to compute normals (batched version).
    
    Args:
        points: [B, N, S, 3] or [N, S, 3] points for each anchor
    
    Returns:
        centers: [B, N, 3] or [N, 3] disk centers
        radii: [B, N] or [N] disk radii
        disk_normals: [B, N, 3] or [N, 3] disk normals (unit vectors, computed via PCA)
    """
    squeeze_batch = False
    if points.dim() == 3:
        points = points.unsqueeze(0)
        squeeze_batch = True
    
    B, N, S, _ = points.shape
    device = points.device
    dtype = points.dtype
    
    # Center = mean of points per anchor
    centers = points.mean(dim=2)  # [B, N, 3]
    
    # Compute normals using PCA for each anchor
    centered_points = points - centers.unsqueeze(2)  # [B, N, S, 3]
    
    # Compute covariance matrix for each anchor: [B, N, 3, 3]
    # centered_points: [B, N, S, 3] -> transpose to [B, N, 3, S]
    cov = torch.matmul(centered_points.transpose(-1, -2), centered_points) / (S - 1)  # [B, N, 3, 3]
    
    # Eigenvalue decomposition
    eigenvals, eigenvecs = torch.linalg.eigh(cov)  # eigenvals: [B, N, 3], eigenvecs: [B, N, 3, 3]
    # Normal is the eigenvector with smallest eigenvalue (perpendicular to plane)
    # eigenvecs[:, :, :, 0] corresponds to smallest eigenvalue
    disk_normals = eigenvecs[:, :, :, 0]  # [B, N, 3]
    
    # Ensure normal points in positive Z direction for consistency
    z_negative = disk_normals[:, :, 2] < 0
    disk_normals[z_negative] = -disk_normals[z_negative]
    
    # Compute radius: project points onto plane and measure distance from center
    proj_along_normal = (centered_points * disk_normals.unsqueeze(2)).sum(dim=-1, keepdim=True)
    points_in_plane = centered_points - proj_along_normal * disk_normals.unsqueeze(2)
    distances_in_plane = points_in_plane.norm(dim=-1)  # [B, N, S]
    radii = distances_in_plane.mean(dim=-1)  # [B, N]
    radii = torch.clamp(radii, min=1e-6)
    
    if squeeze_batch:
        centers = centers.squeeze(0)
        radii = radii.squeeze(0)
        disk_normals = disk_normals.squeeze(0)
    
    return centers, radii, disk_normals


def compute_edge_matrix_batched(
    obj_centers: torch.Tensor,
    obj_radii: torch.Tensor,
    obj_normals: torch.Tensor,
    hand_centers: torch.Tensor,
    hand_radii: torch.Tensor,
    hand_normals: torch.Tensor,
    angle_threshold: float = 120.0,
    distance_threshold: float = 0.05,
    normalize_per_object: bool = True,
) -> torch.Tensor:
    """
    Compute distance-based edge weight matrix between object and hand patches.
    Uses Gaussian weighting based on 3D distance.
    
    Args:
        obj_centers: [B, N, 3] or [N, 3] object disk centers
        obj_radii: [B, N] or [N] object disk radii (not used, kept for compatibility)
        obj_normals: [B, N, 3] or [N, 3] object disk normals (not used, kept for compatibility)
        hand_centers: [B, N, 3] or [N, 3] hand disk centers
        hand_radii: [B, N] or [N] hand disk radii (not used, kept for compatibility)
        hand_normals: [B, N, 3] or [N, 3] hand disk normals (not used, kept for compatibility)
        angle_threshold: Deprecated, kept for compatibility but not used
        distance_threshold: Maximum center distance (meters) for valid edge
        normalize_per_object: If True, normalize weights per object patch so that
                              sum of hand weights for each object equals 1
        
    Returns:
        weight_matrix: [B, N, N] or [N, N] edge weight matrix (normalized if normalize_per_object=True)
    """
    squeeze_batch = False
    if obj_centers.dim() == 2:
        obj_centers = obj_centers.unsqueeze(0)
        obj_radii = obj_radii.unsqueeze(0)
        obj_normals = obj_normals.unsqueeze(0)
        hand_centers = hand_centers.unsqueeze(0)
        hand_radii = hand_radii.unsqueeze(0)
        hand_normals = hand_normals.unsqueeze(0)
        squeeze_batch = True
    
    B, N, _ = obj_centers.shape
    device = obj_centers.device
    dtype = obj_centers.dtype
    
    obj_centers_exp = obj_centers.unsqueeze(2)
    hand_centers_exp = hand_centers.unsqueeze(1)
    
    # Compute 3D distance between all pairs
    center_dist = (hand_centers_exp - obj_centers_exp).norm(dim=-1)  # [B, N, N]
    
    # Filter by distance threshold
    valid_mask = center_dist <= distance_threshold
    
    # Compute Gaussian weight based on distance
    # sigma = distance_threshold / 3, so weight drops to ~0.01 at threshold
    sigma = distance_threshold / 3.0
    weight = torch.exp(-center_dist ** 2 / (2 * sigma ** 2))
    
    # Apply mask: set weight to 0 for pairs beyond threshold
    weight = torch.where(valid_mask, weight, torch.zeros_like(weight))
    
    # Normalize per object patch: sum of hand weights for each object = 1
    if normalize_per_object:
        row_sum = weight.sum(dim=-1, keepdim=True)  # [B, N, 1]
        row_sum = torch.clamp(row_sum, min=1e-6)  # Avoid division by zero
        weight = weight / row_sum  # Each row sums to 1 (if non-zero)
    
    if squeeze_batch:
        weight = weight.squeeze(0)
    
    return weight


def process_batch(
    hand_points: torch.Tensor,
    object_points: torch.Tensor,
    angle_threshold: float = 120.0,
    distance_threshold: float = 0.05,
    normalize_per_object: bool = True,
) -> torch.Tensor:
    """
    Process a batch of pre-computed point clouds to compute IoU matrices.
    Only considers distance threshold for filtering. Normals are computed via PCA internally.
    
    Args:
        hand_points: [B, N, S, 3] or [N, S, 3] pre-computed hand point clouds per anchor
        object_points: [B, N, S, 3] or [N, S, 3] pre-computed object point clouds per anchor
        angle_threshold: Deprecated, kept for compatibility but not used for filtering
        distance_threshold: Center distance threshold for IoU (meters) - only filtering criterion
        normalize_per_object: If True, normalize IoU per object patch so that
                              sum of hand IoUs for each object equals 1
        
    Returns:
        iou_matrices: [B, N, N] or [N, N] IoU matrices tensor (normalized if normalize_per_object=True)
    """
    squeeze_batch = False
    if hand_points.dim() == 3:
        hand_points = hand_points.unsqueeze(0)
        object_points = object_points.unsqueeze(0)
        squeeze_batch = True
    
    # Fit disks for all anchors (batched) - normals computed via PCA internally
    hand_centers, hand_radii, hand_disk_normals = fit_disks_batched(hand_points)
    obj_centers, obj_radii, obj_disk_normals = fit_disks_batched(object_points)
    
    # Compute IoU matrix (batched)
    edge_matrices = compute_edge_matrix_batched(
        obj_centers=obj_centers,
        obj_radii=obj_radii,
        obj_normals=obj_disk_normals,
        hand_centers=hand_centers,
        hand_radii=hand_radii,
        hand_normals=hand_disk_normals,
        angle_threshold=angle_threshold,
        distance_threshold=distance_threshold,
        normalize_per_object=normalize_per_object,
    )
    
    if squeeze_batch:
        edge_matrices = edge_matrices.squeeze(0)
    
    return edge_matrices

