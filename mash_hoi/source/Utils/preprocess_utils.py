"""
Preprocessing utilities for MASH parameter processing.

This module provides utility functions for:
- Splitting MASH parameters into components
- Computing sample points from MASH parameters  
- Denormalizing parameters using statistics
"""

import math
import torch
from typing import Tuple, Optional, Dict

import mash_cpp
from ma_sh.Config.constant import W0
from ma_sh.Method.rotate import compute_rotation_matrix_from_ortho6d


def split_mash_params(
    mash_params: torch.Tensor,
    mask_dim: int,
    sh_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Split MASH parameters into mask, sh, and ortho components.
    
    Args:
        mash_params: [B, N, D] or [N, D] MASH parameters where D = mask_dim + sh_dim + ortho_dim
        mask_dim: Dimension of mask parameters
        sh_dim: Dimension of spherical harmonics parameters
        
    Returns:
        mask_params: [B, N, mask_dim] or [N, mask_dim]
        sh_params: [B, N, sh_dim] or [N, sh_dim]
        ortho_poses: [B, N, ortho_dim] or [N, ortho_dim]
    """
    if mash_params.dim() == 3:
        # Batch mode
        mask_params = mash_params[:, :, :mask_dim]
        sh_params = mash_params[:, :, mask_dim:mask_dim+sh_dim]
        ortho_poses = mash_params[:, :, mask_dim+sh_dim:]
    else:
        # Single mode
        mask_params = mash_params[:, :mask_dim]
        sh_params = mash_params[:, mask_dim:mask_dim+sh_dim]
        ortho_poses = mash_params[:, mask_dim+sh_dim:]
    
    return mask_params, sh_params, ortho_poses


def compute_sample_points_from_mash_params(
    mask_params: torch.Tensor,
    sh_params: torch.Tensor,
    ortho_poses: torch.Tensor,
    positions: torch.Tensor,
    sample_phi_num: int,
    mask_degree_max: int,
    sh_degree_max: int,
    target: str = "hand",
) -> torch.Tensor:
    """
    Compute sample points from MASH parameters (pure function, no shared state).
    
    Args:
        mask_params: [B, N, mask_dim] MASH mask parameters
        sh_params: [B, N, sh_dim] MASH sh parameters
        ortho_poses: [B, N, 6] MASH ortho poses
        positions: [B, N, 3] anchor positions
        sample_phi_num: Number of phi samples
        mask_degree_max: Maximum degree for mask Fourier series
        sh_degree_max: Maximum degree for spherical harmonics
        target: "hand" or "object"
        
    Returns:
        sample_points: [B, N, S, S, 3] sample points where S = sample_phi_num
    """
    B, N = positions.shape[:2]
    S = sample_phi_num
    device = positions.device
    dtype = positions.dtype
    
    # Step 1: Generate sampling grid (phi, theta)
    sample_phis = torch.linspace(
        2 * math.pi / S, 2 * math.pi, S, dtype=dtype, device=device
    )
    sample_thetas = torch.linspace(
        math.pi / S, math.pi, S, dtype=dtype, device=device
    )
    phi_grid, theta_grid = torch.meshgrid(sample_phis, sample_thetas, indexing="ij")
    sample_phi_theta = torch.stack([phi_grid, theta_grid], dim=-1)  # [S, S, 2]
    
    # Step 2: Compute mask (theta weights) using Fourier series
    theta_weights = mask_params[:, :, 0:1].expand(-1, -1, S)  # [B, N, S]
    if mask_degree_max > 0:
        degrees = torch.arange(1, mask_degree_max + 1, dtype=dtype, device=device)
        angles = degrees.unsqueeze(1) * sample_phis.unsqueeze(0)  # [mask_degree_max, S]
        cos_terms = torch.cos(angles)  # [mask_degree_max, S]
        sin_terms = torch.sin(angles)  # [mask_degree_max, S]
        
        cos_params = mask_params[:, :, 1::2]  # [B, N, mask_degree_max]
        sin_params = mask_params[:, :, 2::2]  # [B, N, mask_degree_max]
        theta_weights = theta_weights + torch.matmul(cos_params, cos_terms)  # [B, N, S]
        theta_weights = theta_weights + torch.matmul(sin_params, sin_terms)  # [B, N, S]
    theta_weights = torch.sigmoid(theta_weights)  # [B, N, S]
    
    # Step 3: Apply mask to thetas
    phis = sample_phi_theta[:, :, 0].unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1)  # [B, N, S, S]
    thetas = sample_phi_theta[:, :, 1].unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1)  # [B, N, S, S]
    weighted_thetas = thetas * theta_weights.unsqueeze(-1)  # [B, N, S, S]
    
    # Step 4: Compute SH distances
    BP = B * N
    phis_flat = phis.reshape(BP, S, S)
    thetas_flat = weighted_thetas.reshape(BP, S, S)
    base_values = mash_cpp.toSHBaseValues(phis_flat, thetas_flat, sh_degree_max)  # [(D+1)^2, BP, S, S]
    base_values = base_values.transpose(1, 0)  # [BP, (D+1)^2, S, S]
    
    sh_params_flat = sh_params.reshape(BP, -1)  # [BP, (D+1)^2]
    distances_flat = torch.einsum("bn,bn...->b...", sh_params_flat, base_values)  # [BP, S, S]
    distances = distances_flat.reshape(B, N, S, S)  # [B, N, S, S]
    
    # Step 5: Compute directions from (phi, theta)
    sin_theta, cos_theta = torch.sin(weighted_thetas), torch.cos(weighted_thetas)
    sin_phi, cos_phi = torch.sin(phis), torch.cos(phis)
    directions = torch.stack([
        sin_theta * cos_phi,
        sin_theta * sin_phi,
        cos_theta
    ], dim=-1)  # [B, N, S, S, 3]
    
    # Step 6: Compute move vectors
    move_vectors = directions * distances.unsqueeze(-1)  # [B, N, S, S, 3]
    
    # Step 7: Apply inversion transformation
    inv_radius = W0 * sh_params[:, :, 0]  # [B, N]
    inv_centers = torch.zeros(B, N, 1, 3, dtype=dtype, device=device)
    inv_centers[:, :, 0, 2] = -inv_radius
    
    move_vectors_flat = move_vectors.reshape(B, N, -1, 3)  # [B, N, S*S, 3]
    in_inv_points = move_vectors_flat - inv_centers  # [B, N, S*S, 3]
    in_inv_norms_sq = torch.sum(in_inv_points.square(), dim=3, keepdim=True)  # [B, N, S*S, 1]
    in_inv_weights = torch.rsqrt(in_inv_norms_sq)  # [B, N, S*S, 1]
    in_inv_directions = in_inv_points * in_inv_weights  # [B, N, S*S, 3]
    
    inv_radius_sq_4 = (4.0 * inv_radius.square()).view(B, N, 1, 1)  # [B, N, 1, 1]
    in_inv_lengths = inv_radius_sq_4 * in_inv_weights  # [B, N, S*S, 1]
    inv_points = inv_centers + in_inv_lengths * in_inv_directions  # [B, N, S*S, 3]
    inv_points = inv_points.view(B, N, S, S, 3)  # [B, N, S, S, 3]
    
    # Step 8: Apply rotation and translation
    ortho_flat = ortho_poses.reshape(-1, 6)  # [B*N, 6]
    rotate_mats = compute_rotation_matrix_from_ortho6d(ortho_flat).reshape(B, N, 3, 3)  # [B, N, 3, 3]
    rotated_vectors = torch.einsum("bn...i,bnij->bn...j", inv_points, rotate_mats)  # [B, N, S, S, 3]
    
    positions_expanded = positions.view(B, N, 1, 1, 3)  # [B, N, 1, 1, 3]
    sample_points = positions_expanded + rotated_vectors  # [B, N, S, S, 3]
    
    return sample_points


def denormalize_params(
    hand_params: torch.Tensor,
    object_params: torch.Tensor,
    positions: torch.Tensor,
    statistics_tensors: Optional[Dict],
    mask_dim: int,
    sh_dim: int,
    cmap_params: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Denormalize all parameters using statistics (all at once).
    
    Args:
        hand_params: [B, N, D] normalized hand MASH parameters
        object_params: [B, N, D] normalized object MASH parameters
        positions: [B, N, 3] normalized anchor positions
        statistics_tensors: Dictionary of statistics tensors (mean, std) for denormalization
        mask_dim: Dimension of mask parameters
        sh_dim: Dimension of spherical harmonics parameters
        cmap_params: [B, N, sh_dim] normalized contact map parameters (optional)
        
    Returns:
        hand_params_denorm: [B, N, D]
        object_params_denorm: [B, N, D]
        positions_denorm: [B, N, 3]
        cmap_params_denorm: [B, N, sh_dim] or None
    """
    if statistics_tensors is None:
        return hand_params, object_params, positions, cmap_params
    
    device = hand_params.device
    dtype = hand_params.dtype
    
    # ============================================================================
    # 1. Denormalize hand_params
    # ============================================================================
    hand_mask, hand_sh, hand_ortho = split_mash_params(hand_params, mask_dim, sh_dim)
    hand_stats = statistics_tensors["hand"]
    
    hand_mask_denorm = hand_mask * hand_stats["mask_params"]["std"].to(device=device, dtype=dtype) + \
                      hand_stats["mask_params"]["mean"].to(device=device, dtype=dtype)
    hand_sh_denorm = hand_sh * hand_stats["sh_params"]["std"].to(device=device, dtype=dtype) + \
                    hand_stats["sh_params"]["mean"].to(device=device, dtype=dtype)
    hand_ortho_denorm = hand_ortho * hand_stats["ortho_poses"]["std"].to(device=device, dtype=dtype) + \
                       hand_stats["ortho_poses"]["mean"].to(device=device, dtype=dtype)
    
    hand_params_denorm = torch.cat([hand_mask_denorm, hand_sh_denorm, hand_ortho_denorm], dim=-1)
    
    # ============================================================================
    # 2. Denormalize object_params
    # ============================================================================
    object_mask, object_sh, object_ortho = split_mash_params(object_params, mask_dim, sh_dim)
    object_stats = statistics_tensors["object"]
    
    object_mask_denorm = object_mask * object_stats["mask_params"]["std"].to(device=device, dtype=dtype) + \
                        object_stats["mask_params"]["mean"].to(device=device, dtype=dtype)
    object_sh_denorm = object_sh * object_stats["sh_params"]["std"].to(device=device, dtype=dtype) + \
                      object_stats["sh_params"]["mean"].to(device=device, dtype=dtype)
    object_ortho_denorm = object_ortho * object_stats["ortho_poses"]["std"].to(device=device, dtype=dtype) + \
                         object_stats["ortho_poses"]["mean"].to(device=device, dtype=dtype)
    
    object_params_denorm = torch.cat([object_mask_denorm, object_sh_denorm, object_ortho_denorm], dim=-1)
    
    # ============================================================================
    # 3. Denormalize positions
    # ============================================================================
    pos_stats = statistics_tensors["positions"]
    positions_denorm = positions * pos_stats["std"].to(device=device, dtype=dtype) + \
                      pos_stats["mean"].to(device=device, dtype=dtype)
    
    # ============================================================================
    # 4. Denormalize cmap_params (if provided)
    # ============================================================================
    cmap_params_denorm = None
    if cmap_params is not None and "cmap_params" in statistics_tensors:
        cmap_stats = statistics_tensors["cmap_params"]
        cmap_std = cmap_stats["sh_params"]["std"].to(device=device, dtype=dtype)
        cmap_mean = cmap_stats["sh_params"]["mean"].to(device=device, dtype=dtype)
        
        # Check statistics for issues
        if torch.isnan(cmap_std).any() or torch.isinf(cmap_std).any():
            print(f"[ERROR] cmap_params std has NaN/Inf! std range: [{cmap_std.min()}, {cmap_std.max()}]")
            cmap_std = torch.nan_to_num(cmap_std, nan=1.0, posinf=1.0, neginf=1.0)
        
        if torch.isnan(cmap_mean).any() or torch.isinf(cmap_mean).any():
            print(f"[ERROR] cmap_params mean has NaN/Inf! mean range: [{cmap_mean.min()}, {cmap_mean.max()}]")
            cmap_mean = torch.nan_to_num(cmap_mean, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Check if std has zeros (would cause issues)
        if (cmap_std == 0).any():
            print(f"[WARNING] cmap_params std has zeros! Replacing with 1.0")
            cmap_std = torch.where(cmap_std == 0, torch.ones_like(cmap_std), cmap_std)
        
        # Check input cmap_params
        if torch.isnan(cmap_params).any() or torch.isinf(cmap_params).any():
            print(f"[WARNING] Input cmap_params has NaN/Inf before denormalization!")
            cmap_params = torch.nan_to_num(cmap_params, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Denormalize
        cmap_params_denorm = cmap_params * cmap_std + cmap_mean
        
        # Check for NaN/Inf after denormalization
        if torch.isnan(cmap_params_denorm).any() or torch.isinf(cmap_params_denorm).any():
            print(f"[WARNING] NaN/Inf in cmap_params_denorm after denormalization")
            print(f"  cmap_params range: [{cmap_params.min()}, {cmap_params.max()}]")
            print(f"  cmap_std range: [{cmap_std.min()}, {cmap_std.max()}]")
            print(f"  cmap_mean range: [{cmap_mean.min()}, {cmap_mean.max()}]")
            cmap_params_denorm = torch.nan_to_num(cmap_params_denorm, nan=0.0, posinf=10.0, neginf=-10.0)
    elif cmap_params is not None:
        # No statistics available, keep as is
        cmap_params_denorm = cmap_params
    
    return hand_params_denorm, object_params_denorm, positions_denorm, cmap_params_denorm

