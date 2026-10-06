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
import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

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
    sample_theta_num: int,
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
        sample_phi_num: Number of phi samples (angular direction around axis)
        mask_degree_max: Maximum degree for mask Fourier series
        sh_degree_max: Maximum degree for spherical harmonics
        target: "hand" or "object"
        sample_theta_num: Number of theta samples (radial direction, controls radius sampling density)
        
    Returns:
        sample_points: [B, N, S_phi, S_theta, 3] sample points
    """
    B, N = positions.shape[:2]
    S_phi = sample_phi_num
    S_theta = sample_theta_num
    device = positions.device
    dtype = positions.dtype
    
    # Step 1: Generate sampling grid (phi, theta)
    sample_phis = torch.linspace(
        2 * math.pi / S_phi, 2 * math.pi, S_phi, dtype=dtype, device=device
    )
    sample_thetas = torch.linspace(
        math.pi / S_theta, math.pi, S_theta, dtype=dtype, device=device
    )
    phi_grid, theta_grid = torch.meshgrid(sample_phis, sample_thetas, indexing="ij")
    sample_phi_theta = torch.stack([phi_grid, theta_grid], dim=-1)  # [S_phi, S_theta, 2]
    
    # Step 2: Compute mask (theta weights) using Fourier series
    theta_weights = mask_params[:, :, 0:1].expand(-1, -1, S_phi)  # [B, N, S_phi]
    if mask_degree_max > 0:
        degrees = torch.arange(1, mask_degree_max + 1, dtype=dtype, device=device)
        angles = degrees.unsqueeze(1) * sample_phis.unsqueeze(0)  # [mask_degree_max, S_phi]
        cos_terms = torch.cos(angles)  # [mask_degree_max, S_phi]
        sin_terms = torch.sin(angles)  # [mask_degree_max, S_phi]
        
        cos_params = mask_params[:, :, 1::2]  # [B, N, mask_degree_max]
        sin_params = mask_params[:, :, 2::2]  # [B, N, mask_degree_max]
        theta_weights = theta_weights + torch.matmul(cos_params, cos_terms)  # [B, N, S_phi]
        theta_weights = theta_weights + torch.matmul(sin_params, sin_terms)  # [B, N, S_phi]
    theta_weights = torch.sigmoid(theta_weights)  # [B, N, S_phi]
    
    # Step 3: Apply mask to thetas
    phis = sample_phi_theta[:, :, 0].unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1)  # [B, N, S_phi, S_theta]
    thetas = sample_phi_theta[:, :, 1].unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1)  # [B, N, S_phi, S_theta]
    weighted_thetas = thetas * theta_weights.unsqueeze(-1)  # [B, N, S_phi, S_theta]
    
    # Step 4: Compute SH distances
    BP = B * N
    phis_flat = phis.reshape(BP, S_phi, S_theta)
    thetas_flat = weighted_thetas.reshape(BP, S_phi, S_theta)
    base_values = mash_cpp.toSHBaseValues(phis_flat, thetas_flat, sh_degree_max)  # [(D+1)^2, BP, S_phi, S_theta]
    base_values = base_values.transpose(1, 0)  # [BP, (D+1)^2, S_phi, S_theta]
    
    sh_params_flat = sh_params.reshape(BP, -1)  # [BP, (D+1)^2]
    distances_flat = torch.einsum("bn,bn...->b...", sh_params_flat, base_values)  # [BP, S_phi, S_theta]
    distances = distances_flat.reshape(B, N, S_phi, S_theta)  # [B, N, S_phi, S_theta]
    
    # Step 5: Compute directions from (phi, theta)
    sin_theta, cos_theta = torch.sin(weighted_thetas), torch.cos(weighted_thetas)
    sin_phi, cos_phi = torch.sin(phis), torch.cos(phis)
    directions = torch.stack([
        sin_theta * cos_phi,
        sin_theta * sin_phi,
        cos_theta
    ], dim=-1)  # [B, N, S_phi, S_theta, 3]
    
    # Step 6: Compute move vectors
    move_vectors = directions * distances.unsqueeze(-1)  # [B, N, S_phi, S_theta, 3]
    
    # Step 7: Apply inversion transformation
    inv_radius = W0 * sh_params[:, :, 0]  # [B, N]
    inv_centers = torch.zeros(B, N, 1, 3, dtype=dtype, device=device)
    inv_centers[:, :, 0, 2] = -inv_radius
    
    move_vectors_flat = move_vectors.reshape(B, N, -1, 3)  # [B, N, S_phi*S_theta, 3]
    in_inv_points = move_vectors_flat - inv_centers  # [B, N, S_phi*S_theta, 3]
    in_inv_norms_sq = torch.sum(in_inv_points.square(), dim=3, keepdim=True)  # [B, N, S_phi*S_theta, 1]
    in_inv_weights = torch.rsqrt(in_inv_norms_sq)  # [B, N, S_phi*S_theta, 1]
    in_inv_directions = in_inv_points * in_inv_weights  # [B, N, S_phi*S_theta, 3]
    
    inv_radius_sq_4 = (4.0 * inv_radius.square()).view(B, N, 1, 1)  # [B, N, 1, 1]
    in_inv_lengths = inv_radius_sq_4 * in_inv_weights  # [B, N, S_phi*S_theta, 1]
    inv_points = inv_centers + in_inv_lengths * in_inv_directions  # [B, N, S_phi*S_theta, 3]
    inv_points = inv_points.view(B, N, S_phi, S_theta, 3)  # [B, N, S_phi, S_theta, 3]
    
    # Step 8: Apply rotation and translation
    ortho_flat = ortho_poses.reshape(-1, 6)  # [B*N, 6]
    rotate_mats = compute_rotation_matrix_from_ortho6d(ortho_flat).reshape(B, N, 3, 3)  # [B, N, 3, 3]
    rotated_vectors = torch.einsum("bn...i,bnij->bn...j", inv_points, rotate_mats)  # [B, N, S_phi, S_theta, 3]
    
    positions_expanded = positions.view(B, N, 1, 1, 3)  # [B, N, 1, 1, 3]
    sample_points = positions_expanded + rotated_vectors  # [B, N, S_phi, S_theta, 3]
    
    return sample_points


def _is_flat_statistics_format(statistics: Dict) -> bool:
    """Check if statistics uses flat format (from to_multi_hand_format)."""
    return "mean_hand_mask" in statistics or "mean_positions" in statistics


def _get_stats_for_entity(
    statistics: Dict,
    entity: str,  # "hand" or "object"
    component: str,  # "mask", "sh", "ortho"
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get mean and std for a specific entity/component, handling both statistics formats.
    
    Returns:
        (mean, std) tensors
    """
    if _is_flat_statistics_format(statistics):
        # Flat format: "mean_hand_mask", "std_hand_mask", etc.
        prefix = "hand" if entity == "hand" else "obj"
        mean = statistics[f"mean_{prefix}_{component}"].to(device=device, dtype=dtype)
        std = statistics[f"std_{prefix}_{component}"].to(device=device, dtype=dtype)
    else:
        # Nested format: statistics["hand"]["mask_params"]["mean"], etc.
        component_key = f"{component}_params" if component != "ortho" else "ortho_poses"
        mean = statistics[entity][component_key]["mean"].to(device=device, dtype=dtype)
        std = statistics[entity][component_key]["std"].to(device=device, dtype=dtype)
    return mean, std


def _get_position_stats(
    statistics: Dict,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get mean and std for positions, handling both statistics formats."""
    if _is_flat_statistics_format(statistics):
        mean = statistics["mean_positions"].to(device=device, dtype=dtype)
        std = statistics["std_positions"].to(device=device, dtype=dtype)
    else:
        mean = statistics["positions"]["mean"].to(device=device, dtype=dtype)
        std = statistics["positions"]["std"].to(device=device, dtype=dtype)
    return mean, std


def denormalize_params(
    hand_params: Optional[torch.Tensor],
    object_params: Optional[torch.Tensor],
    positions: torch.Tensor,
    statistics: Optional[Dict],
    mask_dim: int = 7,
    sh_dim: int = 9,
    cmap_params: Optional[torch.Tensor] = None,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
    """
    Denormalize MASH parameters using statistics.
    
    Supports both statistics formats:
    - Nested format: statistics["hand"]["mask_params"]["mean"], etc.
    - Flat format: statistics["mean_hand_mask"], etc. (from to_multi_hand_format)
    
    Args:
        hand_params: [B, N, D] or [N, D] normalized hand MASH parameters (optional)
        object_params: [B, N, D] or [N, D] normalized object MASH parameters (optional)
        positions: [B, N, 3] or [N, 3] normalized anchor positions
        statistics: Dictionary of statistics tensors for denormalization
        mask_dim: Dimension of mask parameters (default: 7)
        sh_dim: Dimension of spherical harmonics parameters (default: 9)
        cmap_params: [B, N, sh_dim] normalized contact map parameters (optional)
        
    Returns:
        hand_params_denorm: same shape as input, or None if input was None
        object_params_denorm: same shape as input, or None if input was None
        positions_denorm: same shape as input
        cmap_params_denorm: same shape as input, or None if input was None
    """
    if statistics is None:
        return hand_params, object_params, positions, cmap_params
    
    # Get device and dtype from first available tensor
    ref_tensor = hand_params if hand_params is not None else (object_params if object_params is not None else positions)
    device = ref_tensor.device
    dtype = ref_tensor.dtype
    
    # ============================================================================
    # 1. Denormalize positions
    # ============================================================================
    pos_mean, pos_std = _get_position_stats(statistics, device, dtype)
    positions_denorm = positions * pos_std + pos_mean
    
    # ============================================================================
    # 2. Denormalize hand_params (if provided)
    # ============================================================================
    hand_params_denorm = None
    if hand_params is not None:
        hand_mask, hand_sh, hand_ortho = split_mash_params(hand_params, mask_dim, sh_dim)
        
        mask_mean, mask_std = _get_stats_for_entity(statistics, "hand", "mask", device, dtype)
        sh_mean, sh_std = _get_stats_for_entity(statistics, "hand", "sh", device, dtype)
        ortho_mean, ortho_std = _get_stats_for_entity(statistics, "hand", "ortho", device, dtype)
        
        hand_mask_denorm = hand_mask * mask_std + mask_mean
        hand_sh_denorm = hand_sh * sh_std + sh_mean
        hand_ortho_denorm = hand_ortho * ortho_std + ortho_mean
        
        hand_params_denorm = torch.cat([hand_mask_denorm, hand_sh_denorm, hand_ortho_denorm], dim=-1)
    
    # ============================================================================
    # 3. Denormalize object_params (if provided)
    # ============================================================================
    object_params_denorm = None
    if object_params is not None:
        object_mask, object_sh, object_ortho = split_mash_params(object_params, mask_dim, sh_dim)
        
        mask_mean, mask_std = _get_stats_for_entity(statistics, "object", "mask", device, dtype)
        sh_mean, sh_std = _get_stats_for_entity(statistics, "object", "sh", device, dtype)
        ortho_mean, ortho_std = _get_stats_for_entity(statistics, "object", "ortho", device, dtype)
        
        object_mask_denorm = object_mask * mask_std + mask_mean
        object_sh_denorm = object_sh * sh_std + sh_mean
        object_ortho_denorm = object_ortho * ortho_std + ortho_mean
        
        object_params_denorm = torch.cat([object_mask_denorm, object_sh_denorm, object_ortho_denorm], dim=-1)
    
    # ============================================================================
    # 4. Denormalize cmap_params (if provided)
    # ============================================================================
    cmap_params_denorm = None
    if cmap_params is not None:
        if _is_flat_statistics_format(statistics):
            # Flat format doesn't have cmap stats typically, keep as is
            cmap_params_denorm = cmap_params
        elif "cmap_params" in statistics:
            cmap_stats = statistics["cmap_params"]
            cmap_std = cmap_stats["sh_params"]["std"].to(device=device, dtype=dtype)
            cmap_mean = cmap_stats["sh_params"]["mean"].to(device=device, dtype=dtype)
            
            # Safety checks
            if torch.isnan(cmap_std).any() or torch.isinf(cmap_std).any():
                cmap_std = torch.nan_to_num(cmap_std, nan=1.0, posinf=1.0, neginf=1.0)
            if torch.isnan(cmap_mean).any() or torch.isinf(cmap_mean).any():
                cmap_mean = torch.nan_to_num(cmap_mean, nan=0.0, posinf=0.0, neginf=0.0)
            if (cmap_std == 0).any():
                cmap_std = torch.where(cmap_std == 0, torch.ones_like(cmap_std), cmap_std)
            if torch.isnan(cmap_params).any() or torch.isinf(cmap_params).any():
                cmap_params = torch.nan_to_num(cmap_params, nan=0.0, posinf=1.0, neginf=-1.0)
            
            cmap_params_denorm = cmap_params * cmap_std + cmap_mean
            
            if torch.isnan(cmap_params_denorm).any() or torch.isinf(cmap_params_denorm).any():
                cmap_params_denorm = torch.nan_to_num(cmap_params_denorm, nan=0.0, posinf=10.0, neginf=-10.0)
        else:
            cmap_params_denorm = cmap_params
    
    return hand_params_denorm, object_params_denorm, positions_denorm, cmap_params_denorm
