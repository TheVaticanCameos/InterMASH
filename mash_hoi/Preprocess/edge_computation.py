"""
Edge computation script for MASH parameters.

This script:
1. Loads MASH parameters from dataset
2. Denormalizes parameters
3. Converts MASH parameters to point clouds
4. Computes edge weight matrices between hand and object disk patches (normals computed via PCA internally)
5. Saves visualizations

Usage:
    python edge_computation.py
"""

import os
import numpy as np
import torch
import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from mash_hoi.source.Module.DataModule.data_dro_sphere import SphereDROTrainDataset
from mash_hoi.Preprocess.utils.mash_utils import (
    split_mash_params,
    compute_sample_points_from_mash_params,
    denormalize_params,
)
from mash_hoi.Preprocess.utils.edge_utils import process_batch
from mash_hoi.Preprocess.utils.visualize_utils import (
    visualize_edge_detailed,
    visualize_edge_matrix_heatmap,
    visualize_edge_matrix_sparse,
)


def to_multi_hand_format_no_cmap(statistics: dict) -> dict:
    """
    Convert statistics to multi-hand format without requiring cmap_params.
    
    Args:
        statistics: Raw statistics dictionary loaded from .npy file
        
    Returns:
        Dictionary with standardized key names for multi-hand training
    """
    multi_hand_statistics = {
        "mean_positions": torch.from_numpy(statistics["positions"]["mean"]),
        "std_positions": torch.from_numpy(statistics["positions"]["std"]),

        "mean_hand_mask": torch.from_numpy(statistics["hand"]["mask_params"]["mean"]),
        "std_hand_mask": torch.from_numpy(statistics["hand"]["mask_params"]["std"]),
        "mean_hand_sh": torch.from_numpy(statistics["hand"]["sh_params"]["mean"]),
        "std_hand_sh": torch.from_numpy(statistics["hand"]["sh_params"]["std"]),
        "mean_hand_ortho": torch.from_numpy(statistics["hand"]["ortho_poses"]["mean"]),
        "std_hand_ortho": torch.from_numpy(statistics["hand"]["ortho_poses"]["std"]),
        
        "mean_obj_mask": torch.from_numpy(statistics["object"]["mask_params"]["mean"]),
        "std_obj_mask": torch.from_numpy(statistics["object"]["mask_params"]["std"]),
        "mean_obj_sh": torch.from_numpy(statistics["object"]["sh_params"]["mean"]),
        "std_obj_sh": torch.from_numpy(statistics["object"]["sh_params"]["std"]),
        "mean_obj_ortho": torch.from_numpy(statistics["object"]["ortho_poses"]["mean"]),
        "std_obj_ortho": torch.from_numpy(statistics["object"]["ortho_poses"]["std"]),
    }
    
    # Add cmap_params if they exist, otherwise use dummy values
    # SphereDROTrainDataset expects mean_cmap and std_cmap keys
    if "cmap_params" in statistics.get("hand", {}):
        multi_hand_statistics["mean_hand_cmap"] = torch.from_numpy(statistics["hand"]["cmap_params"]["mean"])
        multi_hand_statistics["std_hand_cmap"] = torch.from_numpy(statistics["hand"]["cmap_params"]["std"])
        multi_hand_statistics["mean_cmap"] = torch.from_numpy(statistics["hand"]["cmap_params"]["mean"])
        multi_hand_statistics["std_cmap"] = torch.from_numpy(statistics["hand"]["cmap_params"]["std"])
    else:
        # Create dummy cmap stats (1D tensor, value doesn't matter as we don't use cmap)
        multi_hand_statistics["mean_cmap"] = torch.zeros(1)
        multi_hand_statistics["std_cmap"] = torch.ones(1)
    
    if "cmap_params" in statistics.get("object", {}):
        multi_hand_statistics["mean_obj_cmap"] = torch.from_numpy(statistics["object"]["cmap_params"]["mean"])
        multi_hand_statistics["std_obj_cmap"] = torch.from_numpy(statistics["object"]["cmap_params"]["std"])
    
    return multi_hand_statistics

# MASH Constants
MASK_DIM = 7
SH_DIM = 9
MASK_DEGREE_MAX = 3
SH_DEGREE_MAX = 2

# Sampling Constants
SAMPLE_PHI_NUM = 32
SAMPLE_THETA_NUM = 8


def mash_params_to_points(
    hand_params: torch.Tensor,
    object_params: torch.Tensor,
    positions: torch.Tensor,
    sample_phi_num: int = SAMPLE_PHI_NUM,
    sample_theta_num: int = SAMPLE_THETA_NUM,
):
    """
    Convert MASH parameters to point clouds.
    
    Args:
        hand_params: [B, N, D] DENORMALIZED hand MASH parameters
        object_params: [B, N, D] DENORMALIZED object MASH parameters
        positions: [B, N, 3] DENORMALIZED anchor positions
        sample_phi_num: Number of phi samples
        sample_theta_num: Number of theta samples
        
    Returns:
        hand_points: [B, N, S_phi, S_theta, 3] hand point clouds
        object_points: [B, N, S_phi, S_theta, 3] object point clouds
    """
    # Split parameters
    hand_mask, hand_sh, hand_ortho = split_mash_params(hand_params, MASK_DIM, SH_DIM)
    object_mask, object_sh, object_ortho = split_mash_params(object_params, MASK_DIM, SH_DIM)
    
    # Generate sample points
    hand_points = compute_sample_points_from_mash_params(
        mask_params=hand_mask,
        sh_params=hand_sh,
        ortho_poses=hand_ortho,
        positions=positions,
        sample_phi_num=sample_phi_num,
        sample_theta_num=sample_theta_num,
        mask_degree_max=MASK_DEGREE_MAX,
        sh_degree_max=SH_DEGREE_MAX,
        target="hand",
    )
    
    object_points = compute_sample_points_from_mash_params(
        mask_params=object_mask,
        sh_params=object_sh,
        ortho_poses=object_ortho,
        positions=positions,
        sample_phi_num=sample_phi_num,
        sample_theta_num=sample_theta_num,
        mask_degree_max=MASK_DEGREE_MAX,
        sh_degree_max=SH_DEGREE_MAX,
        target="object",
    )
    
    return hand_points, object_points


if __name__ == "__main__":
    # Dataset paths - modify these to your actual paths
    data_root = "/mnt/d/yxz/DATASET/HOI-Dataset/Sphere-CMapDataset"
    template_root = "/mnt/d/yxz/DATASET/template_info/"
    
    data_path = os.path.join(data_root, "shadow_train_data.h5")
    template_path = os.path.join(template_root, "mash_sphere_shadow_template.npy")
    stat_path = os.path.join(data_root, "shadow_statistics.npy")
    
    # Output directory
    output_base_dir = "/mnt/d/yxz/lym29-data/MASH-HOI/output_edge_weight_viz"
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Number of samples to process
    NUM_SAMPLES = 10
    
    stats = to_multi_hand_format_no_cmap(np.load(stat_path, allow_pickle=True)[()])
    dataset = SphereDROTrainDataset(
        data_path=data_path,
        template_mash_path=template_path,
        statistics=stats
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Processing {NUM_SAMPLES} samples...")
    print("=" * 60)
    
    # Collect samples into batch
    hand_params_list = []
    object_params_list = []
    positions_list = []
    sample_names = []
    sample_stats_list = []
    
    for sample_idx in range(min(NUM_SAMPLES, len(dataset))):
        positions, hand_params, object_params, cmap_params, template, name, sample_stats = dataset[sample_idx]
        sample_name = name.split("/")[-1].replace(".npz", "") if "/" in name else f"sample_{sample_idx}"
        
        hand_params_list.append(hand_params)
        object_params_list.append(object_params)
        positions_list.append(positions)
        sample_names.append(sample_name)
        sample_stats_list.append(sample_stats)
    
    # Stack into batch tensors (normalized) and move to CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hand_params_batch = torch.stack(hand_params_list, dim=0).to(device)  # [B, N, 22]
    object_params_batch = torch.stack(object_params_list, dim=0).to(device)  # [B, N, 22]
    positions_batch = torch.stack(positions_list, dim=0).to(device)  # [B, N, 3]
    
    print(f"\nBatch shapes (normalized) on {device}:")
    print(f"  Hand params: {hand_params_batch.shape}")
    print(f"  Object params: {object_params_batch.shape}")
    print(f"  Positions: {positions_batch.shape}")
    
    # Step 1: Denormalize parameters
    print(f"\nStep 1: Denormalizing parameters...")
    hand_params_denorm, object_params_denorm, positions_denorm, _ = denormalize_params(
        hand_params=hand_params_batch,
        object_params=object_params_batch,
        positions=positions_batch,
        statistics=sample_stats_list[0],
        mask_dim=MASK_DIM,
        sh_dim=SH_DIM,
    )
    
    # Step 2: Convert MASH parameters to point clouds
    print(f"Step 2: Converting MASH params to point clouds...")
    hand_points, object_points = mash_params_to_points(
        hand_params=hand_params_denorm,
        object_params=object_params_denorm,
        positions=positions_denorm,
        sample_phi_num=SAMPLE_PHI_NUM,
        sample_theta_num=SAMPLE_THETA_NUM,
    )
    
    print(f"  Hand points shape: {hand_points.shape}")
    print(f"  Object points shape: {object_points.shape}")
    
    # Reshape points for edge processing: [B, N, S_phi, S_theta, 3] -> [B, N, S_total, 3]
    B, N = hand_points.shape[:2]
    S_total = SAMPLE_PHI_NUM * SAMPLE_THETA_NUM
    hand_points_flat = hand_points.reshape(B, N, S_total, 3)
    object_points_flat = object_points.reshape(B, N, S_total, 3)
    
    # Step 3: Compute edge weight matrices (normals computed internally via PCA)
    print(f"Step 3: Computing edge weight matrices with process_batch()...")
    edge_weight_matrices = process_batch(
        hand_points=hand_points_flat,
        object_points=object_points_flat,
        angle_threshold=100.0,
        distance_threshold=0.05,
    )  # [B, N, N]
    
    print(f"  Edge weight matrices shape: {edge_weight_matrices.shape}")
    
    # Step 4: Visualize each sample
    print(f"\nStep 4: Visualizing results...")
    for idx, sample_name in enumerate(sample_names):
        edge_weight_matrix = edge_weight_matrices[idx]
        hand_pts = hand_points[idx]  # [N, S_phi, S_theta, 3]
        object_pts = object_points[idx]  # [N, S_phi, S_theta, 3]
        
        print(f"\n[{idx+1}/{len(sample_names)}] Visualizing: {sample_name}")
        print(f"  Edge weight stats: min={edge_weight_matrix.min().item():.4f}, max={edge_weight_matrix.max().item():.4f}, mean={edge_weight_matrix.mean().item():.4f}")
        print(f"  Non-zero entries: {(edge_weight_matrix > 0).sum().item()} / {edge_weight_matrix.numel()}")
        
        # Visualize edge weights on point clouds (normals are optional for visualization)
        output_dir = os.path.join(output_base_dir, sample_name)
        visualize_edge_detailed(
            hand_points=hand_pts,
            object_points=object_pts,
            edge_matrix=edge_weight_matrix,
            output_dir=output_dir,
            hand_normals=None,
            object_normals=None,
            colormap_name="tab20",
            edge_weight_threshold=0.0,
        )
        
        # Visualize edge weight matrix as heatmap (white=0, light green to dark green for non-zero)
        visualize_edge_matrix_heatmap(
            edge_matrix=edge_weight_matrix,
            output_path=os.path.join(output_dir, "edge_weight_matrix_heatmap.png"),
            title=f"Edge Weight Matrix - {sample_name}",
            colormap_name="white_to_green",
        )
        
        # Visualize sparse matrix view (better for sparse matrices)
        visualize_edge_matrix_sparse(
            edge_matrix=edge_weight_matrix,
            output_path=os.path.join(output_dir, "edge_weight_matrix_sparse.png"),
            title=f"Edge Weights (Sparse) - {sample_name}",
            colormap_name="white_to_green",
        )
    
    print("\n" + "=" * 60)
    print(f"All visualizations saved to: {output_base_dir}")
    print("=" * 60)
