"""
Pre-compute object point clouds and normals from MASH parameters.

This script:
1. Iterates through train and test h5 datasets
2. Extracts unique object names
3. For each unique object, computes high-resolution point cloud and WNNC normals
4. Saves results to object_pointclouds.npz for efficient loading during training

Usage:
    python mash_hoi/Preprocess/precompute_object_pointclouds.py \
        --data_root /mnt/d/yxz/DATASET/HOI-Dataset/Sphere-DexGraspNet/shadow \
        --sample_phi_num 32 \
        --sample_theta_num 8
"""

import os
import re
import argparse
from pathlib import Path
import h5py
import numpy as np
import torch
from tqdm import tqdm
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from mash_hoi.Preprocess.utils.mash_utils import compute_sample_points_from_mash_params, split_mash_params
from mash_hoi.Preprocess.utils.normal import estimate_normals_wnnc
from mash_hoi.source.Utils.constant import MASK_DIM, SH_DIM, ORTHO_DIM

# Import KNN computation from data_dro_graph
from mash_hoi.source.Module.DataModule.data_dro_graph import compute_knn_indices, K_NEIGHBORS

# Constants for MASH computation
MASK_DEGREE_MAX = 3
SH_DEGREE_MAX = 2


def extract_object_name(data_name: str) -> str:
    """
    Extract object name from data name string.
    
    Examples:
        'shadow_mujoco-Room_Essentials_Salad_Plate_Turquoise_164_hoi.npy' 
            -> 'Room_Essentials_Salad_Plate_Turquoise'
        'shadow_mujoco-SAMOA_25_hoi.npy' -> 'SAMOA'
        'SANDWICH_MEAL_100_hoi' -> 'SANDWICH_MEAL'
    
    Args:
        data_name: Original data name from h5 file
        
    Returns:
        Extracted object name
    """
    # Pattern: optional prefix ending with '-', then object name, then '_NUMBER_hoi' and optional '.npy'
    match = re.match(r'(?:.*?-)?(.*?)_\d+_hoi(?:\.npy)?$', data_name)
    if match:
        return match.group(1)
    else:
        # Fallback: use stem (remove extension)
        return Path(data_name).stem


def compute_object_pointcloud_and_normals(
    object_params_np: np.ndarray,
    positions_np: np.ndarray,
    statistics: dict,
    sample_phi_num: int = 32,
    sample_theta_num: int = 8,
    mask_degree_max: int = 3,
    sh_degree_max: int = 2,
    device: str = "cuda"
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute object point cloud and normals from MASH parameters.
    
    Args:
        object_params_np: [N, 22] raw (un-normalized) object MASH parameters from HDF5
        positions_np: [N, 3] raw (un-normalized) anchor positions from HDF5
        statistics: Statistics dictionary (unused, kept for API compatibility)
        sample_phi_num: Number of phi samples (default: 32 for high-res)
        sample_theta_num: Number of theta samples (default: 8)
        mask_degree_max: Maximum degree for mask Fourier series
        sh_degree_max: Maximum degree for spherical harmonics
        device: Device for computation
        
    Returns:
        points: [N, S, 3] structured point cloud where S=sample_phi_num*sample_theta_num
        normals: [N, S, 3] structured normals
    """
    # Use raw data directly (no denormalization needed - HDF5 stores raw data)
    # Split object parameters
    mask_params = object_params_np[:, :MASK_DIM]
    sh_params = object_params_np[:, MASK_DIM:MASK_DIM+SH_DIM]
    ortho_params = object_params_np[:, MASK_DIM+SH_DIM:MASK_DIM+SH_DIM+ORTHO_DIM]
    
    # Convert to tensors
    object_mask = torch.from_numpy(mask_params).float()
    object_sh = torch.from_numpy(sh_params).float()
    object_ortho = torch.from_numpy(ortho_params).float()
    positions = torch.from_numpy(positions_np).float()
    
    # Add batch dimension (B=1) for computation
    object_mask = object_mask.unsqueeze(0)  # [1, N, MASK_DIM]
    object_sh = object_sh.unsqueeze(0)  # [1, N, SH_DIM]
    object_ortho = object_ortho.unsqueeze(0)  # [1, N, ORTHO_DIM]
    positions = positions.unsqueeze(0)  # [1, N, 3]
    
    # Move to device
    device_obj = torch.device(device if torch.cuda.is_available() else "cpu")
    object_mask = object_mask.to(device_obj)
    object_sh = object_sh.to(device_obj)
    object_ortho = object_ortho.to(device_obj)
    positions = positions.to(device_obj)
    
    # Compute sample points
    with torch.no_grad():
        sample_points = compute_sample_points_from_mash_params(
            mask_params=object_mask,
            sh_params=object_sh,
            ortho_poses=object_ortho,
            positions=positions,
            sample_phi_num=sample_phi_num,
            sample_theta_num=sample_theta_num,
            mask_degree_max=mask_degree_max,
            sh_degree_max=sh_degree_max,
            target="object"
        )  # [1, N, sample_phi_num, sample_theta_num, 3]
        
        # Get N (number of patches)
        N = sample_points.shape[1]
        S = sample_phi_num * sample_theta_num
        
        # Flatten to [N*S, 3] for normal computation
        points_flat = sample_points.reshape(-1, 3)
        
        # Compute normals using WNNC
        normals_flat = estimate_normals_wnnc(
            points_flat,
            width_config='l1',
            iterations=40
        )  # [N*S, 3]
        
        # Reshape to structured format [N, S, 3]
        points_structured = points_flat.reshape(N, S, 3)
        normals_structured = normals_flat.reshape(N, S, 3)
    
    # Convert to numpy and move to CPU
    points_np = points_structured.cpu().numpy()
    normals_np = normals_structured.cpu().numpy()
    
    return points_np, normals_np


def process_dataset(
    data_path: str,
    statistics: dict,
    sample_phi_num: int,
    sample_theta_num: int,
    device: str,
    object_cache: dict
) -> dict:
    """
    Process a single h5 dataset and extract unique objects.
    
    Args:
        data_path: Path to h5 dataset file
        statistics: Statistics dictionary
        sample_phi_num: Number of phi samples
        sample_theta_num: Number of theta samples
        device: Device for computation
        object_cache: Dictionary to store computed objects
        
    Returns:
        Updated object_cache dictionary
    """
    print(f"\nProcessing dataset: {data_path}")
    
    if not os.path.exists(data_path):
        print(f"[WARNING] Dataset not found at {data_path}, skipping...")
        return object_cache
    
    with h5py.File(data_path, 'r') as hf:
        num_samples = len(hf.keys())
        print(f"Total samples: {num_samples}")
        
        for idx in tqdm(range(num_samples), desc="Scanning objects", disable=False):
            grp = hf[str(idx)]
            
            # Extract object name (e.g., 'shadow_mujoco-SAMOA_25_hoi.npy' -> 'SAMOA')
            data_name = grp.attrs["name"]
            object_name = extract_object_name(data_name)
            
            # Skip if already processed
            if object_name in object_cache:
                continue
            
            # Get object parameters
            object_params_np = grp["object_params"][:]
            positions_np = grp["positions"][:]
            
            # Compute point cloud and normals
            try:
                points, normals = compute_object_pointcloud_and_normals(
                    object_params_np=object_params_np,
                    positions_np=positions_np,
                    statistics=statistics,
                    sample_phi_num=sample_phi_num,
                    sample_theta_num=sample_theta_num,
                    device=device
                )
                
                # Store in cache (include positions and mash_params for first occurrence)
                object_cache[object_name] = {
                    'points': points,
                    'normals': normals,
                    'positions': positions_np,  # Store first occurrence positions for KNN computation
                    'mash_params': object_params_np  # Store for potential verification
                }
                
                tqdm.write(f"[{len(object_cache)}] Processed new object: '{object_name}' (from sample {idx}, data_name='{data_name}')")
                tqdm.write(f"     Points shape: {points.shape}, Normals shape: {normals.shape}")
                
            except Exception as e:
                tqdm.write(f"[ERROR] Failed to process object '{object_name}' (sample {idx}): {str(e)}")
                continue
    
    return object_cache


def compute_patch_centers(points: np.ndarray) -> torch.Tensor:
    """
    Compute patch centers from structured point cloud.
    
    Args:
        points: [N, S, 3] structured point cloud
        
    Returns:
        centers: [N, 3] patch centers (mean of each patch)
    """
    centers = torch.from_numpy(points).float().mean(dim=1)  # [N, 3]
    return centers


def compute_template_hand_knn(
    template_mash_path: str,
    reference_positions: np.ndarray,
    statistics: dict,
    sample_phi_num: int = 32,
    sample_theta_num: int = 8,
    device: str = "cuda"
) -> np.ndarray:
    """
    Compute KNN for template hand using reference positions.
    
    Args:
        template_mash_path: Path to template MASH file
        reference_positions: [N, 3] reference positions (from first sample)
        statistics: Statistics dictionary (unused, kept for API compatibility)
        sample_phi_num: Number of phi samples
        sample_theta_num: Number of theta samples
        device: Device for computation
        
    Returns:
        knn_indices: [N, K] KNN indices
    """
    # Load template
    template_mash = np.load(template_mash_path, allow_pickle=True)[()]
    hand_mask = template_mash["hand_params"]["mask_params"]
    hand_sh = template_mash["hand_params"]["sh_params"]
    hand_ortho = template_mash["hand_params"]["ortho_poses"]
    
    hand_params = np.concatenate([hand_mask, hand_sh, hand_ortho], axis=1)
    
    # Convert to tensors and add batch dim
    hand_params_t = torch.from_numpy(hand_params).float().unsqueeze(0)
    positions_t = torch.from_numpy(reference_positions).float().unsqueeze(0)
    
    # Move to device
    device_obj = torch.device(device if torch.cuda.is_available() else "cpu")
    hand_params_t = hand_params_t.to(device_obj)
    positions_t = positions_t.to(device_obj)
    
    # Compute sample points
    mask_params, sh_params, ortho_poses = split_mash_params(hand_params_t, MASK_DIM, SH_DIM)
    
    with torch.no_grad():
        sample_points = compute_sample_points_from_mash_params(
            mask_params=mask_params,
            sh_params=sh_params,
            ortho_poses=ortho_poses,
            positions=positions_t,
            sample_phi_num=sample_phi_num,
            sample_theta_num=sample_theta_num,
            mask_degree_max=MASK_DEGREE_MAX,
            sh_degree_max=SH_DEGREE_MAX,
            target="hand"
        )  # [1, N, sample_phi, sample_theta, 3]
        
        # Compute centers: [1, N, S, 3] -> [N, 3]
        N = sample_points.shape[1]
        S = sample_phi_num * sample_theta_num
        centers = sample_points.squeeze(0).reshape(N, S, 3).mean(dim=1)  # [N, 3]
        
        # Compute KNN
        knn_indices = compute_knn_indices(centers, k=K_NEIGHBORS)  # [N, K]
    
    return knn_indices.cpu().numpy().astype(np.int16)


def main():
    parser = argparse.ArgumentParser(description="Pre-compute object point clouds and normals")
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Root directory of the dataset (contains train_data.h5, test_data.h5, statistics.npy)"
    )
    parser.add_argument(
        "--sample_phi_num",
        type=int,
        default=32,
        help="Number of phi samples (angular direction, default: 32 for high-res)"
    )
    parser.add_argument(
        "--sample_theta_num",
        type=int,
        default=8,
        help="Number of theta samples (radial direction, default: 8)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for computation (cuda or cpu)"
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="object_pointclouds.npz",
        help="Output filename (default: object_pointclouds.npz)"
    )
    parser.add_argument(
        "--template_mash_path",
        type=str,
        default="/mnt/d/yxz/DATASET/template_info/mash_sphere_shadow_template.npy",
        help="Path to shadow hand template MASH file"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Pre-computing Object Point Clouds and Normals")
    print("=" * 80)
    print(f"Data root: {args.data_root}")
    print(f"Sampling resolution: phi={args.sample_phi_num}, theta={args.sample_theta_num}")
    print(f"Device: {args.device}")
    print("=" * 80)
    
    # Load statistics
    stat_path = os.path.join(args.data_root, "statistics.npy")
    if not os.path.exists(stat_path):
        print(f"[ERROR] Statistics file not found at {stat_path}")
        return
    
    statistics = np.load(stat_path, allow_pickle=True)[()]
    print(f"[INFO] Loaded statistics from {stat_path}")
    
    # Initialize object cache
    object_cache = {}
    
    # Process train dataset
    train_path = os.path.join(args.data_root, "train_data.h5")
    object_cache = process_dataset(
        data_path=train_path,
        statistics=statistics,
        sample_phi_num=args.sample_phi_num,
        sample_theta_num=args.sample_theta_num,
        device=args.device,
        object_cache=object_cache
    )
    
    # Process test dataset
    test_path = os.path.join(args.data_root, "test_data.h5")
    object_cache = process_dataset(
        data_path=test_path,
        statistics=statistics,
        sample_phi_num=args.sample_phi_num,
        sample_theta_num=args.sample_theta_num,
        device=args.device,
        object_cache=object_cache
    )
    
    # Save to file
    output_path = os.path.join(args.data_root, args.output_name)
    print(f"\n{'=' * 80}")
    print(f"Saving {len(object_cache)} unique objects to {output_path}")
    
    # Convert to numpy-friendly format
    save_dict = {}
    for obj_name, obj_data in object_cache.items():
        save_dict[obj_name] = {
            'points': obj_data['points'],
            'normals': obj_data['normals']
        }
    
    np.savez(output_path, **save_dict)
    print(f"[SUCCESS] Saved pre-computed data to {output_path}")
    
    # Compute and save object KNN indices
    print(f"\n{'=' * 80}")
    print("Computing object KNN indices...")
    object_knn_dict = {}
    for obj_name, obj_data in tqdm(object_cache.items(), desc="Computing object KNN"):
        # Compute patch centers
        centers = compute_patch_centers(obj_data['points'])  # [N, 3]
        
        # Compute KNN indices
        knn_indices = compute_knn_indices(centers, k=K_NEIGHBORS)  # [N, K]
        
        object_knn_dict[obj_name] = knn_indices.cpu().numpy().astype(np.int16)
    
    # Save object KNN
    object_knn_path = os.path.join(args.data_root, "object_knn.npz")
    np.savez(object_knn_path, **object_knn_dict)
    print(f"[SUCCESS] Saved object KNN to {object_knn_path}")
    
    # Compute and save template hand KNN using first sample's positions
    print(f"\n{'=' * 80}")
    print("Computing template hand KNN...")
    
    with h5py.File(train_path, 'r') as hf:
        first_positions = hf['0']['positions'][:]
    
    hand_knn = compute_template_hand_knn(
        template_mash_path=args.template_mash_path,
        reference_positions=first_positions,
        statistics=statistics,
        sample_phi_num=args.sample_phi_num,
        sample_theta_num=args.sample_theta_num,
        device=args.device
    )
    
    hand_knn_path = os.path.join(args.data_root, "template_hand_knn.npz")
    np.savez(hand_knn_path, knn_indices=hand_knn)
    print(f"[SUCCESS] Saved template hand KNN to {hand_knn_path}")
    
    # Print summary
    print(f"\n{'=' * 80}")
    print("Summary:")
    print(f"  Total unique objects: {len(object_cache)}")
    print(f"  Sample resolution: {args.sample_phi_num} x {args.sample_theta_num}")
    print(f"  Object pointclouds file size: {os.path.getsize(output_path) / (1024**2):.2f} MB")
    print(f"  Object KNN file size: {os.path.getsize(object_knn_path) / (1024**2):.2f} MB")
    print(f"  Template hand KNN file size: {os.path.getsize(hand_knn_path) / (1024**2):.2f} MB")
    print("=" * 80)
    
    # Print all object names
    if len(object_cache) > 0:
        print("\nAll processed object names:")
        for i, obj_name in enumerate(sorted(object_cache.keys()), 1):
            points_shape = object_cache[obj_name]['points'].shape
            normals_shape = object_cache[obj_name]['normals'].shape
            print(f"  {i:3d}. {obj_name:50s} - Points: {points_shape}, Normals: {normals_shape}")
        print("=" * 80)


if __name__ == "__main__":
    main()

