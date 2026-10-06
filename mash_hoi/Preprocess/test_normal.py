"""
Test script for normal estimation methods.

This script loads real MASH data, generates point clouds, 
estimates normals using PyTorch3D and Open3D methods,
and saves the results as PLY files for visualization.
"""

import os
import time
import numpy as np
import torch
import open3d as o3d
import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from mash_hoi.source.Module.DataModule.data_dro_sphere import SphereDROTrainDataset
from mash_hoi.source.Module.DataModule.utils_statistics import to_multi_hand_format
from mash_hoi.Preprocess.utils import (
    compute_sample_points_from_mash_params,
    estimate_and_orient_normals_pytorch3d,
    estimate_and_orient_normals_open3d,
    estimate_normals_wnnc,
)


# Constants
MASK_DIM = 7
SH_DIM = 9
MASK_DEGREE_MAX = 3
SH_DEGREE_MAX = 2
SAMPLE_PHI_NUM = 32
SAMPLE_THETA_NUM = 8


def save_pointcloud_with_normals(
    points: np.ndarray,
    normals: np.ndarray,
    filepath: str,
):
    """
    Save point cloud with normals as PLY file.
    
    Args:
        points: [N, 3] point positions
        normals: [N, 3] point normals
        filepath: Output PLY file path
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    
    # Add color based on normal direction for visualization
    # Map normal components to RGB
    colors = (normals + 1) / 2  # Map [-1, 1] to [0, 1]
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    o3d.io.write_point_cloud(filepath, pcd)
    print(f"Saved: {filepath}")


def main():
    print("=" * 60)
    print("Testing Normal Estimation Methods with Real MASH Data")
    print("=" * 60)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Dataset paths - modify these to your actual paths
    # data_root = "/data/yxz/MyProjects/DATASET/HOI-Dataset/Sphere-CMapDataset"
    data_root = "/mnt/d/yxz/DATASET/HOI-Dataset/HandCMap-Sphere-CMapDataset"
    template_root = "/mnt/d/yxz/DATASET/template_info"

    hand_type = "allegro"
    
    data_path = os.path.join(data_root, f"{hand_type}_train_data.h5")
    template_path = os.path.join(template_root, f"mash_sphere_{hand_type}_template.npy")
    stat_path = os.path.join(data_root, f"{hand_type}_statistics.npy")
    
    # Output directory
    # output_dir = "/data/lym/MASH-HOI/test_normal"
    output_dir = "/home/yxz/MyProjects/MASH-HOI/mash_hoi/draw"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    if not os.path.exists(data_path):
        print(f"[ERROR] Dataset not found at {data_path}")
        print("Please modify the data_root path in this script.")
        return
    
    # Load dataset
    print(f"\nLoading dataset from: {data_path}")
    stats = to_multi_hand_format(np.load(stat_path, allow_pickle=True)[()])
    dataset = SphereDROTrainDataset(
        data_path=data_path,
        template_mash_path=template_path,
        statistics=stats
    )
    
    # Get first sample
    positions, hand_params, object_params, cmap_params, template, name, stats = dataset[0]
    sample_name = name.split("/")[-1].replace(".npz", "") if "/" in name else name
    print(f"Processing sample: {sample_name}")
    print(f"Hand params shape: {hand_params.shape}")
    print(f"Object params shape: {object_params.shape}")
    print(f"Positions shape: {positions.shape}")
    
    # Add batch dimension
    hand_params = hand_params.unsqueeze(0)
    object_params = object_params.unsqueeze(0)
    positions = positions.unsqueeze(0)
    
    # Denormalize positions
    mean_pos = stats["mean_positions"]
    std_pos = stats["std_positions"]
    positions_denorm = positions * std_pos + mean_pos
    
    # Denormalize hand params
    mean_hand_mask = stats["mean_hand_mask"]
    std_hand_mask = stats["std_hand_mask"]
    mean_hand_sh = stats["mean_hand_sh"]
    std_hand_sh = stats["std_hand_sh"]
    mean_hand_ortho = stats["mean_hand_ortho"]
    std_hand_ortho = stats["std_hand_ortho"]
    
    hand_mask = hand_params[:, :, :MASK_DIM] * std_hand_mask + mean_hand_mask
    hand_sh = hand_params[:, :, MASK_DIM:MASK_DIM+SH_DIM] * std_hand_sh + mean_hand_sh
    hand_ortho = hand_params[:, :, MASK_DIM+SH_DIM:] * std_hand_ortho + mean_hand_ortho
    
    # Denormalize object params
    mean_obj_mask = stats["mean_obj_mask"]
    std_obj_mask = stats["std_obj_mask"]
    mean_obj_sh = stats["mean_obj_sh"]
    std_obj_sh = stats["std_obj_sh"]
    mean_obj_ortho = stats["mean_obj_ortho"]
    std_obj_ortho = stats["std_obj_ortho"]
    
    object_mask = object_params[:, :, :MASK_DIM] * std_obj_mask + mean_obj_mask
    object_sh = object_params[:, :, MASK_DIM:MASK_DIM+SH_DIM] * std_obj_sh + mean_obj_sh
    object_ortho = object_params[:, :, MASK_DIM+SH_DIM:] * std_obj_ortho + mean_obj_ortho
    
    # =========================================================================
    # Generate Hand Point Cloud
    # =========================================================================
    print("\n" + "=" * 60)
    print("Generating Hand Point Cloud")
    print("=" * 60)
    
    hand_points = compute_sample_points_from_mash_params(
        mask_params=hand_mask,
        sh_params=hand_sh,
        ortho_poses=hand_ortho,
        positions=positions_denorm,
        sample_phi_num=SAMPLE_PHI_NUM,
        sample_theta_num=SAMPLE_THETA_NUM,
        mask_degree_max=MASK_DEGREE_MAX,
        sh_degree_max=SH_DEGREE_MAX,
        target="hand",
    )
    
    hand_points_flat = hand_points.squeeze(0).reshape(-1, 3)
    print(f"Hand point cloud shape: {hand_points_flat.shape}")
    
    # Estimate normals with PyTorch3D (on GPU)
    print("Estimating normals with PyTorch3D (GPU)...")
    hand_points_gpu = hand_points_flat.to(device)
    torch.cuda.synchronize() if device.type == "cuda" else None
    t0 = time.time()
    hand_normals_pt3d = estimate_and_orient_normals_pytorch3d(
        hand_points_gpu, neighborhood_size=30, orient_outward=True
    )
    torch.cuda.synchronize() if device.type == "cuda" else None
    hand_pt3d_time = time.time() - t0
    hand_normals_pt3d = hand_normals_pt3d.cpu()
    print(f"  PyTorch3D time: {hand_pt3d_time:.4f}s")
    
    # Estimate normals with Open3D (on CPU)
    print("Estimating normals with Open3D (CPU)...")
    t0 = time.time()
    hand_normals_o3d = estimate_and_orient_normals_open3d(
        hand_points_flat.cpu(), k=30, orient_outward=True
    )
    hand_o3d_time = time.time() - t0
    print(f"  Open3D time: {hand_o3d_time:.4f}s")
    
    # Estimate normals with WNNC
    print("Estimating normals with WNNC (GPU)...")
    hand_points_wnnc = hand_points_flat.to(device)
    torch.cuda.synchronize() if device.type == "cuda" else None
    t0 = time.time()
    hand_normals_wnnc = estimate_normals_wnnc(
        hand_points_wnnc, width_config='l1', iterations=40
    )
    torch.cuda.synchronize() if device.type == "cuda" else None
    hand_wnnc_time = time.time() - t0
    hand_normals_wnnc = hand_normals_wnnc.cpu()
    print(f"  WNNC time: {hand_wnnc_time:.4f}s")
    
    # Convert to numpy for saving
    hand_points_np = hand_points_flat.cpu().numpy()
    hand_normals_pt3d_np = hand_normals_pt3d.numpy()
    hand_normals_o3d_np = hand_normals_o3d.numpy()
    hand_normals_wnnc_np = hand_normals_wnnc.numpy()
    
    # Save PLY files
    save_pointcloud_with_normals(
        hand_points_np, hand_normals_pt3d_np,
        os.path.join(output_dir, "hand_pt3d.ply")
    )
    save_pointcloud_with_normals(
        hand_points_np, hand_normals_o3d_np,
        os.path.join(output_dir, "hand_o3d.ply")
    )
    save_pointcloud_with_normals(
        hand_points_np, hand_normals_wnnc_np,
        os.path.join(output_dir, "hand_wnnc.ply")
    )
    
    # Compare methods
    print("\nHand Normal Comparison (PyTorch3D vs Open3D):")
    cosine_sim = (hand_normals_pt3d * hand_normals_o3d).sum(dim=-1)
    print(f"  Cosine similarity (mean): {cosine_sim.mean().item():.4f}")
    print(f"  Cosine similarity (min):  {cosine_sim.min().item():.4f}")
    
    print("Hand Normal Comparison (PyTorch3D vs WNNC):")
    cosine_sim_wnnc = (hand_normals_pt3d * hand_normals_wnnc).sum(dim=-1)
    print(f"  Cosine similarity (mean): {cosine_sim_wnnc.mean().item():.4f}")
    print(f"  Cosine similarity (min):  {cosine_sim_wnnc.min().item():.4f}")
    
    # =========================================================================
    # Generate Object Point Cloud
    # =========================================================================
    print("\n" + "=" * 60)
    print("Generating Object Point Cloud")
    print("=" * 60)
    
    object_points = compute_sample_points_from_mash_params(
        mask_params=object_mask,
        sh_params=object_sh,
        ortho_poses=object_ortho,
        positions=positions_denorm,
        sample_phi_num=SAMPLE_PHI_NUM,
        sample_theta_num=SAMPLE_THETA_NUM,
        mask_degree_max=MASK_DEGREE_MAX,
        sh_degree_max=SH_DEGREE_MAX,
        target="object",
    )
    
    object_points_flat = object_points.squeeze(0).reshape(-1, 3)
    print(f"Object point cloud shape: {object_points_flat.shape}")
    
    # Estimate normals with PyTorch3D (on GPU)
    print("Estimating normals with PyTorch3D (GPU)...")
    object_points_gpu = object_points_flat.to(device)
    torch.cuda.synchronize() if device.type == "cuda" else None
    t0 = time.time()
    object_normals_pt3d = estimate_and_orient_normals_pytorch3d(
        object_points_gpu, neighborhood_size=30, orient_outward=True
    )
    torch.cuda.synchronize() if device.type == "cuda" else None
    obj_pt3d_time = time.time() - t0
    object_normals_pt3d = object_normals_pt3d.cpu()
    print(f"  PyTorch3D time: {obj_pt3d_time:.4f}s")
    
    # Estimate normals with Open3D (on CPU)
    print("Estimating normals with Open3D (CPU)...")
    t0 = time.time()
    object_normals_o3d = estimate_and_orient_normals_open3d(
        object_points_flat.cpu(), k=30, orient_outward=True
    )
    obj_o3d_time = time.time() - t0
    print(f"  Open3D time: {obj_o3d_time:.4f}s")
    
    # Estimate normals with WNNC
    print("Estimating normals with WNNC (GPU)...")
    object_points_wnnc = object_points_flat.to(device)
    torch.cuda.synchronize() if device.type == "cuda" else None
    t0 = time.time()
    object_normals_wnnc = estimate_normals_wnnc(
        object_points_wnnc, width_config='l1', iterations=40
    )
    torch.cuda.synchronize() if device.type == "cuda" else None
    obj_wnnc_time = time.time() - t0
    object_normals_wnnc = object_normals_wnnc.cpu()
    print(f"  WNNC time: {obj_wnnc_time:.4f}s")
    
    # Convert to numpy for saving
    object_points_np = object_points_flat.cpu().numpy()
    object_normals_pt3d_np = object_normals_pt3d.numpy()
    object_normals_o3d_np = object_normals_o3d.numpy()
    object_normals_wnnc_np = object_normals_wnnc.numpy()
    
    # Save PLY files
    save_pointcloud_with_normals(
        object_points_np, object_normals_pt3d_np,
        os.path.join(output_dir, "obj_pt3d.ply")
    )
    save_pointcloud_with_normals(
        object_points_np, object_normals_o3d_np,
        os.path.join(output_dir, "obj_o3d.ply")
    )
    save_pointcloud_with_normals(
        object_points_np, object_normals_wnnc_np,
        os.path.join(output_dir, "obj_wnnc.ply")
    )
    
    # Compare methods
    print("\nObject Normal Comparison (PyTorch3D vs Open3D):")
    cosine_sim = (object_normals_pt3d * object_normals_o3d).sum(dim=-1)
    print(f"  Cosine similarity (mean): {cosine_sim.mean().item():.4f}")
    print(f"  Cosine similarity (min):  {cosine_sim.min().item():.4f}")
    
    print("Object Normal Comparison (PyTorch3D vs WNNC):")
    cosine_sim_wnnc = (object_normals_pt3d * object_normals_wnnc).sum(dim=-1)
    print(f"  Cosine similarity (mean): {cosine_sim_wnnc.mean().item():.4f}")
    print(f"  Cosine similarity (min):  {cosine_sim_wnnc.min().item():.4f}")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Generated PLY files in: {output_dir}")
    print("Files:")
    print("  - hand_pt3d.ply  : Hand point cloud with PyTorch3D normals")
    print("  - hand_o3d.ply   : Hand point cloud with Open3D normals")
    print("  - hand_wnnc.ply  : Hand point cloud with WNNC normals")
    print("  - obj_pt3d.ply   : Object point cloud with PyTorch3D normals")
    print("  - obj_o3d.ply    : Object point cloud with Open3D normals")
    print("  - obj_wnnc.ply   : Object point cloud with WNNC normals")
    print("\nColors in PLY files represent normal directions (RGB = normalized XYZ)")
    
    print("\n" + "-" * 60)
    print("Timing Summary")
    print("-" * 60)
    print(f"Hand point cloud ({hand_points_flat.shape[0]} points):")
    print(f"  PyTorch3D: {hand_pt3d_time:.4f}s")
    print(f"  Open3D:    {hand_o3d_time:.4f}s")
    print(f"  WNNC:      {hand_wnnc_time:.4f}s")
    print(f"Object point cloud ({object_points_flat.shape[0]} points):")
    print(f"  PyTorch3D: {obj_pt3d_time:.4f}s")
    print(f"  Open3D:    {obj_o3d_time:.4f}s")
    print(f"  WNNC:      {obj_wnnc_time:.4f}s")
    print(f"\nTotal PyTorch3D: {hand_pt3d_time + obj_pt3d_time:.4f}s")
    print(f"Total Open3D:    {hand_o3d_time + obj_o3d_time:.4f}s")
    print(f"Total WNNC:      {hand_wnnc_time + obj_wnnc_time:.4f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()

