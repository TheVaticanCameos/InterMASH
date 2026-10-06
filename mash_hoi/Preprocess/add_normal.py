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

from ma_sh.Model.hoi_mash import HOIMash

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

    ############################## DexGraspNet [shadow] ###############################
    # data_name = "shadow_core-bowl-ad8a50aa5894bc40c762eb31f1f0c676_169"
    # data_name = "shadow_core-camera-290abe056b205c08240c46d333a693f_189"
    # data_name = "shadow_core-mug-6dd59cc1130a426571215a0b56898e5e_95"
    # data_name = "shadow_ddg-gd_egg_poisson_010_197"

    # output_dir = f"/home/yxz/MyProjects/MASH-HOI/mash_hoi/draw/{data_name}"

    # os.makedirs(output_dir, exist_ok=True)

    # hoi_path = f"/mnt/d/yxz/DATASET/HOI-Dataset/Sphere-DexGraspNet/shadow/train/{data_name}_hoi.npy"
    # hoi_mash = HOIMash.fromParamsFile(hoi_path, 32, 32, device=device)
    # hand_pts = hoi_mash.toSamplePoints("hand", method="fibonacci").reshape(-1, 3)
    # object_pts = hoi_mash.toSamplePoints("object", method="fibonacci").reshape(-1, 3)

    ############################## CMapDataset [allegro] ###############################
    # data_name = "allegro_contactdb+ps_controller_648"
    # data_name = "allegro_ycb+mini_soccer_ball_1028"

    # output_dir = f"/home/yxz/MyProjects/MASH-HOI/mash_hoi/draw/{data_name}"

    # os.makedirs(output_dir, exist_ok=True)

    # hoi_path = f"/mnt/d/yxz/DATASET/HOI-Dataset/HandCMap-Sphere-CMapDataset/allegro/train/{data_name}_hoi.npy"
    # hoi_mash = HOIMash.fromParamsFile(hoi_path, 32, 32, device=device)
    # hand_pts = hoi_mash.toSamplePoints("hand", method="fibonacci").reshape(-1, 3)
    # object_pts = hoi_mash.toSamplePoints("object", method="fibonacci").reshape(-1, 3)

    ############################## CMapDataset [barrett] ###############################
    # data_name = "barrett_contactdb+ps_controller_648"
    # data_name = "barrett_contactdb+train_935"

    # output_dir = f"/home/yxz/MyProjects/MASH-HOI/mash_hoi/draw/{data_name}"

    # os.makedirs(output_dir, exist_ok=True)

    # hoi_path = f"/mnt/d/yxz/DATASET/HOI-Dataset/HandCMap-Sphere-CMapDataset/barrett/train/{data_name}_hoi.npy"
    # hoi_mash = HOIMash.fromParamsFile(hoi_path, 32, 32, device=device)
    # hand_pts = hoi_mash.toSamplePoints("hand", method="fibonacci").reshape(-1, 3)
    # object_pts = hoi_mash.toSamplePoints("object", method="fibonacci").reshape(-1, 3)

    ############################## GRAB [mano] ###############################
    # data_name = "mano_duck_190"
    # data_name = "mano_headphones_100"

    # output_dir = f"/home/yxz/MyProjects/MASH-HOI/mash_hoi/draw/{data_name}"

    # os.makedirs(output_dir, exist_ok=True)

    # hoi_path = f"/mnt/d/yxz/DATASET/HOI-Dataset/GRAB_temp/{data_name}_hoi.npy"
    # hoi_mash = HOIMash.fromParamsFile(hoi_path, 32, 32, device=device)
    # hand_pts = hoi_mash.toSamplePoints("hand", method="fibonacci").reshape(-1, 3)
    # object_pts = hoi_mash.toSamplePoints("object", method="fibonacci").reshape(-1, 3)

    ############################## template ###############################
    output_dir = "/home/yxz/MyProjects/MASH-HOI/mash_hoi/draw/template"
    os.makedirs(output_dir, exist_ok=True)
    
    hand_type = "mano"

    template_path = f"/mnt/d/yxz/DATASET/template_info/mash_sphere_{hand_type}_template.npy"
    template_hoi = HOIMash.fromParamsFile(template_path, 32, 32, device=device)
    hand_pts = template_hoi.toSamplePoints("hand", method="fibonacci").reshape(-1, 3)
    
    
    # # Estimate normals with PyTorch3D (on GPU)
    # print("Estimating normals with PyTorch3D (GPU)...")
    # torch.cuda.synchronize() if device.type == "cuda" else None
    # t0 = time.time()
    # hand_normals_pt3d = estimate_and_orient_normals_pytorch3d(
    #     hand_pts, neighborhood_size=30, orient_outward=True
    # )
    # torch.cuda.synchronize() if device.type == "cuda" else None
    # hand_pt3d_time = time.time() - t0
    # hand_normals_pt3d = hand_normals_pt3d.cpu()
    # print(f"  PyTorch3D time: {hand_pt3d_time:.4f}s")
    
    # # Estimate normals with Open3D (on CPU)
    # print("Estimating normals with Open3D (CPU)...")
    # t0 = time.time()
    # hand_normals_o3d = estimate_and_orient_normals_open3d(
    #     hand_pts.cpu(), k=30, orient_outward=True
    # )
    # hand_o3d_time = time.time() - t0
    # print(f"  Open3D time: {hand_o3d_time:.4f}s")
    
    # Estimate normals with WNNC
    print("Estimating normals with WNNC (GPU)...")
    hand_points_wnnc = hand_pts.to(device)
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
    hand_points_np = hand_pts.cpu().numpy()
    # hand_normals_pt3d_np = hand_normals_pt3d.numpy()
    # hand_normals_o3d_np = hand_normals_o3d.numpy()
    hand_normals_wnnc_np = hand_normals_wnnc.numpy()
    
    # Save PLY files
    # save_pointcloud_with_normals(
    #     hand_points_np, hand_normals_pt3d_np,
    #     os.path.join(output_dir, "hand_pt3d.ply")
    # )
    # save_pointcloud_with_normals(
    #     hand_points_np, hand_normals_o3d_np,
    #     os.path.join(output_dir, "hand_o3d.ply")
    # )
    save_pointcloud_with_normals(
        hand_points_np, hand_normals_wnnc_np,
        os.path.join(output_dir, "hand_wnnc.ply")
    )
    
    # # Compare methods
    # print("\nHand Normal Comparison (PyTorch3D vs Open3D):")
    # cosine_sim = (hand_normals_pt3d * hand_normals_o3d).sum(dim=-1)
    # print(f"  Cosine similarity (mean): {cosine_sim.mean().item():.4f}")
    # print(f"  Cosine similarity (min):  {cosine_sim.min().item():.4f}")
    
    # print("Hand Normal Comparison (PyTorch3D vs WNNC):")
    # cosine_sim_wnnc = (hand_normals_pt3d * hand_normals_wnnc).sum(dim=-1)
    # print(f"  Cosine similarity (mean): {cosine_sim_wnnc.mean().item():.4f}")
    # print(f"  Cosine similarity (min):  {cosine_sim_wnnc.min().item():.4f}")




    
    # # =========================================================================
    # # Generate Object Point Cloud
    # # =========================================================================
    # print("\n" + "=" * 60)
    # print("Generating Object Point Cloud")
    # print("=" * 60)
    
    # # Estimate normals with PyTorch3D (on GPU)
    # print("Estimating normals with PyTorch3D (GPU)...")
    # torch.cuda.synchronize() if device.type == "cuda" else None
    # t0 = time.time()
    # # object_normals_pt3d = estimate_and_orient_normals_pytorch3d(
    # #     object_pts, neighborhood_size=30, orient_outward=True
    # # )
    # # torch.cuda.synchronize() if device.type == "cuda" else None
    # # obj_pt3d_time = time.time() - t0
    # # object_normals_pt3d = object_normals_pt3d.cpu()
    # # print(f"  PyTorch3D time: {obj_pt3d_time:.4f}s")
    
    # # # Estimate normals with Open3D (on CPU)
    # # print("Estimating normals with Open3D (CPU)...")
    # # t0 = time.time()
    # # object_normals_o3d = estimate_and_orient_normals_open3d(
    # #     object_pts.cpu(), k=30, orient_outward=True
    # # )
    # # obj_o3d_time = time.time() - t0
    # # print(f"  Open3D time: {obj_o3d_time:.4f}s")
    
    # # Estimate normals with WNNC
    # print("Estimating normals with WNNC (GPU)...")
    # object_points_wnnc = object_pts.to(device)
    # torch.cuda.synchronize() if device.type == "cuda" else None
    # t0 = time.time()
    # object_normals_wnnc = estimate_normals_wnnc(
    #     object_points_wnnc, width_config='l1', iterations=40
    # )
    # torch.cuda.synchronize() if device.type == "cuda" else None
    # obj_wnnc_time = time.time() - t0
    # object_normals_wnnc = object_normals_wnnc.cpu()
    # print(f"  WNNC time: {obj_wnnc_time:.4f}s")
    
    # # Convert to numpy for saving
    # object_points_np = object_pts.cpu().numpy()
    # # object_normals_pt3d_np = object_normals_pt3d.numpy()
    # # object_normals_o3d_np = object_normals_o3d.numpy()
    # object_normals_wnnc_np = object_normals_wnnc.numpy()
    
    # # Save PLY files
    # # save_pointcloud_with_normals(
    # #     object_points_np, object_normals_pt3d_np,
    # #     os.path.join(output_dir, "obj_pt3d.ply")
    # # )
    # # save_pointcloud_with_normals(
    # #     object_points_np, object_normals_o3d_np,
    # #     os.path.join(output_dir, "obj_o3d.ply")
    # # )
    # save_pointcloud_with_normals(
    #     object_points_np, object_normals_wnnc_np,
    #     os.path.join(output_dir, "obj_wnnc.ply")
    # )








    
    # # Compare methods
    # print("\nObject Normal Comparison (PyTorch3D vs Open3D):")
    # cosine_sim = (object_normals_pt3d * object_normals_o3d).sum(dim=-1)
    # print(f"  Cosine similarity (mean): {cosine_sim.mean().item():.4f}")
    # print(f"  Cosine similarity (min):  {cosine_sim.min().item():.4f}")
    
    # print("Object Normal Comparison (PyTorch3D vs WNNC):")
    # cosine_sim_wnnc = (object_normals_pt3d * object_normals_wnnc).sum(dim=-1)
    # print(f"  Cosine similarity (mean): {cosine_sim_wnnc.mean().item():.4f}")
    # print(f"  Cosine similarity (min):  {cosine_sim_wnnc.min().item():.4f}")
    
    # =========================================================================
    # Summary
    # =========================================================================
    # print("\n" + "=" * 60)
    # print("Summary")
    # print("=" * 60)
    # print(f"Generated PLY files in: {output_dir}")
    # print("Files:")
    # print("  - hand_pt3d.ply  : Hand point cloud with PyTorch3D normals")
    # print("  - hand_o3d.ply   : Hand point cloud with Open3D normals")
    # print("  - hand_wnnc.ply  : Hand point cloud with WNNC normals")
    # print("  - obj_pt3d.ply   : Object point cloud with PyTorch3D normals")
    # print("  - obj_o3d.ply    : Object point cloud with Open3D normals")
    # print("  - obj_wnnc.ply   : Object point cloud with WNNC normals")
    # print("\nColors in PLY files represent normal directions (RGB = normalized XYZ)")
    
    # print("\n" + "-" * 60)
    # print("Timing Summary")
    # print("-" * 60)
    # print(f"Hand point cloud ({hand_pts.shape[0]} points):")
    # print(f"  PyTorch3D: {hand_pt3d_time:.4f}s")
    # print(f"  Open3D:    {hand_o3d_time:.4f}s")
    # print(f"  WNNC:      {hand_wnnc_time:.4f}s")
    # print(f"Object point cloud ({object_pts.shape[0]} points):")
    # print(f"  PyTorch3D: {obj_pt3d_time:.4f}s")
    # print(f"  Open3D:    {obj_o3d_time:.4f}s")
    # print(f"  WNNC:      {obj_wnnc_time:.4f}s")
    # print(f"\nTotal PyTorch3D: {hand_pt3d_time + obj_pt3d_time:.4f}s")
    # print(f"Total Open3D:    {hand_o3d_time + obj_o3d_time:.4f}s")
    # print(f"Total WNNC:      {hand_wnnc_time + obj_wnnc_time:.4f}s")
    # print("=" * 60)


if __name__ == "__main__":
    main()

