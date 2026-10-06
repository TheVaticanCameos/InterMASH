"""
Test script for pre-computed object point clouds and normals.

This script loads pre-computed object data from the dataset and saves
sample point clouds with normals as PLY files for visualization.

Usage:
    python mash_hoi/Preprocess/test_precompute.py \
        --data_root /mnt/d/yxz/DATASET/HOI-Dataset/Sphere-DexGraspNet/shadow \
        --output_dir ./test_precompute_output \
        --num_samples 5
"""

import os
import argparse
import numpy as np
import open3d as o3d
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


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
    parser = argparse.ArgumentParser(
        description="Save pre-computed object point clouds as PLY files for visualization"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Root directory of the dataset (contains object_pointclouds.npz)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./test_precompute_output",
        help="Output directory for PLY files (default: ./test_precompute_output)"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of sample objects to save (default: 5)"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Testing Pre-computed Object Point Clouds")
    print("=" * 80)
    print(f"Data root: {args.data_root}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 80)
    
    # Load pre-computed data
    precomputed_path = os.path.join(args.data_root, "object_pointclouds.npz")
    
    if not os.path.exists(precomputed_path):
        print(f"[ERROR] Pre-computed data not found at {precomputed_path}")
        print(f"Please run: python mash_hoi/Preprocess/precompute_object_pointclouds.py --data_root {args.data_root}")
        return 1
    
    print(f"Loading pre-computed data from: {precomputed_path}")
    data = np.load(precomputed_path, allow_pickle=True)
    
    object_names = list(data.keys())
    num_objects = len(object_names)
    print(f"Total objects in cache: {num_objects}")
    
    if num_objects == 0:
        print("[ERROR] No objects found in pre-computed data")
        return 1
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")
    
    # Save sample objects
    num_to_save = min(args.num_samples, num_objects)
    print(f"\nSaving {num_to_save} sample objects as PLY files...")
    print("=" * 80)
    
    for i, obj_name in enumerate(object_names[:num_to_save]):
        print(f"\n[{i+1}/{num_to_save}] Processing object: {obj_name}")
        
        # Load object data
        obj_data = data[obj_name].item()
        points_structured = obj_data['points']  # [N, S, 3]
        normals_structured = obj_data['normals']  # [N, S, 3]
        
        print(f"  - Points structured shape: {points_structured.shape} [N, S, 3]")
        print(f"  - Normals structured shape: {normals_structured.shape} [N, S, 3]")
        print(f"    * N (patches): {points_structured.shape[0]}")
        print(f"    * S (samples per patch): {points_structured.shape[1]}")
        
        # Flatten for visualization: [N, S, 3] -> [N*S, 3]
        points = points_structured.reshape(-1, 3)
        normals = normals_structured.reshape(-1, 3)
        
        print(f"  - Flattened points shape: {points.shape}")
        print(f"  - Flattened normals shape: {normals.shape}")
        
        # Verify normals are unit vectors
        norms = np.linalg.norm(normals, axis=1)
        print(f"  - Normal magnitudes: min={norms.min():.4f}, max={norms.max():.4f}, mean={norms.mean():.4f}")
        
        if not np.allclose(norms, 1.0, atol=0.01):
            print(f"  [WARNING] Normals are not unit vectors!")
        
        # Compute point cloud statistics
        points_min = points.min(axis=0)
        points_max = points.max(axis=0)
        points_mean = points.mean(axis=0)
        print(f"  - Points range: X=[{points_min[0]:.3f}, {points_max[0]:.3f}], "
              f"Y=[{points_min[1]:.3f}, {points_max[1]:.3f}], "
              f"Z=[{points_min[2]:.3f}, {points_max[2]:.3f}]")
        print(f"  - Points centroid: [{points_mean[0]:.3f}, {points_mean[1]:.3f}, {points_mean[2]:.3f}]")
        
        # Save as PLY
        output_path = os.path.join(args.output_dir, f"{obj_name}.ply")
        save_pointcloud_with_normals(points, normals, output_path)
    
    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Saved {num_to_save} objects to: {args.output_dir}")
    print("\nGenerated PLY files:")
    for i, obj_name in enumerate(object_names[:num_to_save]):
        print(f"  {i+1}. {obj_name}.ply")
    
    print("\nVisualization instructions:")
    print("  - Use MeshLab, CloudCompare, or Open3D to view PLY files")
    print("  - Colors represent normal directions (RGB = normalized XYZ)")
    print("  - Check that:")
    print("    * Point clouds form coherent object surfaces")
    print("    * Normals point outward from surfaces")
    print("    * No NaN or infinite values")
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    exit(main())

