"""
Visualization utilities for IoU matrix and point clouds.

This module provides functions to visualize:
- IoU coverage on object point clouds
- Hand and object anchor correspondences

All input data should be pre-computed before calling these functions.
"""

import os
import numpy as np
import torch
import open3d as o3d
import matplotlib.pyplot as plt


def visualize_iou_on_object_pointcloud(
    object_points: torch.Tensor,
    iou_matrix: torch.Tensor,
    output_path: str,
    colormap_name: str = "tab20",
    iou_threshold: float = 0.0,
) -> None:
    """
    Visualize the IoU matrix on the object point cloud.
    
    Each object anchor region is colored based on the hand disk that covers it
    with the highest IoU. The color represents the hand disk ID.
    
    Args:
        object_points: [N, S, 3] or [N*S, 3] object point cloud (already sampled)
        iou_matrix: [N, N] IoU matrix where [i, j] = IoU of object patch i covered by hand patch j
        output_path: Path to save the colored PLY file
        colormap_name: Matplotlib colormap name for hand disk colors
        iou_threshold: Minimum IoU threshold to consider a valid coverage
    """
    N = iou_matrix.shape[0]
    
    # Flatten points if needed
    if object_points.dim() == 3:
        S = object_points.shape[1]
        object_points_flat = object_points.reshape(-1, 3)
    else:
        object_points_flat = object_points
        S = object_points_flat.shape[0] // N
    
    # Get the hand disk ID with maximum IoU for each object anchor
    max_iou_values, max_hand_ids = iou_matrix.max(dim=1)  # [N], [N]
    
    # Get colormap
    cmap = plt.get_cmap(colormap_name)
    
    # Prepare colors for each point
    all_points = object_points_flat.cpu().numpy()
    all_colors = np.zeros((all_points.shape[0], 3), dtype=np.float64)
    
    for i in range(N):
        hand_id = max_hand_ids[i].item()
        max_iou = max_iou_values[i].item()
        color_idx = hand_id / max(N - 1, 1)
        
        if max_iou > iou_threshold:
            rgba = cmap(color_idx)
            color = np.array(rgba[:3])
        else:
            color = np.array([0.8, 0.8, 0.8])
        
        start_idx = i * S
        end_idx = (i + 1) * S
        all_colors[start_idx:end_idx] = color
    
    # Create and save Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)
    pcd.colors = o3d.utility.Vector3dVector(all_colors)
    o3d.io.write_point_cloud(output_path, pcd)
    print(f"Saved IoU visualization to: {output_path}")
    
    # Print statistics
    covered_count = (max_iou_values > iou_threshold).sum().item()
    print(f"  Covered anchors: {covered_count}/{N} ({100*covered_count/N:.1f}%)")
    if covered_count > 0:
        print(f"  Unique hand disks covering object: {len(torch.unique(max_hand_ids[max_iou_values > iou_threshold]))}")


def visualize_edge_detailed(
    hand_points: torch.Tensor,
    object_points: torch.Tensor,
    edge_matrix: torch.Tensor,
    output_dir: str,
    hand_normals: torch.Tensor = None,
    object_normals: torch.Tensor = None,
    colormap_name: str = "tab20",
    edge_weight_threshold: float = 0.0,
) -> None:
    """
    Generate detailed visualization of IoU matrix including:
    1. Object point cloud colored by covering hand disk ID (with normals if provided)
    2. Hand point cloud with anchor colors (with normals if provided)
    
    Args:
        hand_points: [N, S, 3] or [N*S, 3] pre-computed hand point cloud
        object_points: [N, S, 3] or [N*S, 3] pre-computed object point cloud
        iou_matrix: [N, N] IoU matrix
        output_dir: Directory to save visualization files
        hand_normals: [N, S, 3] or [N*S, 3] pre-computed hand point normals (optional)
        object_normals: [N, S, 3] or [N*S, 3] pre-computed object point normals (optional)
        colormap_name: Matplotlib colormap name
        iou_threshold: Minimum IoU threshold
    """
    os.makedirs(output_dir, exist_ok=True)
    
    N = edge_matrix.shape[0]
    
    # Flatten points if needed
    if hand_points.dim() == 3:
        S = hand_points.shape[1]
        hand_points_flat = hand_points.reshape(-1, 3).cpu().numpy()
        object_points_flat = object_points.reshape(-1, 3).cpu().numpy()
    elif hand_points.dim() == 4:  # [N, S_phi, S_theta, 3]
        S = hand_points.shape[1] * hand_points.shape[2]
        hand_points_flat = hand_points.reshape(-1, 3).cpu().numpy()
        object_points_flat = object_points.reshape(-1, 3).cpu().numpy()
    else:
        hand_points_flat = hand_points.cpu().numpy()
        object_points_flat = object_points.cpu().numpy()
        S = hand_points_flat.shape[0] // N
    
    # Flatten normals if provided
    hand_normals_flat = None
    object_normals_flat = None
    if hand_normals is not None:
        hand_normals_flat = hand_normals.reshape(-1, 3).cpu().numpy()
    if object_normals is not None:
        object_normals_flat = object_normals.reshape(-1, 3).cpu().numpy()
    
    # Get colormap
    cmap = plt.get_cmap(colormap_name)
    
    # Get the hand disk ID with maximum IoU for each object anchor
    max_iou_values, max_hand_ids = edge_matrix.max(dim=1)  # [N], [N]
    
    # =========================================================================
    # 1. Object point cloud colored by covering hand disk ID
    # =========================================================================
    object_colors = np.zeros((N * S, 3), dtype=np.float64)
    
    for i in range(N):
        hand_id = max_hand_ids[i].item()
        max_iou = max_iou_values[i].item()
        color_idx = hand_id / max(N - 1, 1)
        
        if max_iou > edge_weight_threshold:
            rgba = cmap(color_idx)
            color = np.array(rgba[:3])
        else:
            color = np.array([0.8, 0.8, 0.8])
        
        start_idx = i * S
        end_idx = (i + 1) * S
        object_colors[start_idx:end_idx] = color
    
    pcd_obj = o3d.geometry.PointCloud()
    pcd_obj.points = o3d.utility.Vector3dVector(object_points_flat)
    pcd_obj.colors = o3d.utility.Vector3dVector(object_colors)
    if object_normals_flat is not None:
        pcd_obj.normals = o3d.utility.Vector3dVector(object_normals_flat)
    o3d.io.write_point_cloud(os.path.join(output_dir, "object_iou_coverage.ply"), pcd_obj)
    print(f"Saved: object_iou_coverage.ply" + (" (with normals)" if object_normals_flat is not None else ""))
    
    # =========================================================================
    # 2. Hand point cloud with anchor colors (same colormap)
    # =========================================================================
    hand_colors = np.zeros((N * S, 3), dtype=np.float64)
    
    for i in range(N):
        color_idx = i / max(N - 1, 1)
        rgba = cmap(color_idx)
        color = np.array(rgba[:3])
        
        start_idx = i * S
        end_idx = (i + 1) * S
        hand_colors[start_idx:end_idx] = color
    
    pcd_hand = o3d.geometry.PointCloud()
    pcd_hand.points = o3d.utility.Vector3dVector(hand_points_flat)
    pcd_hand.colors = o3d.utility.Vector3dVector(hand_colors)
    if hand_normals_flat is not None:
        pcd_hand.normals = o3d.utility.Vector3dVector(hand_normals_flat)
    o3d.io.write_point_cloud(os.path.join(output_dir, "hand_anchors.ply"), pcd_hand)
    print(f"Saved: hand_anchors.ply" + (" (with normals)" if hand_normals_flat is not None else ""))
    
    # =========================================================================
    # Print statistics
    # =========================================================================
    print(f"\n{'='*60}")
    print("IoU Visualization Statistics")
    print(f"{'='*60}")
    covered_mask = max_iou_values > edge_weight_threshold
    covered_count = covered_mask.sum().item()
    print(f"Object anchors covered: {covered_count}/{N} ({100*covered_count/N:.1f}%)")
    
    if covered_count > 0:
        print(f"IoU of covered regions: min={max_iou_values[covered_mask].min().item():.4f}, "
              f"max={max_iou_values[covered_mask].max().item():.4f}, "
              f"mean={max_iou_values[covered_mask].mean().item():.4f}")
        unique_hands = torch.unique(max_hand_ids[covered_mask])
        print(f"Unique hand anchors involved: {len(unique_hands)}")
    print(f"{'='*60}")


def _create_white_to_green_cmap():
    """Create a custom colormap: white (0) -> light green -> dark green (max)."""
    from matplotlib.colors import LinearSegmentedColormap
    
    # Define colors: white -> light green -> dark green
    colors = [
        (1.0, 1.0, 1.0),      # White for 0
        (0.85, 0.95, 0.85),   # Very light green
        (0.6, 0.9, 0.6),      # Light green
        (0.3, 0.7, 0.3),      # Medium green
        (0.1, 0.5, 0.1),      # Dark green
        (0.0, 0.3, 0.0),      # Very dark green
    ]
    
    cmap = LinearSegmentedColormap.from_list('white_to_green', colors, N=256)
    return cmap


def visualize_edge_matrix_heatmap(
    edge_matrix: torch.Tensor,
    output_path: str,
    title: str = "IoU Matrix",
    colormap_name: str = "white_to_green",
    figsize: tuple = (10, 10),
    show_colorbar: bool = True,
    vmin: float = None,
    vmax: float = None,
) -> None:
    """
    Visualize IoU matrix as a heatmap.
    
    Args:
        iou_matrix: [N, N] IoU/weight matrix
        output_path: Path to save the heatmap image
        title: Title of the heatmap
        colormap_name: Matplotlib colormap name or 'white_to_green' for custom
        figsize: Figure size (width, height)
        show_colorbar: Whether to show colorbar
        vmin: Minimum value for color scale (None for auto)
        vmax: Maximum value for color scale (None for auto)
    """
    matrix_np = edge_matrix.cpu().numpy() if isinstance(edge_matrix, torch.Tensor) else edge_matrix
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Use custom colormap for white -> green
    if colormap_name == "white_to_green":
        cmap = _create_white_to_green_cmap()
    else:
        cmap = colormap_name
    
    # Create heatmap
    im = ax.imshow(matrix_np, cmap=cmap, aspect='equal', vmin=vmin, vmax=vmax)
    
    if show_colorbar:
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Weight', rotation=270, labelpad=15)
    
    ax.set_xlabel('Hand Anchor Index')
    ax.set_ylabel('Object Anchor Index')
    ax.set_title(title)
    
    # Add statistics as text
    nonzero_count = (matrix_np > 0).sum()
    total_count = matrix_np.size
    stats_text = f"Non-zero: {nonzero_count}/{total_count} ({100*nonzero_count/total_count:.2f}%)"
    if nonzero_count > 0:
        nonzero_vals = matrix_np[matrix_np > 0]
        stats_text += f"\nMin: {nonzero_vals.min():.4f}, Max: {nonzero_vals.max():.4f}, Mean: {nonzero_vals.mean():.4f}"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def visualize_edge_matrix_sparse(
    edge_matrix: torch.Tensor,
    output_path: str,
    title: str = "IoU Matrix (Sparse View)",
    colormap_name: str = "white_to_green",
    figsize: tuple = (12, 10),
    point_size: float = 5.0,
) -> None:
    """
    Visualize sparse IoU matrix as a scatter plot (better for sparse matrices).
    
    Args:
        iou_matrix: [N, N] IoU/weight matrix
        output_path: Path to save the image
        title: Title of the plot
        colormap_name: Matplotlib colormap name or 'white_to_green' for custom
        figsize: Figure size
        point_size: Size of scatter points
    """
    matrix_np = edge_matrix.cpu().numpy() if isinstance(edge_matrix, torch.Tensor) else edge_matrix
    N = matrix_np.shape[0]
    
    # Find non-zero entries
    obj_indices, hand_indices = np.where(matrix_np > 0)
    values = matrix_np[obj_indices, hand_indices]
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Use custom colormap for white -> green
    if colormap_name == "white_to_green":
        cmap = _create_white_to_green_cmap()
    else:
        cmap = colormap_name
    
    # Left: Scatter plot of non-zero entries
    ax1 = axes[0]
    if len(values) > 0:
        scatter = ax1.scatter(hand_indices, obj_indices, c=values, cmap=cmap, 
                              s=point_size, alpha=0.8, vmin=0, vmax=values.max())
        plt.colorbar(scatter, ax=ax1, label='Weight')
    ax1.set_xlim(-0.5, N - 0.5)
    ax1.set_ylim(N - 0.5, -0.5)  # Invert y-axis to match matrix convention
    ax1.set_xlabel('Hand Anchor Index')
    ax1.set_ylabel('Object Anchor Index')
    ax1.set_title(title)
    ax1.set_aspect('equal')
    
    # Add grid for reference
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Right: Row-wise sum (number of hand anchors covering each object anchor)
    ax2 = axes[1]
    row_sums = matrix_np.sum(axis=1)
    nonzero_rows = (row_sums > 0).sum()
    
    # Use green color for bar chart
    ax2.barh(np.arange(N), row_sums, color='#2e7d32', alpha=0.7)
    ax2.set_xlabel('Total Weight Sum')
    ax2.set_ylabel('Object Anchor Index')
    ax2.set_title(f'Row Sums ({nonzero_rows}/{N} non-zero rows)')
    ax2.invert_yaxis()
    
    # Statistics
    nonzero_count = len(values)
    total_count = N * N
    fig.suptitle(f"Non-zero entries: {nonzero_count}/{total_count} ({100*nonzero_count/total_count:.2f}%)", 
                 fontsize=10, y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")
