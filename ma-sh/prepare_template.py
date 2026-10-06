import os
import sys
import numpy as np
from scipy.spatial import KDTree
import trimesh
import argparse


sys.path.append("../mesh-cut")
sys.path.append("../mesh-sample")


from mesh_cut.Module.average_mesh_cutter import AverageMeshCutter
from ma_sh.Config.constant import SHADOW_HAND_SKIPPED_FACE_IDS
import open3d as o3d
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare hand template")
    parser.add_argument("--data_root", type=str, default="/data2/yxz/DATASET/GrabDataset/")
    parser.add_argument("--hand_mesh_name", type=str, default="hand_rest.ply")
    parser.add_argument("--hand_type", type=str, default="mano")
    parser.add_argument("--submesh_num", type=int, default=256)
    parser.add_argument("--points_per_submesh", type=int, default=1024)
    parser.add_argument("--dist_max", type=float, default=1.0 / 100)
    parser.add_argument("--is_unique_label", type=bool, default=True)
    parser.add_argument("--print_progress", type=bool, default=True)
    parser.add_argument("--show_visualization", type=bool, default=False)
    parser.add_argument("--target_folder", type=str, default="./")
    args = parser.parse_args()

    data_root = args.data_root
    hand_mesh_name = os.path.join(data_root, args.hand_mesh_name)
    hand_mesh = trimesh.load(hand_mesh_name)
    hand_verts = np.array(hand_mesh.vertices)
    hand_faces = np.array(hand_mesh.faces)
    hand_type = args.hand_type

    submesh_num = args.submesh_num
    points_per_submesh = args.points_per_submesh
    dist_max = args.dist_max
    is_unique_label = args.is_unique_label
    print_progress = args.print_progress
    show_visualization = args.show_visualization
    target_folder = args.target_folder

    assert hand_type in ["mano", "shadow", "allegro", "barrett"], f"hand_type must be 'mano', 'shadow', 'allegro' or 'barrett', but got {hand_type}"
    
    average_mesh_cutter = AverageMeshCutter(
        hand_mesh_name,
        dist_max,
        is_unique_label,
        print_progress,
    )

    average_mesh_cutter.cutMesh(
        submesh_num, 
        points_per_submesh
    )
    template_points = average_mesh_cutter.sub_mesh_sample_points.cpu().numpy()  # (B, N, 3)
    template_points = template_points.reshape(-1, 3)    # (BN, 3)

    # if hand_type == "shadow":    # Filter points based on SHADOW_HAND_SKIPPED_FACE_IDS
    #     # Get triangle IDs for each sampled point
    #     closest_points, distances, triangle_ids = hand_mesh.nearest.on_surface(template_points)
    #     closest_points = closest_points.reshape((submesh_num, points_per_submesh, 3))
    #     triangle_ids = triangle_ids.reshape((submesh_num, points_per_submesh))
    #     # Create a mask to filter out points belonging to skipped face IDs
    #     valid_mask = np.isin(triangle_ids, SHADOW_HAND_SKIPPED_FACE_IDS, invert=True)
    #     template_points = template_points.reshape((submesh_num, points_per_submesh, 3))

    #     # Replace invalid points with random points from the same group
    #     for i in range(submesh_num):
    #         for j in range(points_per_submesh):
    #             if not valid_mask[i, j]:
    #                 # Randomly select a valid point from the same group as a replacement
    #                 valid_indices = np.where(valid_mask[i])[0]
    #                 if len(valid_indices) > 0:
    #                     replace_idx = np.random.choice(valid_indices)
    #                     template_points[i, j] = template_points[i, replace_idx]
    #                 else:
    #                     # If the entire group is invalid, select a random point from that group without further processing
    #                     replace_idx = np.random.randint(points_per_submesh)
    #                     template_points[i, j] = template_points[i, replace_idx]

    if show_visualization:
        # Split hand_points into point cloud patches by submesh and assign each patch a different color
        hand_points = template_points.reshape(submesh_num, points_per_submesh, 3)  # (B, N, 3)
        colors = np.random.rand(submesh_num, 3)  # Generate B random colors

        pcd_list = []
        for i in range(submesh_num):
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(hand_points[i])
            pcd.paint_uniform_color(colors[i])
            pcd_list.append(pcd)

        # Visualize all colored point cloud patches
        o3d.visualization.draw_geometries(pcd_list, window_name="Hand Submesh Patches")

    
    closest_points, distances, triangle_ids = hand_mesh.nearest.on_surface(template_points.reshape(-1, 3))
    target_triangles = hand_mesh.triangles[triangle_ids]
    barycentric_coords = trimesh.triangles.points_to_barycentric(
        triangles=target_triangles,
        points=closest_points
    )


    hand_template = {
        "verts": hand_verts,
        "faces": hand_faces,
        "submesh_num": submesh_num,
        "points_per_submesh": points_per_submesh,
        "hand_points": template_points.reshape(submesh_num, points_per_submesh, 3),
        "triangle_ids": triangle_ids,
        "barycentric_coords": barycentric_coords,
    }

    np.save(os.path.join(target_folder, hand_type + "_template.npy"), hand_template)
