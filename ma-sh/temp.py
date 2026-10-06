import numpy as np
import torch
import math
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import open3d as o3d
import matplotlib.pyplot as plt


# template_path = "/data2/yxz/DATASET/template_info/mano_template.npy"
template_path = "/data2/yxz/DATASET/Original-Meshes/DexGraspNet/mash/shadow_template.npy"

hand_template = np.load(template_path, allow_pickle=True)[()]

verts = hand_template['verts']
faces = hand_template['faces']
triangle_ids = hand_template['triangle_ids']
weights = hand_template['barycentric_coords']
submesh_num = hand_template['submesh_num']
points_per_submesh = hand_template['points_per_submesh']

triangles = verts[faces[triangle_ids]]
sampled_points = np.sum(weights[:, :, np.newaxis] * triangles, axis=1).reshape(submesh_num, points_per_submesh, 3)  # [P, N, 3]

r = 0.2
# offset = -0.03
# N = submesh_num
# i = torch.arange(N)
# golden_angle = torch.tensor(math.pi * (3.0 - math.sqrt(5.0)))
# angles = (i * golden_angle) % (2 * torch.pi)
# radii = torch.sqrt((i + 0.5) / float(N)) * r

# # Compute the coordinate offsets of sampled points in the xz plane
# x_offset = torch.cos(angles) * radii
# z_offset = torch.sin(angles) * radii

# # Construct sampled point coordinates with the same y coordinate as the center
# points_in_disk = torch.stack([
#     x_offset,
#     offset * torch.ones_like(x_offset),
#     z_offset
# ], dim=-1)


N = submesh_num
i = torch.arange(N)
golden_angle = torch.tensor(math.pi * (3.0 - math.sqrt(5.0)))
phis = (i * golden_angle) % (2 * torch.pi)
ys = 1.0 - 2.0 * ((i + 0.5) / float(N))
r_xy = torch.sqrt(torch.clamp(1.0 - ys.square(), min=0.0)) * r
xs = torch.cos(phis) * r_xy
zs = torch.sin(phis) * r_xy
ys = ys * r
points_on_sphere = torch.stack([xs, ys, zs], dim=-1)

patch_centers = sampled_points.mean(axis=1)
anchors = points_on_sphere.numpy()

cost_matrix = cdist(anchors, patch_centers, metric="euclidean")
row_ind, col_ind = linear_sum_assignment(cost_matrix)

hand_template['permutation'] = col_ind


# np.save("/data2/yxz/DATASET/template_info/sphere_mano_template.npy", hand_template, allow_pickle=True)
np.save("/data2/yxz/DATASET/Original-Meshes/DexGraspNet/mash/draw_sphere_shadow_template.npy", hand_template, allow_pickle=True)

# Visualize the point cloud with Open3D using a different color for each patch
P, N, _ = sampled_points.shape
colors = plt.get_cmap('tab20')(np.linspace(0, 1, P))[:, :3]
# Reshape the point cloud to [P*N, 3]
points = sampled_points.reshape(-1, 3)
# Assign each point the color of its patch
point_colors = np.repeat(colors, N, axis=0)  # [P*N, 3]

# Create an Open3D point cloud object
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(point_colors)

# Convert points_on_sphere to NumPy and create a black point cloud
sphere_points_np = points_on_sphere.numpy()
sphere_pcd = o3d.geometry.PointCloud()
sphere_pcd.points = o3d.utility.Vector3dVector(sphere_points_np)
sphere_colors = np.zeros((sphere_points_np.shape[0], 3))  # Black
sphere_pcd.colors = o3d.utility.Vector3dVector(sphere_colors)

# Visualize both point clouds
o3d.visualization.draw_geometries([pcd, sphere_pcd], window_name="Patch 点云可视化", width=800, height=600)
