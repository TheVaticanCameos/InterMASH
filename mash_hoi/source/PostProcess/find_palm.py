import numpy as np
import torch
import trimesh
import open3d as o3d

from typing import Dict, List

from mash_hoi.source.PostProcess.hand_fitter import AllegroHandFitter
from mash_hoi.source.PostProcess.hand_fitter import BarrettHandFitter


# urdf_path = "/data2/yxz/DATASET/assets_clean/allegro/allegro_hand_left_extended.urdf"
# part_path = "/data2/yxz/DATASET/assets_clean/allegro/meshes/visual"

# allegro_template_path = "/data2/yxz/DATASET/template_info/sphere_allegro_template.npy"
# allegro_template = np.load(allegro_template_path, allow_pickle=True)[()]

# barycentric_coords = allegro_template["barycentric_coords"]
# submesh_num = allegro_template["submesh_num"]
# points_per_submesh = allegro_template["points_per_submesh"]
# triangle_ids = allegro_template["triangle_ids"]
# permutation = allegro_template["permutation"]

# template_verts = allegro_template["verts"]
# template_faces = allegro_template["faces"]

urdf_path = "/data2/yxz/DATASET/assets_clean/barrett/model_extended.urdf"
part_path = "/data2/yxz/DATASET/assets_clean/barrett/meshes"

template_info_path = "/data2/yxz/DATASET/template_info/sphere_barrett_template.npy"
hand_template = np.load(template_info_path, allow_pickle=True)[()]
barycentric_coords = hand_template["barycentric_coords"]
submesh_num = hand_template["submesh_num"]
points_per_submesh = hand_template["points_per_submesh"]
triangle_ids = hand_template["triangle_ids"]
permutation = hand_template["permutation"]

template_verts = hand_template["verts"]
template_faces = hand_template["faces"]

barycentric_coords = barycentric_coords.reshape(-1, 3, 1)
template_sample_points = (barycentric_coords * template_verts[template_faces[triangle_ids]]).sum(axis=1).reshape(submesh_num, points_per_submesh, 3)

# # Save the point cloud as a .ply file
# trimesh.PointCloud(template_sample_points).export("template_sample_points.ply")

# allegro_fitter = AllegroHandFitter(
#     urdf_path=urdf_path,
#     part_path=part_path,
#     device="cuda",
#     submesh_num=submesh_num,
#     points_per_submesh=points_per_submesh,
#     barycentric_coords=barycentric_coords,
#     triangle_ids=triangle_ids,
#     permutation=permutation,
#     use_cmap=True,
#     use_joint_limits=True,
#     use_srf=True,
# )

# base_link_face_ids = allegro_fitter.base_link_face_ids

barrett_fitter = BarrettHandFitter(
    urdf_path=urdf_path,
    part_path=part_path,
    device="cuda",
    submesh_num=submesh_num,
    points_per_submesh=points_per_submesh,
    barycentric_coords=barycentric_coords,
    triangle_ids=triangle_ids,
    permutation=permutation,
    use_cmap=True,
    use_joint_limits=True,
    use_srf=True,
)

base_link_face_ids = barrett_fitter.base_link_face_ids

face_id_to_pt_ids: Dict[int, List[int]] = {}
for pt_id, face_id in enumerate(triangle_ids):
    if face_id not in face_id_to_pt_ids.keys():
        face_id_to_pt_ids[int(face_id)] = []
    face_id_to_pt_ids[int(face_id)].append(pt_id)

base_link_sample_points = []

for face_id in base_link_face_ids:
    if face_id not in face_id_to_pt_ids.keys():
        continue
    pt_ids = face_id_to_pt_ids[face_id]
    sample_points = (barycentric_coords[pt_ids] * template_verts[template_faces[face_id]]).sum(axis=1)
    base_link_sample_points.append(sample_points)

base_link_sample_points = np.concatenate(base_link_sample_points, axis=0)

# trimesh.PointCloud(base_link_sample_points).export("base_link_sample_points.ply")
# trimesh.PointCloud(template_sample_points.reshape(-1, 3)).export("template_sample_points.ply")

# # Visualize the point cloud with Open3D
# # Create a red point cloud from base_link_sample_points
# pcd_base = o3d.geometry.PointCloud()
# pcd_base.points = o3d.utility.Vector3dVector(base_link_sample_points)
# pcd_base.paint_uniform_color([1, 0, 0])  # Red

# # Create a blue point cloud from template_sample_points
# pcd_template = o3d.geometry.PointCloud()
# pcd_template.points = o3d.utility.Vector3dVector(template_sample_points.reshape(-1, 3))
# pcd_template.paint_uniform_color([0, 0, 1])  # Blue

# # Visualize
# o3d.visualization.draw_geometries([pcd_base, pcd_template], window_name="Point cloud comparison")

# Count the base_link_sample_points in each submesh
pt_num_in_submesh: Dict[int, int] = {k: 0 for k in range(submesh_num)}

# First map base_link_sample_points to their corresponding face_id values
base_link_pt_to_face = {}
for face_id in base_link_face_ids:
    if face_id not in face_id_to_pt_ids:
        continue
    for pt_id in face_id_to_pt_ids[face_id]:
        base_link_pt_to_face[pt_id] = face_id

# Iterate over sampled points in each submesh and check whether they belong to base_link
for submesh_id in range(submesh_num):
    start_idx = submesh_id * points_per_submesh
    end_idx = (submesh_id + 1) * points_per_submesh
    for pt_id in range(start_idx, end_idx):
        if pt_id in base_link_pt_to_face:
            pt_num_in_submesh[submesh_id] += 1
        

base_link_submesh_ids = [id for id in range(submesh_num) if pt_num_in_submesh[id] > submesh_num // 2]

import pdb
pdb.set_trace()

base_link_submesh = template_sample_points[base_link_submesh_ids]
# Visualize the base_link_submesh point cloud with Open3D
pcd_base_link_submesh = o3d.geometry.PointCloud()
pcd_base_link_submesh.points = o3d.utility.Vector3dVector(base_link_submesh.reshape(-1, 3))
pcd_base_link_submesh.paint_uniform_color([1, 0, 0])  # Red
o3d.visualization.draw_geometries([pcd_base_link_submesh], window_name="base_link_submesh 点云")


# all_verts, all_faces = allegro_fitter.get_meshes_from_joint(torch.zeros(1, 16, device="cuda"))


# # base_link_verts = allegro_fitter.base_link_info["verts"] @ allegro_fitter.R[:3, :3].T
# # base_link_faces = allegro_fitter.base_link_info["faces"]

# # trimesh.Trimesh(vertices=base_link_verts, faces=base_link_faces).export("base_link.ply")

# # Reflect vertices across the xy plane (negate z)
# reflected_verts = all_verts[0].cpu().numpy() @ allegro_fitter.R[:3, :3].T
# reflected_verts[:, 2] = -reflected_verts[:, 2]
# trimesh.Trimesh(vertices=reflected_verts, faces=all_faces.cpu().numpy()).export("template_from_fitter_reflected.ply")





# import trimesh
# import numpy as np


# allegro_ref_path = "/data/yxz/MASH-HOI/MASH/allegro_ref.ply"

# allegro_template_path = "/data2/yxz/DATASET/template_info/sphere_allegro_template.npy"
# allegro_template = np.load(allegro_template_path, allow_pickle=True)[()]

# barycentric_coords = allegro_template["barycentric_coords"]
# submesh_num = allegro_template["submesh_num"]
# points_per_submesh = allegro_template["points_per_submesh"]
# triangle_ids = allegro_template["triangle_ids"]
# permutation = allegro_template["permutation"]

# template_verts = allegro_template["verts"]
# template_faces = allegro_template["faces"]
# barycentric_coords = barycentric_coords.reshape(-1, 3, 1)
# template_sample_points = (barycentric_coords * template_verts[template_faces[triangle_ids]]).sum(axis=1).reshape(submesh_num, points_per_submesh, 3)

# allegro_ref_mesh = trimesh.load(allegro_ref_path)


# import pdb
# pdb.set_trace()
