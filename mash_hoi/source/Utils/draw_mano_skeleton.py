import numpy as np
from manopth.manolayer import ManoLayer
import torch
import trimesh


mano = ManoLayer(
    mano_root='/data/yxz/MASH-HOI/manopth/mano/models',
    use_pca=False,
    ncomps=45,
    flat_hand_mean=True
)

pose = torch.zeros(1, 48)    # 16 joints × 3
shape = torch.zeros(1, 10)

verts, joints = mano(pose, shape)

joints = joints[0].cpu().numpy()  # (21, 3)

radius = 1.5   # Bone thickness (in the same units as the MANO mesh)
sphere_radius = radius * 1.8

meshes = []

# MANO 21-joint skeleton (standard)
bones = [
    (0, 1),  (1, 2),  (2, 3),  (3, 4),      # Thumb
    (0, 5),  (5, 6),  (6, 7),  (7, 8),      # Index
    (0, 9),  (9,10), (10,11),(11,12),      # Middle
    (0,13), (13,14),(14,15),(15,16),      # Ring
    (0,17), (17,18),(18,19),(19,20),      # Little
]

parent_joints = {i for i, _ in bones}
child_joints = {j for _, j in bones}
leaf_joints = child_joints - parent_joints

for i, j in bones:
    p0 = joints[i]
    p1 = joints[j]

    length = np.linalg.norm(p1 - p0)
    if length < 1e-6:
        continue

    # Create a cylinder aligned with the Z-axis
    cyl = trimesh.creation.cylinder(
        radius=radius,
        height=length,
        sections=12
    )

    # Rotate to align with p0 -> p1
    direction = (p1 - p0) / length
    z_axis = np.array([0, 0, 1])
    rot = trimesh.geometry.align_vectors(z_axis, direction)
    cyl.apply_transform(rot)

    # Translate to the midpoint
    cyl.apply_translation((p0 + p1) / 2)

    meshes.append(cyl)

for joint_idx, p in enumerate(joints):
    # if joint_idx in leaf_joints:
    #     continue
    sph = trimesh.creation.icosphere(subdivisions=2, radius=sphere_radius)
    sph.apply_translation(p)
    meshes.append(sph)

skeleton_mesh = trimesh.util.concatenate(meshes)
skeleton_mesh.export("mano_skeleton.obj")

mesh = trimesh.Trimesh(
    vertices=verts[0].detach().cpu().numpy(),
    faces=mano.th_faces.cpu().numpy()
)
mesh.export("mano_mesh.obj")
