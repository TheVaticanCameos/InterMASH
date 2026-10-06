import os
import sys

sys.path.append("../mesh-cut")
sys.path.append("../mesh-sample")

import trimesh
import numpy as np

from mesh_cut.Module.average_mesh_cutter import AverageMeshCutter


def compute_robust_barycentric(triangles: np.ndarray, points: np.ndarray) -> np.ndarray:
    tri = np.asarray(triangles, dtype=np.float64)
    pts = np.asarray(points, dtype=np.float64)
    A = tri[:, 0, :]
    B = tri[:, 1, :]
    C = tri[:, 2, :]
    AB = B - A
    AC = C - A
    N = np.cross(AB, AC)
    nn = (N * N).sum(axis=1, keepdims=True)
    AP = pts - A
    proj = pts - (np.sum(AP * N, axis=1, keepdims=True) / np.maximum(nn, 1e-20)) * N
    v0 = AC
    v1 = AB
    v2 = proj - A
    d00 = np.sum(v0 * v0, axis=1)
    d01 = np.sum(v0 * v1, axis=1)
    d11 = np.sum(v1 * v1, axis=1)
    d20 = np.sum(v2 * v0, axis=1)
    d21 = np.sum(v2 * v1, axis=1)
    denom = d00 * d11 - d01 * d01
    eps = 1e-20
    valid = denom > eps
    v = np.zeros_like(denom)
    w = np.zeros_like(denom)
    v[valid] = (d11[valid] * d20[valid] - d01[valid] * d21[valid]) / denom[valid]
    w[valid] = (d00[valid] * d21[valid] - d01[valid] * d20[valid]) / denom[valid]
    u = 1.0 - v - w
    bary = np.stack([u, v, w], axis=1)
    invalid = ~valid
    if np.any(invalid):
        dA = np.linalg.norm(proj - A, axis=1)
        dB = np.linalg.norm(proj - B, axis=1)
        dC = np.linalg.norm(proj - C, axis=1)
        nearest = np.argmin(np.stack([dA, dB, dC], axis=1), axis=1)
        fallback = np.zeros_like(bary)
        fallback[np.arange(bary.shape[0]), nearest] = 1.0
        bary[invalid] = fallback[invalid]
    bary = np.clip(bary, -1e-6, 1.000001)
    s = bary.sum(axis=1, keepdims=True)
    renorm = np.abs(s - 1.0) > 1e-5
    if np.any(renorm):
        bary[renorm[:, 0]] = bary[renorm[:, 0]] / s[renorm]
    finite = np.isfinite(bary).all(axis=1)
    if not np.all(finite):
        dA = np.linalg.norm(proj - A, axis=1)
        dB = np.linalg.norm(proj - B, axis=1)
        dC = np.linalg.norm(proj - C, axis=1)
        nearest = np.argmin(np.stack([dA, dB, dC], axis=1), axis=1)
        fallback = np.zeros_like(bary)
        fallback[np.arange(bary.shape[0]), nearest] = 1.0
        bary[~finite] = fallback[~finite]
    return bary


def get_object_template(
    object_path: str,
    save_path: str,
    submesh_num: int,
    points_per_submesh: int,
) -> bool:
    object_mesh = trimesh.load(object_path)

    cutter = AverageMeshCutter(
        object_path,
        dist_max=1.0 / 100,
        is_unique_label=True,
        print_progress=False,
    )

    cutter.cutMesh(
        submesh_num,
        points_per_submesh,
    )

    template_points = cutter.sub_mesh_sample_points.cpu().numpy()
    template_points = template_points.reshape(-1, 3)

    closest_points, _, triangle_ids = object_mesh.nearest.on_surface(template_points)
    target_triangles = object_mesh.triangles[triangle_ids]
    barycentric_coords = compute_robust_barycentric(target_triangles, closest_points).astype(np.float32)

    object_info = {
        "verts": np.array(object_mesh.vertices, dtype=np.float32),
        "faces": np.array(object_mesh.faces, dtype=np.int32),
        "submesh_num": submesh_num,
        "points_per_submesh": points_per_submesh,
        "triangle_ids": triangle_ids,
        "barycentric_coords": barycentric_coords,
    }

    np.save(save_path, object_info)

    return True


# if __name__ == "__main__":

#     submesh_num = 256
#     points_per_submesh = 1024

#     # name = "contactdb+hammer"
#     # objject_class = name.split("+", 1)[0]
#     # object_name = name.split("+", 1)[1].rsplit("_", 1)[0]

#     # object_path = f"/data2/yxz/DATASET/DRO-data/data_urdf/object/{objject_class}/{object_name}/{object_name}.stl"
#     object_root = "/data2/yxz/DATASET/grab_data/obj_meshes"
#     obj_list = os.listdir(object_root)
#     for obj_name in obj_list:
#         object_path = f"{object_root}/{obj_name}"
#         save_path = f"/data2/yxz/DATASET/GRAB-Original/object_templates/{obj_name}.npy"

#         object_mesh = trimesh.load(object_path)
#         # object_mesh.apply_scale(1.0)
#         object_mesh.export("test.ply")

#         object_path = "test.ply"

#         get_object_template(
#             object_path,
#             save_path,
#             submesh_num,
#             points_per_submesh,
#         )

if __name__ == "__main__":

    submesh_num = 64
    points_per_submesh = 1024

    object_path = "/data2/yxz/DATASET/Original-Meshes/DexGraspNet/train/shadow_core-bottle-1a7ba1f4c892e2da30711cdbdbc73924_0_object.ply"
    save_path = "/data2/yxz/DATASET/Original-Meshes/DexGraspNet/mash/object_template.npy"

    object_mesh = trimesh.load(object_path)
    object_mesh.export("test.ply")

    object_path = "test.ply"

    get_object_template(
        object_path,
        save_path,
        submesh_num,
        points_per_submesh,
    )
