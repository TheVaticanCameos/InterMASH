import argparse
import os
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import transforms3d
import trimesh
import pytorch_kinematics as pk
from pytorch_kinematics.urdf_parser_py.urdf import URDF, Box, Cylinder, Mesh, Sphere


def _safe_rpy_xyz(origin) -> Tuple[np.ndarray, np.ndarray]:
    if origin is None:
        rpy = np.zeros(3, dtype=np.float64)
        xyz = np.zeros(3, dtype=np.float64)
        return rpy, xyz
    rpy = np.array(getattr(origin, "rpy", [0.0, 0.0, 0.0]), dtype=np.float64)
    xyz = np.array(getattr(origin, "xyz", [0.0, 0.0, 0.0]), dtype=np.float64)
    return rpy, xyz


def _resolve_mesh_path(mesh_filename: str, urdf_dir: str, part_path: str) -> str:
    candidates: List[str] = []
    if mesh_filename.startswith("package://"):
        rel = mesh_filename[len("package://") :]
        rel = rel.lstrip("/").lstrip("\\")
        candidates.append(os.path.join(urdf_dir, rel))
        candidates.append(os.path.join(part_path, rel))
        candidates.append(os.path.join(part_path, os.path.basename(rel)))
    else:
        if os.path.isabs(mesh_filename):
            candidates.append(mesh_filename)
        candidates.append(os.path.join(urdf_dir, mesh_filename))
        candidates.append(os.path.join(part_path, mesh_filename))
        candidates.append(os.path.join(part_path, os.path.basename(mesh_filename)))

    for p in candidates:
        if p and os.path.exists(p):
            return p
    raise FileNotFoundError(f"Cannot resolve mesh path for '{mesh_filename}'. Tried: {candidates}")


def _load_visual_geometry(visual, urdf_dir: str, part_path: str) -> trimesh.Trimesh:
    geom = visual.geometry

    if isinstance(geom, Mesh):
        mesh_path = _resolve_mesh_path(str(geom.filename), urdf_dir=urdf_dir, part_path=part_path)
        mesh = trimesh.load(mesh_path, force="mesh", process=True)
    elif isinstance(geom, Box):
        mesh = trimesh.primitives.Box(extents=geom.size)
    elif isinstance(geom, Cylinder):
        mesh = trimesh.primitives.Cylinder(radius=geom.radius, height=geom.length)
    elif isinstance(geom, Sphere):
        mesh = trimesh.primitives.Sphere(radius=geom.radius)
    else:
        raise NotImplementedError(f"Unsupported visual geometry: {type(geom)}")

    try:
        scale = np.array(geom.scale, dtype=np.float64).reshape(1, 3)
    except Exception:
        scale = np.ones((1, 3), dtype=np.float64)

    vertices = np.asarray(mesh.vertices, dtype=np.float64) * scale
    rpy, xyz = _safe_rpy_xyz(getattr(visual, "origin", None))
    rotation = transforms3d.euler.euler2mat(*rpy)
    vertices = (rotation @ vertices.T).T + xyz.reshape(1, 3)

    return trimesh.Trimesh(vertices=vertices, faces=np.asarray(mesh.faces), process=False)


def _build_link_meshes(urdf_model: URDF, urdf_dir: str, part_path: str, exclude_links: Iterable[str]) -> Dict[str, trimesh.Trimesh]:
    exclude = set(exclude_links)
    link_meshes: Dict[str, trimesh.Trimesh] = {}
    for link in urdf_model.links:
        if link.name in exclude:
            continue
        if not getattr(link, "visuals", None):
            continue

        parts: List[trimesh.Trimesh] = []
        for visual in link.visuals:
            parts.append(_load_visual_geometry(visual, urdf_dir=urdf_dir, part_path=part_path))
        link_meshes[link.name] = trimesh.util.concatenate(parts)
    return link_meshes


def _forward_link_transforms(chain: pk.Chain, joint_angles: torch.Tensor) -> Dict[str, np.ndarray]:
    status = chain.forward_kinematics(joint_angles)
    out: Dict[str, np.ndarray] = {}
    for name, tf in status.items():
        mat = tf.get_matrix()[0].detach().cpu().numpy()
        out[name] = mat
    return out


def _transform_mesh(mesh: trimesh.Trimesh, matrix: np.ndarray) -> trimesh.Trimesh:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    v_h = np.concatenate([vertices, np.ones((vertices.shape[0], 1), dtype=np.float64)], axis=1)
    v_out = (matrix @ v_h.T).T[:, :3]
    return trimesh.Trimesh(vertices=v_out, faces=np.asarray(mesh.faces), process=False)


def _concat_meshes(meshes: Sequence[trimesh.Trimesh]) -> trimesh.Trimesh:
    if len(meshes) == 0:
        return trimesh.Trimesh(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3), dtype=np.int64), process=False)

    vertices_list: List[np.ndarray] = []
    faces_list: List[np.ndarray] = []
    v_offset = 0
    for m in meshes:
        v = np.asarray(m.vertices, dtype=np.float64)
        f = np.asarray(m.faces, dtype=np.int64)
        vertices_list.append(v)
        faces_list.append(f + v_offset)
        v_offset += v.shape[0]

    vertices = np.concatenate(vertices_list, axis=0)
    faces = np.concatenate(faces_list, axis=0)
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def _align_cylinder_z_to_direction(direction: np.ndarray) -> np.ndarray:
    direction = np.asarray(direction, dtype=np.float64)
    direction = direction / (np.linalg.norm(direction) + 1e-12)
    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return trimesh.geometry.align_vectors(z_axis, direction)


def _build_skeleton_mesh(
    link_positions: Dict[str, np.ndarray],
    bones: Sequence[Tuple[str, str]],
    radius: float,
    sphere_radius: float,
    exclude_links: Iterable[str],
) -> trimesh.Trimesh:
    exclude = set(exclude_links)
    meshes: List[trimesh.Trimesh] = []

    parent_nodes = {p for p, _ in bones}
    child_nodes = {c for _, c in bones}

    for parent, child in bones:
        if parent in exclude or child in exclude:
            continue
        p0 = link_positions.get(parent, None)
        p1 = link_positions.get(child, None)
        if p0 is None or p1 is None:
            continue
        length = float(np.linalg.norm(p1 - p0))
        if length < 1e-8:
            continue
        cyl = trimesh.creation.cylinder(radius=radius, height=length, sections=12)
        rot = _align_cylinder_z_to_direction(p1 - p0)
        cyl.apply_transform(rot)
        cyl.apply_translation((p0 + p1) / 2.0)
        meshes.append(cyl)

    nodes = (parent_nodes | child_nodes) - exclude
    for node in nodes:
        pos = link_positions.get(node, None)
        if pos is None:
            continue
        sph = trimesh.creation.icosphere(subdivisions=2, radius=sphere_radius)
        sph.apply_translation(pos)
        meshes.append(sph)

    return trimesh.util.concatenate(meshes) if len(meshes) > 0 else trimesh.Trimesh(process=False)


def export_barrett_standard_pose(
    urdf_path: str,
    part_path: str,
    out_dir: str,
    mesh_name: str = "barrett_mesh.obj",
    skeleton_name: str = "barrett_skeleton.obj",
    radius: float = 0.003,
    sphere_radius: float = 0.005,
    exclude_links: Sequence[str] = ("forearm",),
) -> Tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)

    urdf_text = open(urdf_path).read()
    urdf_model = URDF.from_xml_string(urdf_text)
    chain = pk.build_chain_from_urdf(urdf_text).to(dtype=torch.float32, device=torch.device("cpu"))

    urdf_dir = os.path.dirname(os.path.abspath(urdf_path))
    link_meshes = _build_link_meshes(
        urdf_model=urdf_model,
        urdf_dir=urdf_dir,
        part_path=part_path,
        exclude_links=exclude_links,
    )
    joint_angles = torch.zeros(1, len(chain.get_joint_parameter_names()), dtype=torch.float32)
    link_transforms = _forward_link_transforms(chain=chain, joint_angles=joint_angles)

    world_meshes: List[trimesh.Trimesh] = []
    for link_name, mesh in link_meshes.items():
        if link_name not in link_transforms:
            continue
        world_meshes.append(_transform_mesh(mesh, link_transforms[link_name]))
    mesh_out = _concat_meshes(world_meshes)
    mesh_path = os.path.join(out_dir, mesh_name)
    mesh_out.export(mesh_path)

    bones: List[Tuple[str, str]] = []
    for j in urdf_model.joints:
        bones.append((j.parent, j.child))

    link_positions = {name: tf[:3, 3] for name, tf in link_transforms.items()}
    skeleton_mesh = _build_skeleton_mesh(
        link_positions=link_positions,
        bones=bones,
        radius=radius,
        sphere_radius=sphere_radius,
        exclude_links=exclude_links,
    )
    skeleton_path = os.path.join(out_dir, skeleton_name)
    skeleton_mesh.export(skeleton_path)

    return mesh_path, skeleton_path


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--urdf_path",
        type=str,
        default="/data2/yxz/DATASET/assets_clean/barrett/model_extended.urdf",
    )
    parser.add_argument(
        "--part_path",
        type=str,
        default="/data2/yxz/DATASET/assets_clean/barrett/meshes",
    )
    parser.add_argument("--out_dir", type=str, default=".")
    parser.add_argument("--mesh_name", type=str, default="barrett_mesh.obj")
    parser.add_argument("--skeleton_name", type=str, default="barrett_skeleton.obj")
    parser.add_argument("--radius", type=float, default=0.003)
    parser.add_argument("--sphere_radius", type=float, default=0.005)
    parser.add_argument("--exclude_links", type=str, nargs="*", default=["forearm"])
    return parser


def main() -> None:
    args = _build_argparser().parse_args()
    export_barrett_standard_pose(
        urdf_path=args.urdf_path,
        part_path=args.part_path,
        out_dir=args.out_dir,
        mesh_name=args.mesh_name,
        skeleton_name=args.skeleton_name,
        radius=args.radius,
        sphere_radius=args.sphere_radius,
        exclude_links=args.exclude_links,
    )


if __name__ == "__main__":
    main()

