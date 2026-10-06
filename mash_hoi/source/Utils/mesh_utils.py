import trimesh
import numpy as np
from scipy.spatial.transform import Rotation as R
import torch
import torch.nn.functional as F


def parse_origin(element):
    """Parse the origin element for translation and rotation."""
    origin = element.find("origin")
    xyz = np.zeros(3)
    rotation = np.eye(3)
    if origin is not None:
        xyz = np.fromstring(origin.attrib.get("xyz", "0 0 0"), sep=" ")
        rpy = np.fromstring(origin.attrib.get("rpy", "0 0 0"), sep=" ")
        rotation = R.from_euler("xyz", rpy).as_matrix()
    return xyz, rotation


def as_mesh(scene_or_mesh):
    """
    Convert a possible scene to a mesh.

    If conversion occurs, the returned mesh has only vertex and face data.
    """
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
            # we lose texture information here
            mesh = trimesh.util.concatenate(
                tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)for g in scene_or_mesh.geometry.values())
            )
    else:
        assert isinstance(scene_or_mesh, trimesh.Trimesh)
        mesh = scene_or_mesh
    return mesh


def apply_transform(mesh, translation, rotation):
    """Apply translation and rotation to a mesh."""
    # mesh.apply_translation(-mesh.centroid)
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    mesh.apply_transform(transform)
    return mesh


def create_primitive_mesh(geometry, translation, rotation):
    """Create a trimesh object from primitive geometry definitions with transformations."""
    if geometry.tag.endswith("box"):
        size = np.fromstring(geometry.attrib["size"], sep=" ")
        mesh = trimesh.creation.box(extents=size)
    elif geometry.tag.endswith("sphere"):
        radius = float(geometry.attrib["radius"])
        mesh = trimesh.creation.icosphere(radius=radius)
    elif geometry.tag.endswith("cylinder"):
        radius = float(geometry.attrib["radius"])
        length = float(geometry.attrib["length"])
        mesh = trimesh.creation.cylinder(radius=radius, height=length)
    else:
        raise ValueError(f"Unsupported geometry type: {geometry.tag}")
    return apply_transform(mesh, translation, rotation)


def compute_rotation_matrix_from_ortho6d(poses):
    x_raw = poses[:, 0:3]
    y_raw = poses[:, 3:6]

    x = F.normalize(x_raw)
    y = F.normalize(y_raw - (x * y_raw).sum(dim=-1, keepdim=True) * x)
    z = torch.linalg.cross(x, y)
    R = torch.stack([x, y, z], dim=-1)

    return R
