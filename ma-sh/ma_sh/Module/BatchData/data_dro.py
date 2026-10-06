import os
import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import trimesh
from tqdm import tqdm
import json
from typing import List, Tuple


class DRODataset(Dataset):
    """
    DRO Dataset for HOI generation
    """
    def __init__(
        self, 
        data_root: str,
        object_template_root: str,
        hand_template_path: str,
        use_permutation: bool = True,
    ) -> None:
        self.data_root = data_root
        self.object_template_root = object_template_root

        hand_template = np.load(hand_template_path, allow_pickle=True).item()
        self.hand_weights = hand_template["barycentric_coords"]
        self.hand_triangle_ids = hand_template["triangle_ids"]
        self.submesh_num = hand_template["submesh_num"]
        self.points_per_submesh = hand_template["points_per_submesh"]
        self.permutation = hand_template["permutation"]
        self.use_permutation = use_permutation

        object_names = os.listdir(self.object_template_root)
        object_template_paths = [os.path.join(self.object_template_root, p) for p in object_names]
        self.object_template = {}
        for p in tqdm(object_template_paths, desc="Loading object template"):
            self.object_template[Path(p).stem] = np.load(p, allow_pickle=True).item()

        data_names = os.listdir(self.data_root)
        self.hand_mesh_paths = [os.path.join(self.data_root, p) for p in data_names if p.endswith("hand.ply")]
        self.hand_mesh_paths.sort()
        self.object_mesh_paths = [os.path.join(self.data_root, p) for p in data_names if p.endswith("object.ply")]
        self.object_mesh_paths.sort()
        self.cmap_paths = [os.path.join(self.data_root, p) for p in data_names if p.endswith("cmap.npy")]
        self.cmap_paths.sort()

        assert len(self.hand_mesh_paths) == len(self.object_mesh_paths) == len(self.cmap_paths)

    def __len__(self) -> int:
        return len(self.hand_mesh_paths)

    def __getitem__(self, idx: int):
        hand_mesh = trimesh.load(self.hand_mesh_paths[idx])
        hand_verts = np.array(hand_mesh.vertices)
        hand_faces = np.array(hand_mesh.faces)
        hand_triangles = hand_verts[hand_faces[self.hand_triangle_ids]]
        hand_points_np = np.sum(self.hand_weights[:, :, np.newaxis] * hand_triangles, axis=1)
        hand_points = torch.from_numpy(hand_points_np.reshape(self.submesh_num, self.points_per_submesh, 3)).float()

        if self.use_permutation:
            hand_points = hand_points[self.permutation, :, :]

        object_name = Path(self.object_mesh_paths[idx]).stem.split("_", 1)[1].rsplit("_", 2)[0].split('+', 1)[1]
        object_template = self.object_template[object_name]
        object_mesh = trimesh.load(self.object_mesh_paths[idx])
        object_verts = np.array(object_mesh.vertices)
        object_faces = np.array(object_mesh.faces)

        object_triangles = object_verts[object_faces[object_template["triangle_ids"]]]
        object_points_np = np.sum(object_template["barycentric_coords"][:, :, np.newaxis] * object_triangles, axis=1)
        object_points = torch.from_numpy(object_points_np.reshape(self.submesh_num, self.points_per_submesh, 3)).float()

        cmap = torch.from_numpy(np.load(self.cmap_paths[idx], allow_pickle=True)).float()
        data_name = Path(self.object_mesh_paths[idx]).stem.rsplit('_', 1)[0]

        return hand_points, object_points, cmap, data_name


class SphereDRODataset(Dataset):
    """
    DRO Dataset for HOI fitting [Test Split] [Sphere Anchor]
    """
    def __init__(
        self, 
        object_root: str, 
        object_template_root: str, 
        split_path: str, 
        submesh_num: int = 256, 
        points_per_submesh: int = 1024
    ) -> None:
        self.object_root = object_root
        self.object_template_root = object_template_root
        self.submesh_num = submesh_num
        self.points_per_submesh = points_per_submesh

        with open(split_path, "r") as f:
            self.test_object_names: List[str] = json.load(f)["validate"]

        object_names = os.listdir(self.object_template_root)
        object_template_paths = [os.path.join(self.object_template_root, p) for p in object_names]
        self.object_template = {}
        for p in tqdm(object_template_paths, desc="Loading object template"):
            self.object_template[Path(p).stem] = np.load(p, allow_pickle=True).item()

    def __len__(self) -> int:
        return len(self.test_object_names)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        object_name = self.test_object_names[idx]
        object_template = self.object_template[object_name]

        object_path = os.path.join(self.object_root, object_name.split('+', 1)[0], object_name.split('+', 1)[1], f"{object_name.split('+', 1)[1]}.stl")

        object_mesh = trimesh.load(object_path)
        object_verts = np.array(object_mesh.vertices)
        object_faces = np.array(object_mesh.faces)

        object_triangles = object_verts[object_faces[object_template["triangle_ids"]]]
        object_points_np = np.sum(object_template["barycentric_coords"][:, :, np.newaxis] * object_triangles, axis=1)
        object_points = torch.from_numpy(object_points_np.reshape(self.submesh_num, self.points_per_submesh, 3)).float()

        return object_points, object_name


class DRODataset4OnlyCMap(Dataset):
    def __init__(self, mash_root: str, cmap_root: str) -> None:
        self.mash_root = mash_root
        self.cmap_root = cmap_root
        self.mash_file_names = [Path(name).stem.rsplit('_', 1)[0] for name in os.listdir(mash_root)]
    
    def __len__(self) -> int:
        return len(self.mash_file_names)

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor]:
        hand_cmap = np.load(os.path.join(self.cmap_root, f"{self.mash_file_names[idx]}_hand_cmap.npy"))
        hand_cmap = torch.from_numpy(hand_cmap).float()
        return os.path.join(self.mash_root, f"{self.mash_file_names[idx]}_hoi.npy"), hand_cmap


if __name__ == "__main__":
    cmap_dataset = DRODataset4OnlyCMap(
        mash_root="/mnt/d/yxz/DATASET/HOI-Dataset/Sphere-CMapDataset/shadow/train",
        cmap_root="/mnt/d/yxz/DATASET/Original-Meshes/Sphere-CMapDataset/shadow/train",
    )
    path, cmap = cmap_dataset[0]
    import pdb
    pdb.set_trace()
