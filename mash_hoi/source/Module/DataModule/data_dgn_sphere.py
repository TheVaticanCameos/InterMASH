import os
import re
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import numpy as np
from pathlib import Path
import h5py

from typing import Tuple, Optional, Dict, Any

from mash_hoi.source.Module.DataModule.utils_statistics import get_statistics, MASK_DIM, SH_DIM


"""
Dataset & DataModule for DexGraspNet dataset, using Sphere Anchors

[Note] This is the current used version of dataset for DexGraspNet.
       The Disk Anchor version is now deprecated. 

"""


def extract_object_name(data_name: str) -> str:
    """
    Extract object name from data name string.
    
    Examples:
        'shadow_mujoco-Room_Essentials_Salad_Plate_Turquoise_164_hoi.npy' 
            -> 'Room_Essentials_Salad_Plate_Turquoise'
        'shadow_mujoco-SAMOA_25_hoi.npy' -> 'SAMOA'
        'SANDWICH_MEAL_100_hoi' -> 'SANDWICH_MEAL'
    
    Args:
        data_name: Original data name from h5 file
        
    Returns:
        Extracted object name
    """
    # Pattern: optional prefix ending with '-', then object name, then '_NUMBER_hoi' and optional '.npy'
    match = re.match(r'(?:.*?-)?(.*?)_\d+_hoi(?:\.npy)?$', data_name)
    if match:
        return match.group(1)
    else:
        # Fallback: use stem (remove extension)
        return Path(data_name).stem


class SphereDGNTrainDataset(Dataset):
    def __init__(self, data_path: str, template_mash_path: str, statistics: Dict):
        super().__init__()
        self.data_path = data_path
        with h5py.File(self.data_path, 'r') as hf:
            self.length = len(hf.keys())
        self.statistics = statistics
        template_mash: Dict[str, Any] = np.load(template_mash_path, allow_pickle=True)[()]
        hand_mask_params = (template_mash["hand_params"]["mask_params"] - self.statistics["hand"]["mask_params"]["mean"]) / self.statistics["hand"]["mask_params"]["std"]
        hand_sh_params = (template_mash["hand_params"]["sh_params"] - self.statistics["hand"]["sh_params"]["mean"]) / self.statistics["hand"]["sh_params"]["std"]
        hand_ortho_poses = (template_mash["hand_params"]["ortho_poses"] - self.statistics["hand"]["ortho_poses"]["mean"]) / self.statistics["hand"]["ortho_poses"]["std"]
        self.template_hand_params = torch.cat([
            torch.from_numpy(hand_mask_params).float(),
            torch.from_numpy(hand_sh_params).float(),
            torch.from_numpy(hand_ortho_poses).float()
        ], dim=1)
        self._hf = None
        
        # Load pre-computed object data (required)
        self.precomputed_path = os.path.join(
            os.path.dirname(self.data_path), 
            "object_pointclouds.npz"
        )
        
        if not os.path.exists(self.precomputed_path):
            raise FileNotFoundError(
                f"Pre-computed object data not found at {self.precomputed_path}\n"
                f"Please run: python mash_hoi/Preprocess/precompute_object_pointclouds.py "
                f"--data_root {os.path.dirname(self.data_path)}"
            )
        
        # Lazy loading: Load NPZ file handle (not data)
        self._precomputed_data = np.load(self.precomputed_path, allow_pickle=True)
        self._object_cache = {}  # In-memory cache for loaded objects
        print(f"[INFO] Loaded pre-computed object data from {self.precomputed_path}")
    
    def __getstate__(self):
        state = self.__dict__.copy()
        state['_hf'] = None
        return state
    
    def __del__(self):
        try:
            if getattr(self, "_hf", None) is not None:
                self._hf.close()
        except Exception:
            pass
    
    def __len__(self) -> int:
        return self.length
    
    def __getitem__(
        self, 
        idx: int
    ) -> Tuple[
        torch.Tensor,   # [N, 3] positions
        torch.Tensor,   # [N, 22] hand_params
        torch.Tensor,   # [N, 22] object_params
        torch.Tensor,   # [N, 4] cmap
        torch.Tensor,   # [N, 22] template_hand_params
        str,            # data_name
        torch.Tensor,   # [M, 3] object_points (pre-computed)
        torch.Tensor    # [M, 3] object_normals (pre-computed)
    ]:
        if self._hf is None:
            self._hf = h5py.File(self.data_path, 'r', swmr=True)
        grp = self._hf[str(idx)]
        positions_np = grp["positions"][:]
        hand_np = grp["hand_params"][:]
        object_np = grp["object_params"][:]
        cmap_np = grp["cmap_params"][:]
        data_name = grp.attrs["name"]
        
        positions = torch.tensor((positions_np - self.statistics["positions"]["mean"]) / self.statistics["positions"]["std"], dtype=torch.float32)
        
        hand_mask = (hand_np[:, :MASK_DIM] - self.statistics["hand"]["mask_params"]["mean"]) / self.statistics["hand"]["mask_params"]["std"]
        hand_sh = (hand_np[:, MASK_DIM:MASK_DIM+SH_DIM] - self.statistics["hand"]["sh_params"]["mean"]) / self.statistics["hand"]["sh_params"]["std"]
        hand_ortho = (hand_np[:, MASK_DIM+SH_DIM:] - self.statistics["hand"]["ortho_poses"]["mean"]) / self.statistics["hand"]["ortho_poses"]["std"]
        hand_params = torch.cat([
            torch.tensor(hand_mask, dtype=torch.float32),
            torch.tensor(hand_sh, dtype=torch.float32),
            torch.tensor(hand_ortho, dtype=torch.float32),
        ], dim=1)
        
        object_mask = (object_np[:, :MASK_DIM] - self.statistics["object"]["mask_params"]["mean"]) / self.statistics["object"]["mask_params"]["std"]
        object_sh = (object_np[:, MASK_DIM:MASK_DIM+SH_DIM] - self.statistics["object"]["sh_params"]["mean"]) / self.statistics["object"]["sh_params"]["std"]
        object_ortho = (object_np[:, MASK_DIM+SH_DIM:] - self.statistics["object"]["ortho_poses"]["mean"]) / self.statistics["object"]["ortho_poses"]["std"]
        object_params = torch.cat([
            torch.tensor(object_mask, dtype=torch.float32),
            torch.tensor(object_sh, dtype=torch.float32),
            torch.tensor(object_ortho, dtype=torch.float32),
        ], dim=1)
        
        cmap_sh = (cmap_np - self.statistics["cmap_params"]["sh_params"]["mean"]) / self.statistics["cmap_params"]["sh_params"]["std"]
        cmap_params = torch.tensor(cmap_sh, dtype=torch.float32)
        
        # Load pre-computed object points and normals
        # Extract object name (e.g., 'shadow_mujoco-SAMOA_25_hoi.npy' -> 'SAMOA')
        object_name = extract_object_name(data_name)
        if object_name not in self._object_cache:
            obj_data = self._precomputed_data[object_name].item()
            self._object_cache[object_name] = {
                'points': torch.from_numpy(obj_data['points']).float(),
                'normals': torch.from_numpy(obj_data['normals']).float()
            }
        obj_points = self._object_cache[object_name]['points']
        obj_normals = self._object_cache[object_name]['normals']

        return positions, hand_params, object_params, cmap_params, self.template_hand_params, data_name, obj_points, obj_normals


class SphereDGNTestDataset(Dataset):
    def __init__(self, data_path: str, template_mash_path: str, statistics: Dict, expand_num: int = 64):
        super().__init__()
        self.data_path = data_path
        with h5py.File(self.data_path, 'r') as hf:
            self.length = len(hf.keys())
        self.statistics = statistics
        template_mash: Dict[str, Any] = np.load(template_mash_path, allow_pickle=True)[()]
        hand_mask_params = (template_mash["hand_params"]["mask_params"] - self.statistics["hand"]["mask_params"]["mean"]) / self.statistics["hand"]["mask_params"]["std"]
        hand_sh_params = (template_mash["hand_params"]["sh_params"] - self.statistics["hand"]["sh_params"]["mean"]) / self.statistics["hand"]["sh_params"]["std"]
        hand_ortho_poses = (template_mash["hand_params"]["ortho_poses"] - self.statistics["hand"]["ortho_poses"]["mean"]) / self.statistics["hand"]["ortho_poses"]["std"]
        self.template_hand_params = torch.cat([
            torch.from_numpy(hand_mask_params).float(),
            torch.from_numpy(hand_sh_params).float(),
            torch.from_numpy(hand_ortho_poses).float()
        ], dim=1)
        self._hf = None
        self.expand_num = expand_num
        
        # Load pre-computed object data (required)
        self.precomputed_path = os.path.join(
            os.path.dirname(self.data_path), 
            "object_pointclouds.npz"
        )
        
        if not os.path.exists(self.precomputed_path):
            raise FileNotFoundError(
                f"Pre-computed object data not found at {self.precomputed_path}\n"
                f"Please run: python mash_hoi/Preprocess/precompute_object_pointclouds.py "
                f"--data_root {os.path.dirname(self.data_path)}"
            )
        
        # Lazy loading: Load NPZ file handle (not data)
        self._precomputed_data = np.load(self.precomputed_path, allow_pickle=True)
        self._object_cache = {}  # In-memory cache for loaded objects
        print(f"[INFO] Loaded pre-computed object data from {self.precomputed_path}")
    
    def __getstate__(self):
        state = self.__dict__.copy()
        state['_hf'] = None
        return state
    
    def __del__(self):
        try:
            if getattr(self, "_hf", None) is not None:
                self._hf.close()
        except Exception:
            pass
    
    def __len__(self) -> int:
        return self.length * self.expand_num
    
    def __getitem__(
        self, 
        idx: int
    ) -> Tuple[
        torch.Tensor,   # [N, 3] positions
        torch.Tensor,   # [N, 22] object_params
        torch.Tensor,   # [N, 22] template_hand_params
        str,            # data_name
        torch.Tensor,   # [M, 3] object_points (pre-computed)
        torch.Tensor    # [M, 3] object_normals (pre-computed)
    ]:
        data_count = idx // self.length
        idx = idx % self.length
        if self._hf is None:
            self._hf = h5py.File(self.data_path, 'r', swmr=True)
        grp = self._hf[str(idx)]
        positions_np = grp["positions"][:]
        object_np = grp["object_params"][:]
        object_name_base = Path(grp.attrs["name"]).stem
        data_name = object_name_base + f"_{data_count}"
        
        positions = torch.tensor((positions_np - self.statistics["positions"]["mean"]) / self.statistics["positions"]["std"], dtype=torch.float32)
        
        object_mask = (object_np[:, :MASK_DIM] - self.statistics["object"]["mask_params"]["mean"]) / self.statistics["object"]["mask_params"]["std"]
        object_sh = (object_np[:, MASK_DIM:MASK_DIM+SH_DIM] - self.statistics["object"]["sh_params"]["mean"]) / self.statistics["object"]["sh_params"]["std"]
        object_ortho = (object_np[:, MASK_DIM+SH_DIM:] - self.statistics["object"]["ortho_poses"]["mean"]) / self.statistics["object"]["ortho_poses"]["std"]
        object_params = torch.cat([
            torch.tensor(object_mask, dtype=torch.float32),
            torch.tensor(object_sh, dtype=torch.float32),
            torch.tensor(object_ortho, dtype=torch.float32),
        ], dim=1)
        
        # Load pre-computed object points and normals
        # Extract object name (e.g., 'shadow_mujoco-SAMOA_25_hoi.npy' -> 'SAMOA')
        object_name = extract_object_name(object_name_base)
        if object_name not in self._object_cache:
            obj_data = self._precomputed_data[object_name].item()
            self._object_cache[object_name] = {
                'points': torch.from_numpy(obj_data['points']).float(),
                'normals': torch.from_numpy(obj_data['normals']).float()
            }
        obj_points = self._object_cache[object_name]['points']
        obj_normals = self._object_cache[object_name]['normals']
        
        return positions, object_params, self.template_hand_params, data_name, obj_points, obj_normals


class SphereDGNDataModule(pl.LightningDataModule):
    def __init__(self, data_root: str, template_mash_path: str, batch_size: int, num_workers: int) -> None:
        super().__init__()
        self.data_root = data_root
        self.template_mash_path = template_mash_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_path = os.path.join(self.data_root, "train_data.h5")
        self.test_path = os.path.join(self.data_root, "test_data.h5")
        self.stat_path = os.path.join(self.data_root, "statistics.npy")
        if os.path.exists(self.stat_path):
            self.statistics: Dict[str, Any] = np.load(self.stat_path, allow_pickle=True)[()]
        else:
            self.statistics = get_statistics(self.train_path)
            np.save(self.stat_path, self.statistics, allow_pickle=True)
    
    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            self.train_dataset = SphereDGNTrainDataset(self.train_path, self.template_mash_path, self.statistics)
            self.val_dataset = SphereDGNTrainDataset(self.train_path, self.template_mash_path, self.statistics)
        
        elif stage == "test":
            self.test_dataset = SphereDGNTestDataset(self.test_path, self.template_mash_path, self.statistics)
    
    def train_dataloader(self) -> DataLoader:
        kwargs = {
            "batch_size": self.batch_size,
            "shuffle": True,
            "num_workers": self.num_workers,
            "pin_memory": True,
            "drop_last": True,
        }
        if self.num_workers > 0:
            kwargs["persistent_workers"] = True
            kwargs["prefetch_factor"] = 4
        return DataLoader(self.train_dataset, **kwargs)

    def val_dataloader(self) -> DataLoader:
        kwargs = {
            "batch_size": self.batch_size,
            "shuffle": False,
            "num_workers": self.num_workers,
            "pin_memory": True,
        }
        if self.num_workers > 0:
            kwargs["persistent_workers"] = True
            kwargs["prefetch_factor"] = 4
        return DataLoader(self.val_dataset, **kwargs)
    
    def test_dataloader(self) -> DataLoader:
        kwargs = {
            "batch_size": self.batch_size,
            "shuffle": False,
            "num_workers": self.num_workers,
            "pin_memory": True,
        }
        if self.num_workers > 0:
            kwargs["persistent_workers"] = True
            kwargs["prefetch_factor"] = 4
        return DataLoader(self.test_dataset, **kwargs)


if __name__ == "__main__":
    test_dataset = SphereDGNTestDataset(
        data_path="/mnt/d/yxz/DATASET/HOI-Dataset/Sphere-DexGraspNet/shadow/test_data.h5",
        template_mash_path="/mnt/d/yxz/DATASET/template_info/mash_sphere_shadow_template.npy",
        statistics=np.load("/mnt/d/yxz/DATASET/HOI-Dataset/Sphere-DexGraspNet/shadow/statistics.npy", allow_pickle=True)[()]
    )
    test_dataset[0]
