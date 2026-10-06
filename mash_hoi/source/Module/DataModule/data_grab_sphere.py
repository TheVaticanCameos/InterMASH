import os
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import numpy as np
from pathlib import Path
import h5py

from typing import Tuple, Optional, Dict, Any

from mash_hoi.source.Module.DataModule.utils_statistics import get_statistics
from mash_hoi.source.Utils.constant import MASK_DIM, SH_DIM, ORTHO_DIM


"""
Dataset & DataModule for GRAB dataset, using Sphere Anchors

"""


class SphereGRABTrainDataset(Dataset):
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
        torch.Tensor,   # [N, 26] hand_params
        torch.Tensor,   # [N, 26] object_params
        torch.Tensor,   # [N, 22] template_hand_params
        str             # data_name
    ]:
        if self._hf is None:
            self._hf = h5py.File(self.data_path, 'r', swmr=True)
        grp = self._hf[str(idx)]
        positions_np = grp["positions"][:]
        hand_np = grp["hand_params"][:]
        object_np = grp["object_params"][:]
        data_name = grp.attrs["name"]
        
        positions = torch.tensor((positions_np - self.statistics["positions"]["mean"]) / self.statistics["positions"]["std"], dtype=torch.float32)
        
        hand_mask = (hand_np[:, :MASK_DIM] - self.statistics["hand"]["mask_params"]["mean"]) / self.statistics["hand"]["mask_params"]["std"]
        hand_sh = (hand_np[:, MASK_DIM:MASK_DIM+SH_DIM] - self.statistics["hand"]["sh_params"]["mean"]) / self.statistics["hand"]["sh_params"]["std"]
        hand_ortho = (hand_np[:, MASK_DIM+SH_DIM:MASK_DIM+SH_DIM+ORTHO_DIM] - self.statistics["hand"]["ortho_poses"]["mean"]) / self.statistics["hand"]["ortho_poses"]["std"]
        hand_cmap = (hand_np[:, MASK_DIM+SH_DIM+ORTHO_DIM:] - self.statistics["hand"]["cmap_params"]["mean"]) / self.statistics["hand"]["cmap_params"]["std"]
        hand_params = torch.cat([
            torch.tensor(hand_mask, dtype=torch.float32),
            torch.tensor(hand_sh, dtype=torch.float32),
            torch.tensor(hand_ortho, dtype=torch.float32),
            torch.tensor(hand_cmap, dtype=torch.float32),
        ], dim=1)
        
        object_mask = (object_np[:, :MASK_DIM] - self.statistics["object"]["mask_params"]["mean"]) / self.statistics["object"]["mask_params"]["std"]
        object_sh = (object_np[:, MASK_DIM:MASK_DIM+SH_DIM] - self.statistics["object"]["sh_params"]["mean"]) / self.statistics["object"]["sh_params"]["std"]
        object_ortho = (object_np[:, MASK_DIM+SH_DIM:MASK_DIM+SH_DIM+ORTHO_DIM] - self.statistics["object"]["ortho_poses"]["mean"]) / self.statistics["object"]["ortho_poses"]["std"]
        object_cmap = (object_np[:, MASK_DIM+SH_DIM+ORTHO_DIM:] - self.statistics["object"]["cmap_params"]["mean"]) / self.statistics["object"]["cmap_params"]["std"]
        object_params = torch.cat([
            torch.tensor(object_mask, dtype=torch.float32),
            torch.tensor(object_sh, dtype=torch.float32),
            torch.tensor(object_ortho, dtype=torch.float32),
            torch.tensor(object_cmap, dtype=torch.float32),
        ], dim=1)

        return positions, hand_params, object_params, self.template_hand_params, data_name


class SphereGRABTestDataset(Dataset):
    def __init__(self, data_path: str, template_mash_path: str, statistics: Dict, expand_num: int = 20):
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
        str             # data_name
    ]:
        data_count = idx // self.length
        idx = idx % self.length
        if self._hf is None:
            self._hf = h5py.File(self.data_path, 'r', swmr=True)
        grp = self._hf[str(idx)]
        positions_np = grp["positions"][:]
        object_np = grp["object_params"][:]
        data_name = Path(grp.attrs["name"]).stem + f"_{data_count}"
        
        positions = torch.tensor((positions_np - self.statistics["positions"]["mean"]) / self.statistics["positions"]["std"], dtype=torch.float32)
        
        object_mask = (object_np[:, :MASK_DIM] - self.statistics["object"]["mask_params"]["mean"]) / self.statistics["object"]["mask_params"]["std"]
        object_sh = (object_np[:, MASK_DIM:MASK_DIM+SH_DIM] - self.statistics["object"]["sh_params"]["mean"]) / self.statistics["object"]["sh_params"]["std"]
        object_ortho = (object_np[:, MASK_DIM+SH_DIM:MASK_DIM+SH_DIM+ORTHO_DIM] - self.statistics["object"]["ortho_poses"]["mean"]) / self.statistics["object"]["ortho_poses"]["std"]
        object_params = torch.cat([
            torch.tensor(object_mask, dtype=torch.float32),
            torch.tensor(object_sh, dtype=torch.float32),
            torch.tensor(object_ortho, dtype=torch.float32),
        ], dim=1)
        
        return positions, object_params, self.template_hand_params, data_name


class SphereGRABDataModule(pl.LightningDataModule):
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
            self.train_dataset = SphereGRABTrainDataset(self.train_path, self.template_mash_path, self.statistics)
            self.val_dataset = SphereGRABTrainDataset(self.train_path, self.template_mash_path, self.statistics)
        
        elif stage == "test":
            self.test_dataset = SphereGRABTestDataset(self.test_path, self.template_mash_path, self.statistics)
    
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
