import os
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import pytorch_lightning as pl
import numpy as np
from pathlib import Path
import h5py

from typing import Tuple, Dict, Optional, Any

from mash_hoi.source.Module.DataModule.utils_statistics import get_statistics, to_multi_hand_format
from mash_hoi.source.Utils.constant import MASK_DIM, SH_DIM, ORTHO_DIM, CMAP_DIM


""" [For Develops]

In this script, we define the Dataset for DRO-Sphere with both hand-cmap and object-cmap.
The corresponding `.h5` file consists of the following structure:

 - "positions": (N, 3)
 - "hand_params": (N, MASK_DIM + SH_DIM + ORTHO_DIM + CMAP_DIM)
 - "object_params": (N, MASK_DIM + SH_DIM + ORTHO_DIM + CMAP_DIM)

FYI: $PROJ_HOME/ma-sh/dataset/npy2h5.py

"""


""" [For Developers]

In order to inform the model of the hand type, which is necessary for collision loss computation, we must output the statistics of the current hand in the Dataset.

Besides, for computational efficiency, we want this hand type info to be straight-forward to use, (such as directly output the hand_template and statistics), so as to avoid `if-else` statements in the model, which prohibites the use of `model.compile()`.

The dict of statistics should be like: [str, Tensor]
    "mean_positions":   (3,)
    "std_positions":    (3,)

    "mean_hand_mask":   (7,)
    "std_hand_mask":    (7,)
    "mean_hand_sh":     (9,)
    "std_hand_sh":      (9,)
    "mean_hand_ortho":  (6,)
    "std_hand_ortho":   (6,)
    
    "mean_obj_mask":    (7,)
    "std_obj_mask":     (7,)
    "mean_obj_sh":      (9,)
    "std_obj_sh":       (9,)
    "mean_obj_ortho":   (6,)
    "std_obj_ortho":    (6,)

    "mean_cmap":        (4,)
    "std_cmap":         (4,)

"""


class SphereDROTrainDataset(Dataset):
    def __init__(self, data_path: str, template_mash_path: str, statistics: Dict[str, torch.Tensor]):
        super().__init__()
        self.data_path = data_path
        with h5py.File(self.data_path, 'r') as hf:
            self.length = len(hf.keys())
        self.statistics: Dict[str, torch.Tensor] = statistics
        for key, value in self.statistics.items():
            assert type(value) is torch.Tensor
        
        template_mash: Dict[str, Any] = np.load(template_mash_path, allow_pickle=True)[()]
        
        hand_mask_params = torch.from_numpy(template_mash["hand_params"]["mask_params"]).float()
        hand_mask_params = (hand_mask_params - statistics["mean_hand_mask"]) / statistics["std_hand_mask"]
        hand_sh_params = torch.from_numpy(template_mash["hand_params"]["sh_params"]).float()
        hand_sh_params = (hand_sh_params - statistics["mean_hand_sh"]) / statistics["std_hand_sh"]
        hand_ortho_poses = torch.from_numpy(template_mash["hand_params"]["ortho_poses"]).float()
        hand_ortho_poses = (hand_ortho_poses - statistics["mean_hand_ortho"]) / statistics["std_hand_ortho"]
        
        self.template_hand_params = torch.cat([
            hand_mask_params,
            hand_sh_params,
            hand_ortho_poses,
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
        torch.Tensor,   # [N, 26] hand_params (22 + 4)
        torch.Tensor,   # [N, 26] object_params (22 + 4)
        torch.Tensor,   # [N, 22] template_hand_params
        str,            # data_name
        Dict[str, torch.Tensor] # statistics
    ]:
        if self._hf is None:
            self._hf = h5py.File(self.data_path, 'r', swmr=True)
        grp = self._hf[str(idx)]
        positions_np = grp["positions"][:]
        hand_np = grp["hand_params"][:]
        object_np = grp["object_params"][:]
        data_name = grp.attrs["name"]
        
        positions = (torch.from_numpy(positions_np).float() - self.statistics["mean_positions"]) / self.statistics["std_positions"]
        
        hand_mask = (torch.from_numpy(hand_np[:, :MASK_DIM]).float() - self.statistics["mean_hand_mask"]) / self.statistics["std_hand_mask"]
        hand_sh = (torch.from_numpy(hand_np[:, MASK_DIM:MASK_DIM+SH_DIM]).float() - self.statistics["mean_hand_sh"]) / self.statistics["std_hand_sh"]
        hand_ortho = (torch.from_numpy(hand_np[:, MASK_DIM+SH_DIM:MASK_DIM+SH_DIM+ORTHO_DIM]).float() - self.statistics["mean_hand_ortho"]) / self.statistics["std_hand_ortho"]
        hand_cmap = (torch.from_numpy(hand_np[:, MASK_DIM+SH_DIM+ORTHO_DIM:]).float() - self.statistics["mean_hand_cmap"]) / self.statistics["std_hand_cmap"]
        hand_params = torch.cat([
            hand_mask,
            hand_sh,
            hand_ortho,
            hand_cmap
        ], dim=1)
        
        object_mask = (torch.from_numpy(object_np[:, :MASK_DIM]).float() - self.statistics["mean_obj_mask"]) / self.statistics["std_obj_mask"]
        object_sh = (torch.from_numpy(object_np[:, MASK_DIM:MASK_DIM+SH_DIM]).float() - self.statistics["mean_obj_sh"]) / self.statistics["std_obj_sh"]
        object_ortho = (torch.from_numpy(object_np[:, MASK_DIM+SH_DIM:MASK_DIM+SH_DIM+ORTHO_DIM]).float() - self.statistics["mean_obj_ortho"]) / self.statistics["std_obj_ortho"]
        object_cmap = (torch.from_numpy(object_np[:, MASK_DIM+SH_DIM+ORTHO_DIM:]).float() - self.statistics["mean_obj_cmap"]) / self.statistics["std_obj_cmap"]
        object_params = torch.cat([
            object_mask,
            object_sh,
            object_ortho,
            object_cmap
        ], dim=1)
        
        return positions, hand_params, object_params, self.template_hand_params, data_name, self.statistics


class SphereDROTestDataset(Dataset):
    def __init__(self, data_path: str, template_mash_path: str, statistics: Dict[str, torch.Tensor], expand_num: int = 100):
        super().__init__()
        self.data_path = data_path
        with h5py.File(self.data_path, 'r') as hf:
            self.length = len(hf.keys())
        self.statistics: Dict[str, torch.Tensor] = statistics
        for key, value in self.statistics.items():
            assert type(value) is torch.Tensor
        
        template_mash: Dict[str, Any] = np.load(template_mash_path, allow_pickle=True)[()]
        hand_mask_params = (torch.from_numpy(template_mash["hand_params"]["mask_params"]).float() - self.statistics["mean_hand_mask"]) / self.statistics["std_hand_mask"]
        hand_sh_params = (torch.from_numpy(template_mash["hand_params"]["sh_params"]).float() - self.statistics["mean_hand_sh"]) / self.statistics["std_hand_sh"]
        hand_ortho_poses = (torch.from_numpy(template_mash["hand_params"]["ortho_poses"]).float() - self.statistics["mean_hand_ortho"]) / self.statistics["std_hand_ortho"]
        self.template_hand_params = torch.cat([
            hand_mask_params,
            hand_sh_params,
            hand_ortho_poses,
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
        str,            # data_name
        Dict[str, torch.Tensor],    # statistics
    ]:
        data_count = idx // self.length
        idx = idx % self.length
        if self._hf is None:
            self._hf = h5py.File(self.data_path, 'r', swmr=True)
        grp = self._hf[str(idx)]
        positions_np = grp["positions"][:]
        object_np = grp["object_params"][:]
        data_name = Path(grp.attrs["name"]).stem + f"_{data_count}"
        
        positions = (torch.from_numpy(positions_np).float() - self.statistics["mean_positions"]) / self.statistics["std_positions"]
        
        object_mask = (torch.from_numpy(object_np[:, :MASK_DIM]).float() - self.statistics["mean_obj_mask"]) / self.statistics["std_obj_mask"]
        object_sh = (torch.from_numpy(object_np[:, MASK_DIM:MASK_DIM+SH_DIM]).float() - self.statistics["mean_obj_sh"]) / self.statistics["std_obj_sh"]
        object_ortho = (torch.from_numpy(object_np[:, MASK_DIM+SH_DIM:MASK_DIM+SH_DIM+ORTHO_DIM]).float() - self.statistics["mean_obj_ortho"]) / self.statistics["std_obj_ortho"]
        object_params = torch.cat([
            object_mask,
            object_sh,
            object_ortho,
        ], dim=1)
        
        return positions, object_params, self.template_hand_params, data_name, self.statistics


class SphereDROTrainDataModule(pl.LightningDataModule):
    def __init__(self, data_root: str, template_mash_root: str, batch_size: int, num_workers: int) -> None:
        super().__init__()
        self.shadow_data_path = os.path.join(data_root, "shadow_train_data.h5")
        self.allegro_data_path = os.path.join(data_root, "allegro_train_data.h5")
        self.barrett_data_path = os.path.join(data_root, "barrett_train_data.h5")

        self.shadow_template_path = os.path.join(template_mash_root, "mash_sphere_shadow_template.npy")
        self.allegro_template_path = os.path.join(template_mash_root, "mash_sphere_allegro_template.npy")
        self.barrett_template_path = os.path.join(template_mash_root, "mash_sphere_barrett_template.npy")

        self.shadow_stat_path = os.path.join(data_root, "shadow_statistics.npy")
        self.allegro_stat_path = os.path.join(data_root, "allegro_statistics.npy")
        self.barrett_stat_path = os.path.join(data_root, "barrett_statistics.npy")
        if os.path.exists(self.shadow_stat_path):
            self.shadow_statistics = to_multi_hand_format(np.load(self.shadow_stat_path, allow_pickle=True)[()])
        else:
            self.shadow_statistics = get_statistics(self.shadow_data_path)
            np.save(self.shadow_stat_path, self.shadow_statistics, allow_pickle=True)
            self.shadow_statistics = to_multi_hand_format(self.shadow_statistics)
        if os.path.exists(self.allegro_stat_path):
            self.allegro_statistics = to_multi_hand_format(np.load(self.allegro_stat_path, allow_pickle=True)[()])
        else:
            self.allegro_statistics = get_statistics(self.allegro_data_path)
            np.save(self.allegro_stat_path, self.allegro_statistics, allow_pickle=True)
            self.allegro_statistics = to_multi_hand_format(self.allegro_statistics)
        if os.path.exists(self.barrett_stat_path):
            self.barrett_statistics = to_multi_hand_format(np.load(self.barrett_stat_path, allow_pickle=True)[()])
        else:
            self.barrett_statistics = get_statistics(self.barrett_data_path)
            np.save(self.barrett_stat_path, self.barrett_statistics, allow_pickle=True)
            self.barrett_statistics = to_multi_hand_format(self.barrett_statistics)

        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            self.shadow_train_dataset = SphereDROTrainDataset(
                data_path=self.shadow_data_path,
                template_mash_path=self.shadow_template_path,
                statistics=self.shadow_statistics,
            )
            self.allegro_train_dataset = SphereDROTrainDataset(
                data_path=self.allegro_data_path,
                template_mash_path=self.allegro_template_path,
                statistics=self.allegro_statistics,
            )
            self.barrett_train_dataset = SphereDROTrainDataset(
                data_path=self.barrett_data_path,
                template_mash_path=self.barrett_template_path,
                statistics=self.barrett_statistics,
            )
            self.train_dataset = ConcatDataset([
                self.shadow_train_dataset,
                self.allegro_train_dataset,
                self.barrett_train_dataset,
            ])

            self.shadow_val_dataset = SphereDROTrainDataset(
                data_path=self.shadow_data_path,
                template_mash_path=self.shadow_template_path,
                statistics=self.shadow_statistics,
            )
            self.allegro_val_dataset = SphereDROTrainDataset(
                data_path=self.allegro_data_path,
                template_mash_path=self.allegro_template_path,
                statistics=self.allegro_statistics,
            )
            self.barrett_val_dataset = SphereDROTrainDataset(
                data_path=self.barrett_data_path,
                template_mash_path=self.barrett_template_path,
                statistics=self.barrett_statistics,
            )
            self.val_dataset = ConcatDataset([
                self.shadow_val_dataset,
                self.allegro_val_dataset,
                self.barrett_val_dataset,
            ])

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )


class SphereDROTestDataModule(pl.LightningDataModule):
    def __init__(self, data_root: str, template_mash_root: str, batch_size: int, num_workers: int, hand_type: str) -> None:
        super().__init__()
        self.hand_type = hand_type
        assert hand_type in ["shadow", "allegro", "barrett"], f"ERROR: unknown hand type {hand_type}!"

        self.data_path = os.path.join(data_root, f"test_data.h5")
        self.stat_path = os.path.join(data_root, f"{hand_type}_statistics.npy")
        self.template_path = os.path.join(template_mash_root, f"mash_sphere_{hand_type}_template.npy")

        self.statistics = to_multi_hand_format(np.load(self.stat_path, allow_pickle=True)[()])

        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "test" or stage is None:
            self.test_dataset = SphereDROTestDataset(
                data_path=self.data_path, 
                template_mash_path=self.template_path,
                statistics=self.statistics,
            )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )


if __name__ == "__main__":
    datamodule = SphereDROTrainDataModule(
        data_root="/mnt/d/yxz/DATASET/HOI-Dataset/HandCMap-Sphere-CMapDataset",
        template_mash_root="/mnt/d/yxz/DATASET/template_info",
        batch_size=32,
        num_workers=4,
    )
    datamodule.setup(stage="fit")
    print(datamodule.train_dataset[0])
