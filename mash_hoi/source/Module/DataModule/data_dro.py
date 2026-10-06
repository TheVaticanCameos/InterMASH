import os
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import pytorch_lightning as pl
import numpy as np
import h5py

from typing import Tuple, Dict, Optional, Any

from .utils_statistics import get_statistics, to_multi_hand_format, MASK_DIM, SH_DIM


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


class DROTrainDataset(Dataset):
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
        torch.Tensor,   # [N, 22] hand_params
        torch.Tensor,   # [N, 22] object_params
        torch.Tensor,   # [N, 4] cmap
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
        cmap_np = grp["cmap_params"][:]
        data_name = grp.attrs["name"]
        
        positions = (torch.from_numpy(positions_np).float() - self.statistics["mean_positions"]) / self.statistics["std_positions"]
        hand_mask = (torch.from_numpy(hand_np[:, :MASK_DIM]).float() - self.statistics["mean_hand_mask"]) / self.statistics["std_hand_mask"]
        hand_sh = (torch.from_numpy(hand_np[:, MASK_DIM:MASK_DIM+SH_DIM]).float() - self.statistics["mean_hand_sh"]) / self.statistics["std_hand_sh"]
        hand_ortho = (torch.from_numpy(hand_np[:, MASK_DIM+SH_DIM:]).float() - self.statistics["mean_hand_ortho"]) / self.statistics["std_hand_ortho"]
        hand_params = torch.cat([
            hand_mask,
            hand_sh,
            hand_ortho,
        ], dim=1)
        
        object_mask = (torch.from_numpy(object_np[:, :MASK_DIM]).float() - self.statistics["mean_obj_mask"]) / self.statistics["std_obj_mask"]
        object_sh = (torch.from_numpy(object_np[:, MASK_DIM:MASK_DIM+SH_DIM]).float() - self.statistics["mean_obj_sh"]) / self.statistics["std_obj_sh"]
        object_ortho = (torch.from_numpy(object_np[:, MASK_DIM+SH_DIM:]).float() - self.statistics["mean_obj_ortho"]) / self.statistics["std_obj_ortho"]
        object_params = torch.cat([
            object_mask,
            object_sh,
            object_ortho,
        ], dim=1)
        
        cmap_params = (torch.from_numpy(cmap_np, dtype=torch.float32) - self.statistics["mean_cmap"]) / self.statistics["std_cmap"]
        
        return positions, hand_params, object_params, cmap_params, self.template_hand_params, data_name, self.statistics


class DROTestDataset(Dataset):
    def __init__(self, data_path: str, template_mash_path: str, statistics: Dict[str, torch.Tensor]):
        super().__init__()
        self.data_path = data_path
        with h5py.File(self.data_path, 'r') as hf:
            self.length = len(hf.keys())
        self.statistics = statistics
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
        torch.Tensor,   # [N, 22] object_params
        torch.Tensor,   # [N, 22] template_hand_params
        str,            # data_name
        Dict[str, torch.Tensor] # statistics
    ]:
        if self._hf is None:
            self._hf = h5py.File(self.data_path, 'r', swmr=True)
        grp = self._hf[str(idx)]
        positions_np = grp["positions"][:]
        object_np = grp["object_params"][:]
        data_name = grp.attrs["name"]
        
        positions = (torch.from_numpy(positions_np, dtype=torch.float32) - self.statistics["mean_positions"]) / self.statistics["std_positions"]
        
        object_mask = (torch.from_numpy(object_np[:, :MASK_DIM]).float() - self.statistics["mean_obj_mask"]) / self.statistics["std_obj_mask"]
        object_sh = (torch.from_numpy(object_np[:, MASK_DIM:MASK_DIM+SH_DIM]).float() - self.statistics["mean_obj_sh"]) / self.statistics["std_obj_sh"]
        object_ortho = (torch.from_numpy(object_np[:, MASK_DIM+SH_DIM:]).float() - self.statistics["mean_obj_ortho"]) / self.statistics["std_obj_ortho"]
        object_params = torch.cat([
            object_mask,
            object_sh,
            object_ortho,
        ], dim=1)
        
        return positions, object_params, self.template_hand_params, data_name, self.statistics


class DROTrainDataModule(pl.LightningDataModule):
    def __init__(self, data_root: str, template_mash_root: str, batch_size: int, num_workers: int) -> None:
        super().__init__()
        self.shadow_data_path = os.path.join(data_root, "shadow_train_data.h5")
        self.allegro_data_path = os.path.join(data_root, "allegro_train_data.h5")
        self.barrett_data_path = os.path.join(data_root, "barrett_train_data.h5")

        self.shadow_template_path = os.path.join(template_mash_root, "mash_shadow_template.npy")
        self.allegro_template_path = os.path.join(template_mash_root, "mash_allegro_template.npy")
        self.barrett_template_path = os.path.join(template_mash_root, "mash_barrett_template.npy")

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
            self.shadow_train_dataset = DROTrainDataset(
                data_path=self.shadow_data_path,
                template_mash_path=self.shadow_template_path,
                statistics=self.shadow_statistics,
            )
            self.allegro_train_dataset = DROTrainDataset(
                data_path=self.allegro_data_path,
                template_mash_path=self.allegro_template_path,
                statistics=self.allegro_statistics,
            )
            self.barrett_train_dataset = DROTrainDataset(
                data_path=self.barrett_data_path,
                template_mash_path=self.barrett_template_path,
                statistics=self.barrett_statistics,
            )
            self.train_dataset = ConcatDataset([
                self.shadow_train_dataset,
                self.allegro_train_dataset,
                self.barrett_train_dataset,
            ])

            self.shadow_val_dataset = DROTrainDataset(
                data_path=self.shadow_data_path,
                template_mash_path=self.shadow_template_path,
                statistics=self.shadow_statistics,
            )
            self.allegro_val_dataset = DROTrainDataset(
                data_path=self.allegro_data_path,
                template_mash_path=self.allegro_template_path,
                statistics=self.allegro_statistics,
            )
            self.barrett_val_dataset = DROTrainDataset(
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


class DROTestDataModule(pl.LightningDataModule):
    def __init__(self, data_root: str, template_mash_root: str, batch_size: int, num_workers: int, hand_type: str) -> None:
        super().__init__()
        self.hand_type = hand_type
        assert hand_type in ["shadow", "allegro", "barrett"], f"ERROR: unknown hand type {hand_type}!"

        self.data_path = os.path.join(data_root, f"{hand_type}_test_data.h5")
        self.stat_path = os.path.join(data_root, f"{hand_type}_statistics.npy")
        self.template_path = os.path.join(template_mash_root, f"mash_{hand_type}_template.npy")
        
        self.statistics = to_multi_hand_format(np.load(self.stat_path, allow_pickle=True)[()])

        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "test" or stage is None:
            self.test_dataset = DROTestDataset(
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
