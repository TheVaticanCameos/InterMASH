"""
Graph-aware GRAB Dataset with precomputed KNN indices.

"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import h5py
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from mash_hoi.source.Module.DataModule.data_dgn_sphere import extract_object_name
from mash_hoi.source.Module.DataModule.utils_statistics import get_statistics
from mash_hoi.source.Utils.constant import MASK_DIM, ORTHO_DIM, SH_DIM


K_NEIGHBORS = 8


class GraphGRABTrainDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        template_mash_path: str,
        statistics: Dict,
        k_neighbors: int = K_NEIGHBORS,
    ):
        super().__init__()
        self.data_path = data_path
        self.k_neighbors = k_neighbors

        with h5py.File(self.data_path, "r") as hf:
            self.length = len(hf.keys())

        self.statistics = statistics

        template_mash: Dict[str, Any] = np.load(template_mash_path, allow_pickle=True)[()]
        hand_mask_params = (template_mash["hand_params"]["mask_params"] - self.statistics["hand"]["mask_params"]["mean"]) / self.statistics["hand"]["mask_params"]["std"]
        hand_sh_params = (template_mash["hand_params"]["sh_params"] - self.statistics["hand"]["sh_params"]["mean"]) / self.statistics["hand"]["sh_params"]["std"]
        hand_ortho_poses = (template_mash["hand_params"]["ortho_poses"] - self.statistics["hand"]["ortho_poses"]["mean"]) / self.statistics["hand"]["ortho_poses"]["std"]
        self.template_hand_params = torch.cat(
            [
                torch.from_numpy(hand_mask_params).float(),
                torch.from_numpy(hand_sh_params).float(),
                torch.from_numpy(hand_ortho_poses).float(),
            ],
            dim=1,
        )

        data_dir = os.path.dirname(self.data_path)
        self.precomputed_path = os.path.join(data_dir, "object_pointclouds.npz")
        self.object_knn_path = os.path.join(data_dir, "object_knn.npz")
        self.hand_knn_path = os.path.join(data_dir, "template_hand_knn.npz")

        for path in [self.precomputed_path, self.object_knn_path, self.hand_knn_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Required file not found: {path}\n"
                    f"Please run: python mash_hoi/Preprocess/precompute_object_pointclouds.py "
                    f"--data_root {data_dir} --template_mash_path <path_to_template>"
                )

        self._precomputed_data = None
        self._object_knn_data = None
        hand_knn_npz = np.load(self.hand_knn_path, allow_pickle=True)
        self.template_hand_knn = torch.from_numpy(hand_knn_npz["knn_indices"]).long()

        self._object_cache: Dict[str, Dict[str, torch.Tensor]] = {}
        self._hf = None

    def _ensure_precomputed_loaded(self) -> None:
        if self._precomputed_data is None:
            self._precomputed_data = np.load(self.precomputed_path, allow_pickle=True)
        if self._object_knn_data is None:
            self._object_knn_data = np.load(self.object_knn_path, allow_pickle=True)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_hf"] = None
        state["_precomputed_data"] = None
        state["_object_knn_data"] = None
        return state

    def __del__(self):
        try:
            if getattr(self, "_hf", None) is not None:
                self._hf.close()
        except Exception:
            pass
        try:
            if getattr(self, "_precomputed_data", None) is not None:
                self._precomputed_data.close()
        except Exception:
            pass
        try:
            if getattr(self, "_object_knn_data", None) is not None:
                self._object_knn_data.close()
        except Exception:
            pass

    def __len__(self) -> int:
        return self.length

    def __getitem__(
        self,
        idx: int,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        str,
        torch.Tensor,
        torch.Tensor,
    ]:
        if self._hf is None:
            self._hf = h5py.File(self.data_path, "r", swmr=True)

        grp = self._hf[str(idx)]
        positions_np = grp["positions"][:]
        hand_np = grp["hand_params"][:]
        object_np = grp["object_params"][:]
        data_name = grp.attrs["name"]

        positions = torch.tensor(
            (positions_np - self.statistics["positions"]["mean"]) / self.statistics["positions"]["std"],
            dtype=torch.float32,
        )

        hand_mask = (hand_np[:, :MASK_DIM] - self.statistics["hand"]["mask_params"]["mean"]) / self.statistics["hand"]["mask_params"]["std"]
        hand_sh = (hand_np[:, MASK_DIM : MASK_DIM + SH_DIM] - self.statistics["hand"]["sh_params"]["mean"]) / self.statistics["hand"]["sh_params"]["std"]
        hand_ortho = (hand_np[:, MASK_DIM + SH_DIM : MASK_DIM + SH_DIM + ORTHO_DIM] - self.statistics["hand"]["ortho_poses"]["mean"]) / self.statistics["hand"]["ortho_poses"]["std"]
        hand_params = torch.cat(
            [
                torch.tensor(hand_mask, dtype=torch.float32),
                torch.tensor(hand_sh, dtype=torch.float32),
                torch.tensor(hand_ortho, dtype=torch.float32),
            ],
            dim=1,
        )

        object_mask = (object_np[:, :MASK_DIM] - self.statistics["object"]["mask_params"]["mean"]) / self.statistics["object"]["mask_params"]["std"]
        object_sh = (object_np[:, MASK_DIM : MASK_DIM + SH_DIM] - self.statistics["object"]["sh_params"]["mean"]) / self.statistics["object"]["sh_params"]["std"]
        object_ortho = (object_np[:, MASK_DIM + SH_DIM : MASK_DIM + SH_DIM + ORTHO_DIM] - self.statistics["object"]["ortho_poses"]["mean"]) / self.statistics["object"]["ortho_poses"]["std"]
        object_params = torch.cat(
            [
                torch.tensor(object_mask, dtype=torch.float32),
                torch.tensor(object_sh, dtype=torch.float32),
                torch.tensor(object_ortho, dtype=torch.float32),
            ],
            dim=1,
        )

        object_cmap = (object_np[:, MASK_DIM + SH_DIM + ORTHO_DIM :] - self.statistics["object"]["cmap_params"]["mean"]) / self.statistics["object"]["cmap_params"]["std"]
        cmap_params = torch.tensor(object_cmap, dtype=torch.float32)

        self._ensure_precomputed_loaded()
        object_name = extract_object_name(data_name)
        if object_name not in self._object_cache:
            try:
                obj_data = self._precomputed_data[object_name].item()
                obj_knn = self._object_knn_data[object_name]
            except KeyError as e:
                raise KeyError(
                    f"Object '{object_name}' not found in precomputed npz files.\n"
                    f"data_name='{data_name}'\n"
                    f"Make sure precompute_object_pointclouds.py used the same object name extraction."
                ) from e
            self._object_cache[object_name] = {
                "points": torch.from_numpy(obj_data["points"]).float(),
                "normals": torch.from_numpy(obj_data["normals"]).float(),
                "knn": torch.from_numpy(obj_knn).long(),
            }

        obj_points = self._object_cache[object_name]["points"]
        obj_normals = self._object_cache[object_name]["normals"]
        obj_knn = self._object_cache[object_name]["knn"]
        hand_knn = self.template_hand_knn

        return (
            positions,
            hand_params,
            object_params,
            cmap_params,
            self.template_hand_params,
            hand_knn,
            obj_knn,
            data_name,
            obj_points,
            obj_normals,
        )


class GraphGRABTestDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        template_mash_path: str,
        statistics: Dict,
        k_neighbors: int = K_NEIGHBORS,
        expand_num: int = 64,
    ):
        super().__init__()
        self.data_path = data_path
        self.k_neighbors = k_neighbors
        self.expand_num = expand_num

        with h5py.File(self.data_path, "r") as hf:
            self.length = len(hf.keys())

        self.statistics = statistics

        template_mash: Dict[str, Any] = np.load(template_mash_path, allow_pickle=True)[()]
        hand_mask_params = (template_mash["hand_params"]["mask_params"] - self.statistics["hand"]["mask_params"]["mean"]) / self.statistics["hand"]["mask_params"]["std"]
        hand_sh_params = (template_mash["hand_params"]["sh_params"] - self.statistics["hand"]["sh_params"]["mean"]) / self.statistics["hand"]["sh_params"]["std"]
        hand_ortho_poses = (template_mash["hand_params"]["ortho_poses"] - self.statistics["hand"]["ortho_poses"]["mean"]) / self.statistics["hand"]["ortho_poses"]["std"]
        self.template_hand_params = torch.cat(
            [
                torch.from_numpy(hand_mask_params).float(),
                torch.from_numpy(hand_sh_params).float(),
                torch.from_numpy(hand_ortho_poses).float(),
            ],
            dim=1,
        )

        data_dir = os.path.dirname(self.data_path)
        self.precomputed_path = os.path.join(data_dir, "object_pointclouds.npz")
        self.object_knn_path = os.path.join(data_dir, "object_knn.npz")
        self.hand_knn_path = os.path.join(data_dir, "template_hand_knn.npz")

        for path in [self.precomputed_path, self.object_knn_path, self.hand_knn_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Required file not found: {path}\n"
                    f"Please run: python mash_hoi/Preprocess/precompute_object_pointclouds.py "
                    f"--data_root {data_dir} --template_mash_path <path_to_template>"
                )

        self._precomputed_data = None
        self._object_knn_data = None
        hand_knn_npz = np.load(self.hand_knn_path, allow_pickle=True)
        self.template_hand_knn = torch.from_numpy(hand_knn_npz["knn_indices"]).long()

        self._object_cache: Dict[str, Dict[str, torch.Tensor]] = {}
        self._hf = None

    def _ensure_precomputed_loaded(self) -> None:
        if self._precomputed_data is None:
            self._precomputed_data = np.load(self.precomputed_path, allow_pickle=True)
        if self._object_knn_data is None:
            self._object_knn_data = np.load(self.object_knn_path, allow_pickle=True)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_hf"] = None
        state["_precomputed_data"] = None
        state["_object_knn_data"] = None
        return state

    def __del__(self):
        try:
            if getattr(self, "_hf", None) is not None:
                self._hf.close()
        except Exception:
            pass
        try:
            if getattr(self, "_precomputed_data", None) is not None:
                self._precomputed_data.close()
        except Exception:
            pass
        try:
            if getattr(self, "_object_knn_data", None) is not None:
                self._object_knn_data.close()
        except Exception:
            pass

    def __len__(self) -> int:
        return self.length * self.expand_num

    def __getitem__(
        self,
        idx: int,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        str,
        torch.Tensor,
        torch.Tensor,
    ]:
        data_count = idx // self.length
        idx = idx % self.length

        if self._hf is None:
            self._hf = h5py.File(self.data_path, "r", swmr=True)

        grp = self._hf[str(idx)]
        positions_np = grp["positions"][:]
        object_np = grp["object_params"][:]
        object_name_base = Path(grp.attrs["name"]).stem
        data_name = object_name_base + f"_{data_count}"

        positions = torch.tensor(
            (positions_np - self.statistics["positions"]["mean"]) / self.statistics["positions"]["std"],
            dtype=torch.float32,
        )

        object_mask = (object_np[:, :MASK_DIM] - self.statistics["object"]["mask_params"]["mean"]) / self.statistics["object"]["mask_params"]["std"]
        object_sh = (object_np[:, MASK_DIM : MASK_DIM + SH_DIM] - self.statistics["object"]["sh_params"]["mean"]) / self.statistics["object"]["sh_params"]["std"]
        object_ortho = (object_np[:, MASK_DIM + SH_DIM : MASK_DIM + SH_DIM + ORTHO_DIM] - self.statistics["object"]["ortho_poses"]["mean"]) / self.statistics["object"]["ortho_poses"]["std"]
        object_params = torch.cat(
            [
                torch.tensor(object_mask, dtype=torch.float32),
                torch.tensor(object_sh, dtype=torch.float32),
                torch.tensor(object_ortho, dtype=torch.float32),
            ],
            dim=1,
        )

        self._ensure_precomputed_loaded()
        object_name = extract_object_name(object_name_base)
        if object_name not in self._object_cache:
            try:
                obj_data = self._precomputed_data[object_name].item()
                obj_knn = self._object_knn_data[object_name]
            except KeyError as e:
                raise KeyError(
                    f"Object '{object_name}' not found in precomputed npz files.\n"
                    f"object_name_base='{object_name_base}'\n"
                    f"Make sure precompute_object_pointclouds.py used the same object name extraction."
                ) from e
            self._object_cache[object_name] = {
                "points": torch.from_numpy(obj_data["points"]).float(),
                "normals": torch.from_numpy(obj_data["normals"]).float(),
                "knn": torch.from_numpy(obj_knn).long(),
            }

        obj_points = self._object_cache[object_name]["points"]
        obj_normals = self._object_cache[object_name]["normals"]
        obj_knn = self._object_cache[object_name]["knn"]
        hand_knn = self.template_hand_knn

        return (
            positions,
            object_params,
            self.template_hand_params,
            hand_knn,
            obj_knn,
            data_name,
            obj_points,
            obj_normals,
        )


class GraphGRABDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_root: str,
        template_mash_path: str,
        batch_size: int,
        num_workers: int,
        k_neighbors: int = K_NEIGHBORS,
    ) -> None:
        super().__init__()
        self.data_root = data_root
        self.template_mash_path = template_mash_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.k_neighbors = k_neighbors

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
            self.train_dataset = GraphGRABTrainDataset(
                self.train_path,
                self.template_mash_path,
                self.statistics,
                self.k_neighbors,
            )
            self.val_dataset = GraphGRABTrainDataset(
                self.train_path,
                self.template_mash_path,
                self.statistics,
                self.k_neighbors,
            )
        elif stage == "test":
            self.test_dataset = GraphGRABTestDataset(
                self.test_path,
                self.template_mash_path,
                self.statistics,
                self.k_neighbors,
            )

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
    dataset = GraphGRABTrainDataset(
        data_path="/data2/yxz/DATASET/HOI-Dataset/HandCMap-Sphere-GRAB/train_data.h5",
        template_mash_path="/data2/yxz/DATASET/template_info/mash_sphere_mano_template.npy",
        statistics=np.load("/data2/yxz/DATASET/HOI-Dataset/HandCMap-Sphere-GRAB/statistics.npy", allow_pickle=True)[()],
        k_neighbors=K_NEIGHBORS,
    )
    positions, hand_params, object_params, cmap_params, template_hand_params,hand_knn, obj_knn, data_name, obj_points, obj_normals = dataset[0]
    import pdb
    pdb.set_trace()
