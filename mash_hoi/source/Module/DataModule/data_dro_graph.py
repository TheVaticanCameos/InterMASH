"""
Graph-aware DRO Dataset and DataModule with KNN precomputation and caching.

This module extends the base SphereDRO dataset to include:
- K-nearest neighbor (KNN) indices for both template hand and object patches
- Precomputation and caching of KNN indices for efficient training
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import pytorch_lightning as pl
import numpy as np
from pathlib import Path
import h5py
from tqdm import tqdm
import time

from typing import Tuple, Dict, Optional, Any

from .utils_statistics import get_statistics, to_multi_hand_format, MASK_DIM, SH_DIM

# Import MASH utilities for point cloud sampling
from mash_hoi.Preprocess.utils.mash_utils import (
    compute_sample_points_from_mash_params,
    split_mash_params,
)


# Constants for KNN computation
K_NEIGHBORS = 8
SAMPLE_PHI_NUM = 8
SAMPLE_THETA_NUM = 8
MASK_DEGREE_MAX = 3
SH_DEGREE_MAX = 2


def compute_knn_indices(centers: torch.Tensor, k: int = K_NEIGHBORS) -> torch.Tensor:
    """
    Compute K-nearest neighbor indices based on Euclidean distance.
    
    Args:
        centers: [N, 3] or [B, N, 3] patch center positions
        k: Number of neighbors
        
    Returns:
        knn_indices: [N, K] or [B, N, K] indices of K nearest neighbors (excluding self)
    """
    if centers.dim() == 2:
        # Single sample: [N, 3]
        dist = torch.cdist(centers, centers)  # [N, N]
        # Get k+1 nearest (including self), then exclude self
        _, indices = dist.topk(k + 1, dim=-1, largest=False)  # [N, K+1]
        return indices[:, 1:]  # [N, K], exclude self
    else:
        # Batched: [B, N, 3]
        dist = torch.cdist(centers, centers)  # [B, N, N]
        _, indices = dist.topk(k + 1, dim=-1, largest=False)  # [B, N, K+1]
        return indices[:, :, 1:]  # [B, N, K]


def compute_patch_centers_from_mash(
    mash_params: torch.Tensor,
    positions: torch.Tensor,
    statistics: Dict[str, torch.Tensor],
    entity: str = "hand",
) -> torch.Tensor:
    """
    Compute patch centers by sampling small point clouds from MASH parameters.
    
    Args:
        mash_params: [B, N, 22] or [N, 22] MASH parameters (raw, not normalized)
        positions: [B, N, 3] or [N, 3] anchor positions (raw, not normalized)
        statistics: Dictionary of statistics (not used, kept for compatibility)
        entity: "hand" or "object"
        
    Returns:
        centers: [B, N, 3] or [N, 3] patch centers
    """
    device = mash_params.device
    dtype = mash_params.dtype
    
    # Handle single sample case
    single_sample = mash_params.dim() == 2
    if single_sample:
        mash_params = mash_params.unsqueeze(0)
        positions = positions.unsqueeze(0)
    
    B, N, _ = mash_params.shape
    
    # Split parameters (data is already raw/denormalized)
    mask_params, sh_params, ortho_poses = split_mash_params(mash_params, MASK_DIM, SH_DIM)
    # No denormalization needed - data is already raw
    
    # Sample point clouds (8x8 = 64 points per patch)
    sample_points = compute_sample_points_from_mash_params(
        mask_params=mask_params,
        sh_params=sh_params,
        ortho_poses=ortho_poses,
        positions=positions,
        sample_phi_num=SAMPLE_PHI_NUM,
        sample_theta_num=SAMPLE_THETA_NUM,
        mask_degree_max=MASK_DEGREE_MAX,
        sh_degree_max=SH_DEGREE_MAX,
        target=entity,
    )  # [B, N, S_phi, S_theta, 3]
    
    # Compute centers as mean of sampled points
    centers = sample_points.reshape(B, N, -1, 3).mean(dim=2)  # [B, N, 3]
    
    if single_sample:
        centers = centers.squeeze(0)
    
    return centers


def precompute_knn_cache(
    data_path: str,
    cache_path: str,
    statistics: Dict[str, torch.Tensor],
    template_hand_params_raw: torch.Tensor,
    batch_size: int = 64,
    k: int = K_NEIGHBORS,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> None:
    """
    Precompute and cache KNN indices for all samples in the dataset.
    
    Only rank 0 process checks/creates cache, other processes wait.
    
    Args:
        data_path: Path to H5 data file
        cache_path: Path to save KNN cache H5 file
        statistics: Dictionary of statistics for denormalization
        template_hand_params_raw: Template hand params [N, 22] (raw, unnormalized).
                                  Used to compute hand KNN for all samples (consistent with inference).
        batch_size: Batch size for processing
        k: Number of neighbors
        device: Device for computation
    """
    # Check if we're in a distributed setting
    is_distributed = torch.distributed.is_initialized() if hasattr(torch.distributed, 'is_initialized') else False
    if is_distributed:
        rank = torch.distributed.get_rank()
    else:
        rank = 0
    
    # Ensure cache directory exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    
    # Only rank 0 checks/creates cache
    if rank == 0:
        # Check if cache already exists and is complete
        if os.path.exists(cache_path):
            try:
                with h5py.File(cache_path, 'r') as hf_check:
                    with h5py.File(data_path, 'r') as hf_data:
                        num_samples = len(hf_data.keys())
                        cache_samples = len(hf_check.keys())
                        if cache_samples == num_samples:
                            # Verify a few samples to ensure cache is valid
                            check_indices = [0, num_samples // 2, num_samples - 1]
                            all_valid = True
                            for idx in check_indices:
                                if str(idx) not in hf_check:
                                    all_valid = False
                                    break
                                if "hand_knn" not in hf_check[str(idx)] or "obj_knn" not in hf_check[str(idx)]:
                                    all_valid = False
                                    break
                            if all_valid:
                                print(f"KNN cache already exists and is complete: {cache_path}")
                                # Broadcast completion to other processes
                                if is_distributed:
                                    torch.distributed.barrier()
                                return
            except Exception as e:
                print(f"Error checking existing cache: {e}, will recompute")
        
        print(f"Precomputing KNN cache: {cache_path}")
        
        # Template is already raw, just move to device
        template_hand_raw = template_hand_params_raw.to(device)
        
        with h5py.File(data_path, 'r') as hf_data:
            num_samples = len(hf_data.keys())
            
            with h5py.File(cache_path, 'w') as hf_cache:
                pbar = tqdm(range(0, num_samples, batch_size), desc="Computing KNN")
                
                for start_idx in pbar:
                    end_idx = min(start_idx + batch_size, num_samples)
                    actual_batch_size = end_idx - start_idx
                    
                    # Load batch data
                    obj_params_list = []
                    positions_list = []
                    
                    for idx in range(start_idx, end_idx):
                        grp = hf_data[str(idx)]
                        obj_params_list.append(grp["object_params"][:])
                        positions_list.append(grp["positions"][:])
                    
                    # Stack to tensors (raw data from H5 file)
                    obj_params = torch.from_numpy(np.stack(obj_params_list)).float().to(device)
                    positions = torch.from_numpy(np.stack(positions_list)).float().to(device)
                    
                    # Compute patch centers (data from H5 is raw, no normalization needed)
                    with torch.no_grad():
                        obj_centers = compute_patch_centers_from_mash(
                            obj_params, positions, statistics, entity="object"
                        )  # [B, N, 3]
                        
                        # Compute object KNN indices
                        obj_knn = compute_knn_indices(obj_centers, k)    # [B, N, K]
                        
                        # Compute hand KNN indices using template hand params
                        # Expand template to batch size and use same positions
                        template_hand_batch = template_hand_raw.unsqueeze(0).expand(actual_batch_size, -1, -1)  # [B, N, 22]
                        
                        hand_centers = compute_patch_centers_from_mash(
                            template_hand_batch, positions, statistics, entity="hand"
                        )  # [B, N, 3]
                        
                        # Compute hand KNN indices
                        hand_knn = compute_knn_indices(hand_centers, k)  # [B, N, K]
                    
                    # Save to cache
                    hand_knn_np = hand_knn.cpu().numpy().astype(np.int16)
                    obj_knn_np = obj_knn.cpu().numpy().astype(np.int16)
                    
                    for i, idx in enumerate(range(start_idx, end_idx)):
                        grp = hf_cache.create_group(str(idx))
                        grp.create_dataset("hand_knn", data=hand_knn_np[i], compression="gzip")
                        grp.create_dataset("obj_knn", data=obj_knn_np[i], compression="gzip")
        
        print(f"KNN cache saved to: {cache_path}")
        
        # Broadcast completion to other processes
        if is_distributed:
            torch.distributed.barrier()
    else:
        # Non-rank-0 processes wait for cache to be created
        if is_distributed:
            torch.distributed.barrier()
        # Verify cache exists after barrier
        if not os.path.exists(cache_path):
            raise RuntimeError(f"KNN cache not found after waiting: {cache_path}")


class GraphDROTrainDataset(Dataset):
    """
    Graph-aware DRO training dataset with KNN indices.
    
    Extends SphereDROTrainDataset to include precomputed KNN indices
    for graph-aware attention in the diffusion model.
    """
    def __init__(
        self,
        data_path: str,
        template_mash_path: str,
        statistics: Dict[str, torch.Tensor],
        knn_cache_path: str,
        k_neighbors: int = K_NEIGHBORS,
        precompute_device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()
        self.data_path = data_path
        self.knn_cache_path = knn_cache_path
        self.k_neighbors = k_neighbors
        
        with h5py.File(self.data_path, 'r') as hf:
            self.length = len(hf.keys())
        
        self.statistics: Dict[str, torch.Tensor] = statistics
        for key, value in self.statistics.items():
            assert type(value) is torch.Tensor
        
        # Load template (raw)
        template_mash: Dict[str, Any] = np.load(template_mash_path, allow_pickle=True)[()]
        
        hand_mask_params_raw = torch.from_numpy(template_mash["hand_params"]["mask_params"]).float()
        hand_sh_params_raw = torch.from_numpy(template_mash["hand_params"]["sh_params"]).float()
        hand_ortho_poses_raw = torch.from_numpy(template_mash["hand_params"]["ortho_poses"]).float()
        
        # Raw template for KNN computation
        template_hand_params_raw = torch.cat([
            hand_mask_params_raw,
            hand_sh_params_raw,
            hand_ortho_poses_raw,
        ], dim=1)
        
        # Normalized template for training return
        hand_mask_params_norm = (hand_mask_params_raw - statistics["mean_hand_mask"]) / statistics["std_hand_mask"]
        hand_sh_params_norm = (hand_sh_params_raw - statistics["mean_hand_sh"]) / statistics["std_hand_sh"]
        hand_ortho_poses_norm = (hand_ortho_poses_raw - statistics["mean_hand_ortho"]) / statistics["std_hand_ortho"]
        
        self.template_hand_params = torch.cat([
            hand_mask_params_norm,
            hand_sh_params_norm,
            hand_ortho_poses_norm,
        ], dim=1)
        
        # Precompute KNN cache if not exists
        # Use raw template hand params to compute hand KNN (consistent with inference)
        if not os.path.exists(knn_cache_path):
            precompute_knn_cache(
                data_path=data_path,
                cache_path=knn_cache_path,
                statistics=statistics,
                template_hand_params_raw=template_hand_params_raw,
                k=k_neighbors,
                device=precompute_device,
            )
        
        self._data_hf = None
        self._knn_hf = None
    
    def __getstate__(self):
        state = self.__dict__.copy()
        state['_data_hf'] = None
        state['_knn_hf'] = None
        return state
    
    def __del__(self):
        try:
            if getattr(self, "_data_hf", None) is not None:
                self._data_hf.close()
            if getattr(self, "_knn_hf", None) is not None:
                self._knn_hf.close()
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
        torch.Tensor,   # [N, K] hand_knn
        torch.Tensor,   # [N, K] obj_knn
        str,            # data_name
        Dict[str, torch.Tensor]  # statistics
    ]:
        # Lazy load file handles
        if self._data_hf is None:
            self._data_hf = h5py.File(self.data_path, 'r', swmr=True)
        if self._knn_hf is None:
            self._knn_hf = h5py.File(self.knn_cache_path, 'r', swmr=True)
        
        # Load original data
        grp = self._data_hf[str(idx)]
        positions_np = grp["positions"][:]
        hand_np = grp["hand_params"][:]
        object_np = grp["object_params"][:]
        cmap_np = grp["cmap_params"][:]
        data_name = grp.attrs["name"]
        
        # Load KNN indices
        knn_grp = self._knn_hf[str(idx)]
        hand_knn = torch.from_numpy(knn_grp["hand_knn"][:]).long()
        obj_knn = torch.from_numpy(knn_grp["obj_knn"][:]).long()
        
        # Normalize data
        positions = (torch.from_numpy(positions_np).float() - self.statistics["mean_positions"]) / self.statistics["std_positions"]
        hand_mask = (torch.from_numpy(hand_np[:, :MASK_DIM]).float() - self.statistics["mean_hand_mask"]) / self.statistics["std_hand_mask"]
        hand_sh = (torch.from_numpy(hand_np[:, MASK_DIM:MASK_DIM+SH_DIM]).float() - self.statistics["mean_hand_sh"]) / self.statistics["std_hand_sh"]
        hand_ortho = (torch.from_numpy(hand_np[:, MASK_DIM+SH_DIM:]).float() - self.statistics["mean_hand_ortho"]) / self.statistics["std_hand_ortho"]
        hand_params = torch.cat([hand_mask, hand_sh, hand_ortho], dim=1)
        
        object_mask = (torch.from_numpy(object_np[:, :MASK_DIM]).float() - self.statistics["mean_obj_mask"]) / self.statistics["std_obj_mask"]
        object_sh = (torch.from_numpy(object_np[:, MASK_DIM:MASK_DIM+SH_DIM]).float() - self.statistics["mean_obj_sh"]) / self.statistics["std_obj_sh"]
        object_ortho = (torch.from_numpy(object_np[:, MASK_DIM+SH_DIM:]).float() - self.statistics["mean_obj_ortho"]) / self.statistics["std_obj_ortho"]
        object_params = torch.cat([object_mask, object_sh, object_ortho], dim=1)
        
        cmap_params = (torch.from_numpy(cmap_np).float() - self.statistics["mean_cmap"]) / self.statistics["std_cmap"]
        
        return (
            positions,
            hand_params,
            object_params,
            cmap_params,
            self.template_hand_params,
            hand_knn,
            obj_knn,
            data_name,
            self.statistics
        )


class GraphDROTestDataset(Dataset):
    """
    Graph-aware DRO test dataset with KNN indices.
    """
    def __init__(
        self,
        data_path: str,
        template_mash_path: str,
        statistics: Dict[str, torch.Tensor],
        knn_cache_path: str,
        k_neighbors: int = K_NEIGHBORS,
        expand_num: int = 100,
        precompute_device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()
        self.data_path = data_path
        self.knn_cache_path = knn_cache_path
        self.k_neighbors = k_neighbors
        self.expand_num = expand_num
        
        with h5py.File(self.data_path, 'r') as hf:
            self.length = len(hf.keys())
        
        self.statistics: Dict[str, torch.Tensor] = statistics
        for key, value in self.statistics.items():
            assert type(value) is torch.Tensor
        
        # Load template (raw)
        template_mash: Dict[str, Any] = np.load(template_mash_path, allow_pickle=True)[()]
        hand_mask_params_raw = torch.from_numpy(template_mash["hand_params"]["mask_params"]).float()
        hand_sh_params_raw = torch.from_numpy(template_mash["hand_params"]["sh_params"]).float()
        hand_ortho_poses_raw = torch.from_numpy(template_mash["hand_params"]["ortho_poses"]).float()
        
        # Raw template for KNN computation
        template_hand_params_raw = torch.cat([hand_mask_params_raw, hand_sh_params_raw, hand_ortho_poses_raw], dim=1)
        
        # Normalized template for inference return
        hand_mask_params_norm = (hand_mask_params_raw - statistics["mean_hand_mask"]) / statistics["std_hand_mask"]
        hand_sh_params_norm = (hand_sh_params_raw - statistics["mean_hand_sh"]) / statistics["std_hand_sh"]
        hand_ortho_poses_norm = (hand_ortho_poses_raw - statistics["mean_hand_ortho"]) / statistics["std_hand_ortho"]
        self.template_hand_params = torch.cat([hand_mask_params_norm, hand_sh_params_norm, hand_ortho_poses_norm], dim=1)
        
        # Precompute KNN cache if not exists
        # Use raw template hand params to compute hand KNN (consistent with inference)
        if not os.path.exists(knn_cache_path):
            precompute_knn_cache(
                data_path=data_path,
                cache_path=knn_cache_path,
                statistics=statistics,
                template_hand_params_raw=template_hand_params_raw,
                k=k_neighbors,
                device=precompute_device,
            )
        
        self._data_hf = None
        self._knn_hf = None
    
    def __getstate__(self):
        state = self.__dict__.copy()
        state['_data_hf'] = None
        state['_knn_hf'] = None
        return state
    
    def __del__(self):
        try:
            if getattr(self, "_data_hf", None) is not None:
                self._data_hf.close()
            if getattr(self, "_knn_hf", None) is not None:
                self._knn_hf.close()
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
        torch.Tensor,   # [N, K] hand_knn (computed from template)
        torch.Tensor,   # [N, K] obj_knn
        str,            # data_name
        Dict[str, torch.Tensor]  # statistics
    ]:
        data_count = idx // self.length
        idx = idx % self.length
        
        if self._data_hf is None:
            self._data_hf = h5py.File(self.data_path, 'r', swmr=True)
        if self._knn_hf is None:
            self._knn_hf = h5py.File(self.knn_cache_path, 'r', swmr=True)
        
        grp = self._data_hf[str(idx)]
        positions_np = grp["positions"][:]
        object_np = grp["object_params"][:]
        data_name = Path(grp.attrs["name"]).stem + f"_{data_count}"
        
        # Load KNN indices (both hand and object KNN for test)
        # hand_knn is computed from template, so it's available during inference
        knn_grp = self._knn_hf[str(idx)]
        hand_knn = torch.from_numpy(knn_grp["hand_knn"][:]).long()
        obj_knn = torch.from_numpy(knn_grp["obj_knn"][:]).long()
        
        positions = (torch.from_numpy(positions_np).float() - self.statistics["mean_positions"]) / self.statistics["std_positions"]
        
        object_mask = (torch.from_numpy(object_np[:, :MASK_DIM]).float() - self.statistics["mean_obj_mask"]) / self.statistics["std_obj_mask"]
        object_sh = (torch.from_numpy(object_np[:, MASK_DIM:MASK_DIM+SH_DIM]).float() - self.statistics["mean_obj_sh"]) / self.statistics["std_obj_sh"]
        object_ortho = (torch.from_numpy(object_np[:, MASK_DIM+SH_DIM:]).float() - self.statistics["mean_obj_ortho"]) / self.statistics["std_obj_ortho"]
        object_params = torch.cat([object_mask, object_sh, object_ortho], dim=1)
        
        return positions, object_params, self.template_hand_params, hand_knn, obj_knn, data_name, self.statistics


class GraphDROTrainDataModule(pl.LightningDataModule):
    """
    DataModule for graph-aware DRO training with multi-hand support.
    """
    def __init__(
        self,
        data_root: str,
        template_mash_root: str,
        batch_size: int,
        num_workers: int,
        knn_cache_dir: Optional[str] = None,
        k_neighbors: int = K_NEIGHBORS,
        precompute_device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        super().__init__()
        self.data_root = data_root
        self.template_mash_root = template_mash_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.k_neighbors = k_neighbors
        self.precompute_device = precompute_device
        
        # KNN cache directory
        if knn_cache_dir is None:
            self.knn_cache_dir = os.path.join(data_root, ".cache", "knn_cache")
        else:
            self.knn_cache_dir = knn_cache_dir
        os.makedirs(self.knn_cache_dir, exist_ok=True)
        
        # Data paths
        self.shadow_data_path = os.path.join(data_root, "shadow_train_data.h5")
        self.allegro_data_path = os.path.join(data_root, "allegro_train_data.h5")
        self.barrett_data_path = os.path.join(data_root, "barrett_train_data.h5")
        
        self.shadow_template_path = os.path.join(template_mash_root, "mash_sphere_shadow_template.npy")
        self.allegro_template_path = os.path.join(template_mash_root, "mash_sphere_allegro_template.npy")
        self.barrett_template_path = os.path.join(template_mash_root, "mash_sphere_barrett_template.npy")
        
        # KNN cache paths
        self.shadow_knn_cache = os.path.join(self.knn_cache_dir, "shadow_knn_cache.h5")
        self.allegro_knn_cache = os.path.join(self.knn_cache_dir, "allegro_knn_cache.h5")
        self.barrett_knn_cache = os.path.join(self.knn_cache_dir, "barrett_knn_cache.h5")
        
        # Statistics paths
        self.shadow_stat_path = os.path.join(data_root, "shadow_statistics.npy")
        self.allegro_stat_path = os.path.join(data_root, "allegro_statistics.npy")
        self.barrett_stat_path = os.path.join(data_root, "barrett_statistics.npy")
        
        # Load statistics
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

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            # Training datasets
            self.shadow_train_dataset = GraphDROTrainDataset(
                data_path=self.shadow_data_path,
                template_mash_path=self.shadow_template_path,
                statistics=self.shadow_statistics,
                knn_cache_path=self.shadow_knn_cache,
                k_neighbors=self.k_neighbors,
                precompute_device=self.precompute_device,
            )
            self.allegro_train_dataset = GraphDROTrainDataset(
                data_path=self.allegro_data_path,
                template_mash_path=self.allegro_template_path,
                statistics=self.allegro_statistics,
                knn_cache_path=self.allegro_knn_cache,
                k_neighbors=self.k_neighbors,
                precompute_device=self.precompute_device,
            )
            self.barrett_train_dataset = GraphDROTrainDataset(
                data_path=self.barrett_data_path,
                template_mash_path=self.barrett_template_path,
                statistics=self.barrett_statistics,
                knn_cache_path=self.barrett_knn_cache,
                k_neighbors=self.k_neighbors,
                precompute_device=self.precompute_device,
            )
            self.train_dataset = ConcatDataset([
                self.shadow_train_dataset,
                self.allegro_train_dataset,
                self.barrett_train_dataset,
            ])
            
            # Validation datasets (same as train)
            self.val_dataset = ConcatDataset([
                GraphDROTrainDataset(
                    data_path=self.shadow_data_path,
                    template_mash_path=self.shadow_template_path,
                    statistics=self.shadow_statistics,
                    knn_cache_path=self.shadow_knn_cache,
                    k_neighbors=self.k_neighbors,
                ),
                GraphDROTrainDataset(
                    data_path=self.allegro_data_path,
                    template_mash_path=self.allegro_template_path,
                    statistics=self.allegro_statistics,
                    knn_cache_path=self.allegro_knn_cache,
                    k_neighbors=self.k_neighbors,
                ),
                GraphDROTrainDataset(
                    data_path=self.barrett_data_path,
                    template_mash_path=self.barrett_template_path,
                    statistics=self.barrett_statistics,
                    knn_cache_path=self.barrett_knn_cache,
                    k_neighbors=self.k_neighbors,
                ),
            ])

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


class GraphDROTestDataModule(pl.LightningDataModule):
    """
    DataModule for graph-aware DRO testing.
    """
    def __init__(
        self,
        data_root: str,
        template_mash_root: str,
        batch_size: int,
        num_workers: int,
        hand_type: str,
        knn_cache_dir: Optional[str] = None,
        k_neighbors: int = K_NEIGHBORS,
        precompute_device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        super().__init__()
        self.hand_type = hand_type
        assert hand_type in ["shadow", "allegro", "barrett"], f"ERROR: unknown hand type {hand_type}!"
        
        self.data_path = os.path.join(data_root, "test_data.h5")
        self.stat_path = os.path.join(data_root, f"{hand_type}_statistics.npy")
        self.template_path = os.path.join(template_mash_root, f"mash_sphere_{hand_type}_template.npy")
        
        if knn_cache_dir is None:
            knn_cache_dir = os.path.join(data_root, ".cache", "knn_cache")
        self.knn_cache_path = os.path.join(knn_cache_dir, f"{hand_type}_test_knn_cache.h5")
        
        self.statistics = to_multi_hand_format(np.load(self.stat_path, allow_pickle=True)[()])
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.k_neighbors = k_neighbors
        self.precompute_device = precompute_device

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "test" or stage is None:
            self.test_dataset = GraphDROTestDataset(
                data_path=self.data_path,
                template_mash_path=self.template_path,
                statistics=self.statistics,
                knn_cache_path=self.knn_cache_path,
                k_neighbors=self.k_neighbors,
                precompute_device=self.precompute_device,
            )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

