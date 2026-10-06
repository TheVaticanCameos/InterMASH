import os
import math
import torch
import numpy as np
import time
from torch.optim import Adam
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
from pathlib import Path

from typing import Dict, Any, List

import sys
sys.path.append("../chamfer-distance")

from ma_sh.Config.constant import EPSILON
from ma_sh.Method.data import toNumpy
from ma_sh.Model.batch_hoi_mash import BatchHOIMash
from ma_sh.Module.data_prepare import DRODataset, DRODataset4OnlyCMap
from ma_sh.Module.early_stopping import EarlyStopping
from chamfer_distance.Module.chamfer_distances import ChamferDistances


class BatchHOITrainer:
    def __init__(
        self,
        anchor_num: int = 256,          # P
        mask_degree_max: int = 3,
        sh_degree_max: int = 2,
        sample_phi_num: int = 32,
        sample_theta_num: int = 32,
        points_per_submesh: int = 1024, # N
        batch_size: int = 1,            # B
        dtype: torch.dtype = torch.float32,
        device: str = "cuda",
        lr: float = 2e-3,
        warmup_step_num: int = 80,
        patience: int = 2,
        save_log_folder_path: str | None = None,
        initial_method: str = "middle_plane",
    ) -> None:
        self.lr = lr
        self.warmup_step_num = warmup_step_num
        self.patience = patience
        self.save_log_folder_path = save_log_folder_path
        self.submesh_num = anchor_num
        self.points_per_submesh = points_per_submesh
        self.batch_size = batch_size
        self.initial_method = initial_method

        self.writer = SummaryWriter(log_dir=os.path.join(save_log_folder_path, "tensorboard"), flush_secs=1) if save_log_folder_path else None

        self.batch_hoi_mash = BatchHOIMash(
            anchor_num=anchor_num,
            mask_degree_max=mask_degree_max,
            sh_degree_max=sh_degree_max,
            sample_phi_num=sample_phi_num,
            sample_theta_num=sample_theta_num,
            dtype=dtype,
            device=device,
            batch_size=batch_size,
        )

        self.optimizer = Adam(
            [
                self.batch_hoi_mash.hand_params["mask_params"],
                self.batch_hoi_mash.hand_params["sh_params"],
                self.batch_hoi_mash.hand_params["ortho_poses"],
                self.batch_hoi_mash.object_params["mask_params"],
                self.batch_hoi_mash.object_params["sh_params"],
                self.batch_hoi_mash.object_params["ortho_poses"],
            ],
            lr=self.lr
        )

        self.batch_hand_gt_points: torch.Tensor | None = None
        self.batch_object_gt_points: torch.Tensor | None = None
        self.batch_hand_cmap: torch.Tensor | None = None
        self.batch_object_cmap: torch.Tensor | None = None
        self.error = 0
        self.hand_error = 0
        self.object_error = 0
        self.epoch = 0
        self.step = 0
        
        return
    
    @classmethod
    def loadFromFile(
        cls,
        mash_path_list: List[str],
        cmap_batch: torch.Tensor,
        anchor_num: int = 256,          # P
        mask_degree_max: int = 3,
        sh_degree_max: int = 2,
        sample_phi_num: int = 32,
        sample_theta_num: int = 32,
        points_per_submesh: int = 1024, # N
        batch_size: int = 1,            # B
        dtype: torch.dtype = torch.float32,
        device: str = "cuda",
        lr: float = 2e-3,
        warmup_step_num: int = 80,
        patience: int = 2,
        save_log_folder_path: str | None = None,
        initial_method: str = "middle_plane",
    ) -> "BatchHOITrainer":
        resolved_batch_size = len(mash_path_list)
        trainer = cls(
            anchor_num=anchor_num,
            mask_degree_max=mask_degree_max,
            sh_degree_max=sh_degree_max,
            sample_phi_num=sample_phi_num,
            sample_theta_num=sample_theta_num,
            points_per_submesh=points_per_submesh,
            batch_size=resolved_batch_size,
            dtype=dtype,
            device=device,
            lr=lr,
            warmup_step_num=warmup_step_num,
            patience=patience,
            save_log_folder_path=save_log_folder_path,
            initial_method=initial_method,
        )

        batch_hoi_mash = BatchHOIMash(
            anchor_num=anchor_num,
            mask_degree_max=mask_degree_max,
            sh_degree_max=sh_degree_max,
            sample_phi_num=sample_phi_num,
            sample_theta_num=sample_theta_num,
            dtype=dtype,
            device=device,
            batch_size=resolved_batch_size,
        )

        for i, mash_path in enumerate(mash_path_list):
            params_dict = np.load(mash_path, allow_pickle=True).item()
            batch_hoi_mash.positions[i] = torch.from_numpy(params_dict["positions"]).to(device=device, dtype=dtype)
            for key in batch_hoi_mash.hand_params.keys():
                if key in params_dict["hand_params"].keys():
                    batch_hoi_mash.hand_params[key][i] = torch.from_numpy(params_dict["hand_params"][key]).to(device=device, dtype=dtype)
            for key in batch_hoi_mash.object_params.keys():
                if key in params_dict["object_params"].keys():
                    batch_hoi_mash.object_params[key][i] = torch.from_numpy(params_dict["object_params"][key]).to(device=device, dtype=dtype)
            
            batch_hoi_mash.object_params["cmap_params"][i] = torch.from_numpy(params_dict["cmap_params"]["sh_params"]).to(device=device, dtype=dtype)

        trainer.batch_hoi_mash = batch_hoi_mash
        trainer.optimizer = Adam(
            [
                trainer.batch_hoi_mash.hand_params["mask_params"],
                trainer.batch_hoi_mash.hand_params["sh_params"],
                trainer.batch_hoi_mash.hand_params["ortho_poses"],
                trainer.batch_hoi_mash.object_params["mask_params"],
                trainer.batch_hoi_mash.object_params["sh_params"],
                trainer.batch_hoi_mash.object_params["ortho_poses"],
            ],
            lr=trainer.lr,
        )

        trainer.batch_hand_cmap = cmap_batch.to(trainer.batch_hoi_mash.device, dtype=trainer.batch_hoi_mash.dtype)
        trainer.batch_object_cmap = None

        return trainer

    def loadMeshFile(
        self,
        batch_hand_points: torch.Tensor,    # [B, P, N, 3]
        batch_object_points: torch.Tensor,  # [B, P, N, 3]
        batch_hand_cmap: torch.Tensor,      # [B, M, 4]
        batch_object_cmap: torch.Tensor,    # [B, M, 4]
    ) -> bool:
        assert batch_hand_points.shape[0] == batch_object_points.shape[0] == batch_hand_cmap.shape[0] == batch_object_cmap.shape[0] == self.batch_size

        self.batch_hand_gt_points = batch_hand_points       # [B, P, N, 3]
        self.batch_object_gt_points = batch_object_points   # [B, P, N, 3]
        self.batch_hand_cmap = batch_hand_cmap              # [B, M, 4]
        self.batch_object_cmap = batch_object_cmap          # [B, M, 4]
        
        if self.initial_method == "middle_plane":
            self.batch_hoi_mash.middlePlaneInit()
            self.matchBothGT()
        elif self.initial_method == "middle_plane_fixed":
            self.batch_hoi_mash.middlePlaneFixInit()
            self.matchObjectGT()
        elif self.initial_method == "sphere_fixed":
            self.batch_hoi_mash.sphereFixInit()
            self.matchObjectGT()
        else:
            raise ValueError(f"Unknown initial method: {self.initial_method}")

        self.batch_hand_gt_points = self.batch_hand_gt_points.to(self.batch_hoi_mash.device)
        self.batch_object_gt_points = self.batch_object_gt_points.to(self.batch_hoi_mash.device)
        self.batch_hand_cmap = self.batch_hand_cmap.to(self.batch_hoi_mash.device)
        self.batch_object_cmap = self.batch_object_cmap.to(self.batch_hoi_mash.device)
        self.batch_hoi_mash.shParamInit(self.batch_hand_gt_points, self.batch_object_gt_points)

        return True

    def matchBothGT(self) -> bool:
        batch_hand_gt_np = self.batch_hand_gt_points.numpy()            # [B, P, N, 3]
        batch_hand_patch_centers = batch_hand_gt_np.mean(axis=2)        # [B, P, 3]
        batch_object_gt_np = self.batch_object_gt_points.numpy()        # [B, P, N, 3]
        batch_object_patch_centers = batch_object_gt_np.mean(axis=2)    # [B, P, 3]
        positions_np = self.batch_hoi_mash.positions.cpu().numpy()      # [B, P, 3]
        for i in range(self.batch_size):
            hand_cost_matrix = cdist(positions_np[i], batch_hand_patch_centers[i], metric="euclidean")
            hand_row_ind, hand_col_ind = linear_sum_assignment(hand_cost_matrix)
            assert (hand_row_ind == np.arange(self.submesh_num)).all()
            self.batch_hand_gt_points[i] = self.batch_hand_gt_points[i][hand_col_ind]

            object_cost_matrix = cdist(positions_np[i], batch_object_patch_centers[i], metric="euclidean")
            object_row_ind, object_col_ind = linear_sum_assignment(object_cost_matrix)
            assert (object_row_ind == np.arange(self.submesh_num)).all()
            self.batch_object_gt_points[i] = self.batch_object_gt_points[i][object_col_ind]
        
        return True

    def matchObjectGT(self) -> bool:
        batch_object_gt_np = self.batch_object_gt_points.numpy()        # [B, P, N, 3]
        batch_object_patch_centers = batch_object_gt_np.mean(axis=2)    # [B, P, 3]
        positions_np = self.batch_hoi_mash.positions.cpu().numpy()      # [B, P, 3]
        for i in range(self.batch_size):
            object_cost_matrix = cdist(positions_np[i], batch_object_patch_centers[i], metric="euclidean")
            try:
                object_row_ind, object_col_ind = linear_sum_assignment(object_cost_matrix)
            except ValueError:
                print(f"Error in {i}")
                import pdb
                pdb.set_trace()
            assert (object_row_ind == np.arange(self.submesh_num)).all()
            self.batch_object_gt_points[i] = self.batch_object_gt_points[i][object_col_ind]
        
        return True

    def updateLr(self, lr: float) -> bool:
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return True
    
    def closeLog(self) -> None:
        if self.writer is not None:
            self.writer.close()
        return
    
    def upSampleCMapGT(self, target: str) -> torch.Tensor:   # [B, P*N]
        batch_up_sampled_pc = self.batch_hoi_mash.toSamplePoints(target).reshape(self.batch_size, -1, 3)  # [B, P*N, 3]

        gt_dtype = self.batch_hoi_mash.dtype

        with torch.no_grad():
            # batch_cmap: [B, M, 4] -> pc=[..., :3], val=[..., 3]
            if target == "hand":
                pc_sparse = self.batch_hand_cmap[..., :3]
                cmap_sparse = self.batch_hand_cmap[..., 3]
            elif target == "object":
                pc_sparse = self.batch_object_cmap[..., :3]
                cmap_sparse = self.batch_object_cmap[..., 3]
            else:
                raise ValueError(f"Unknown target: {target}")

            if isinstance(pc_sparse, torch.Tensor):
                p = pc_sparse.detach().to(self.batch_hoi_mash.device, dtype=gt_dtype)  # [B, M, 3]
            else:
                p = torch.from_numpy(np.asarray(pc_sparse)).to(self.batch_hoi_mash.device, dtype=gt_dtype)

            if isinstance(cmap_sparse, torch.Tensor):
                v = cmap_sparse.detach().to(self.batch_hoi_mash.device, dtype=gt_dtype)  # [B, M]
            else:
                v = torch.from_numpy(np.asarray(cmap_sparse)).to(self.batch_hoi_mash.device, dtype=gt_dtype)

            q = batch_up_sampled_pc.to(self.batch_hoi_mash.device, dtype=gt_dtype)  # [B, P*N, 3]

            n = p.shape[1]
            k = int(min(8, max(1, n)))
            eps = 1e-8

            pred = torch.zeros(self.batch_size, q.shape[1], device=self.batch_hoi_mash.device, dtype=gt_dtype)  # [B, P*N]
            chunk_size = 8192

            for b in range(self.batch_size):
                pb = p[b]         # [M, 3]
                vb = v[b]         # [M]
                for s in range(0, q.shape[1], chunk_size):
                    e = min(s + chunk_size, q.shape[1])
                    qi = q[b, s:e]  # [L, 3]
                    di = torch.cdist(qi, pb)  # [L, M]
                    k_use = min(k, n)
                    dk, idxk = torch.topk(di, k_use, dim=1, largest=False)
                    vals = vb[idxk]  # [L, k]
                    w = 1.0 / ((dk + eps) ** 2)
                    denom = w.sum(dim=1)
                    num = (w * vals).sum(dim=1)
                    predi = num / torch.maximum(denom, torch.tensor(eps, device=self.batch_hoi_mash.device, dtype=gt_dtype))
                    has_zero = (dk <= eps).any(dim=1)
                    if has_zero.any():
                        zero_first = (dk <= eps).float().argmax(dim=1)
                        rows = torch.nonzero(has_zero, as_tuple=True)[0]
                        predi[rows] = vals[rows, zero_first[rows]]
                    pred[b, s:e] = predi

            cmap_gt = pred.detach()  # [B, P*N]
            return cmap_gt
    
    def trainStep(
        self,
        fit_loss_weight: float,
        coverage_loss_weight: float,
    ) -> Dict[str, Any] | None:
        self.optimizer.zero_grad()

        batch_hand_mash_pts = self.batch_hoi_mash.toSamplePoints("hand").reshape(self.batch_size, self.submesh_num, self.points_per_submesh, 3)   # [B, P, N, 3]
        batch_object_mash_pts = self.batch_hoi_mash.toSamplePoints("object").reshape(self.batch_size, self.submesh_num, self.points_per_submesh, 3)   # [B, P, N, 3]

        if torch.cuda.is_available() and self.batch_hand_gt_points.device != "cpu" and self.batch_object_gt_points.device != "cpu":
            flat_hand_fit_dist2, flat_hand_coverage_dist2 = ChamferDistances.namedAlgo("cuda")(
                batch_hand_mash_pts.reshape(-1, self.points_per_submesh, 3),
                self.batch_hand_gt_points.reshape(-1, self.points_per_submesh, 3),
            )[:2]
            flat_object_fit_dist2, flat_object_coverage_dist2 = ChamferDistances.namedAlgo("cuda")(
                batch_object_mash_pts.reshape(-1, self.points_per_submesh, 3),
                self.batch_object_gt_points.reshape(-1, self.points_per_submesh, 3),
            )[:2]
        else:
            flat_hand_fit_dist2, flat_hand_coverage_dist2 = ChamferDistances.namedAlgo("default")(
                batch_hand_mash_pts.reshape(-1, self.points_per_submesh, 3),
                self.batch_hand_gt_points.reshape(-1, self.points_per_submesh, 3),
            )[:2]
            flat_object_fit_dist2, flat_object_coverage_dist2 = ChamferDistances.namedAlgo("default")(
                batch_object_mash_pts.reshape(-1, self.points_per_submesh, 3),
                self.batch_object_gt_points.reshape(-1, self.points_per_submesh, 3),
            )[:2]
        
        flat_hand_dists = torch.sqrt(flat_hand_fit_dist2 + EPSILON)
        flat_hand_coverage_dists = torch.sqrt(flat_hand_coverage_dist2 + EPSILON)
        flat_object_dists = torch.sqrt(flat_object_fit_dist2 + EPSILON)
        flat_object_coverage_dists = torch.sqrt(flat_object_coverage_dist2 + EPSILON)

        hand_fit_loss = torch.mean(flat_hand_dists)
        hand_coverage_loss = torch.mean(flat_hand_coverage_dists)
        object_fit_loss = torch.mean(flat_object_dists)
        object_coverage_loss = torch.mean(flat_object_coverage_dists)

        hand_weighed_fit_loss = fit_loss_weight * hand_fit_loss
        hand_weighed_coverage_loss = coverage_loss_weight * hand_coverage_loss
        object_weighed_fit_loss = fit_loss_weight * object_fit_loss
        object_weighed_coverage_loss = coverage_loss_weight * object_coverage_loss

        loss = hand_weighed_fit_loss + hand_weighed_coverage_loss + object_weighed_fit_loss + object_weighed_coverage_loss

        if torch.isnan(loss).any():
            print("loss is nan!")
            return None
        
        loss.backward()
        self.optimizer.step()

        hand_cd = toNumpy(hand_fit_loss) + toNumpy(hand_coverage_loss)
        object_cd = toNumpy(object_fit_loss) + toNumpy(object_coverage_loss)
        self.error = hand_cd + object_cd
        self.hand_error = hand_cd
        self.object_error = object_cd

        loss_dict = {
            "Train/epoch": self.epoch,
            "Train/hand_cd": hand_cd,
            "Train/object_cd": object_cd,
            "Train/error": self.error,
        }
        
        if self.writer is not None:
            for name, value in loss_dict.items():
                if name != "Train/epoch":
                    self.writer.add_scalar(name, value, self.step)
            
            self.writer.add_scalar("Train/lr", self.optimizer.param_groups[0]["lr"], self.step)
        
        return loss_dict
    
    def warmUpEpoch(
        self,
        fit_loss_weight: float, 
        coverage_loss_weight: float,
        train_step_max: int,
    ) -> bool:
        pbar = tqdm(total=train_step_max)
        start_step = self.step
        for i in range(train_step_max):
            current_lr = self.lr * (i + 1.0) / train_step_max
            self.updateLr(current_lr)
            loss_dict = self.trainStep(fit_loss_weight, coverage_loss_weight)
            loss = loss_dict["Train/error"]
            pbar.set_description(f"LOSS: {loss:.6f}")
            self.step += 1
            pbar.update(1)

            if self.step >= start_step + train_step_max:
                break
        
        self.epoch += 1
        return True
    
    def trainEpoch(
        self,
        fit_loss_weight: float,
        coverage_loss_weight: float,
        threshold: float = 0.003,
    ) -> float:
        early_stopping = EarlyStopping(self.patience, 1e-6)
        pbar = tqdm()
        min_loss = float("inf")
        min_loss_reached_time = 0
        while True:
            loss_dict = self.trainStep(fit_loss_weight, coverage_loss_weight)
            loss = loss_dict["Train/error"]
            self.step += 1
            pbar.set_description(f"LOSS: {loss:.6f}")
            pbar.update(1)

            if self.batch_hoi_mash.sh_degree_max == 2 and loss < threshold:
                break

            if loss < min_loss:
                min_loss = loss
                min_loss_reached_time = 0
            else:
                min_loss_reached_time += 1
            
            if min_loss_reached_time >= self.patience:
                break
            
            if early_stopping(loss.item()):
                break
        
        self.epoch += 1
        return loss
    
    def trainStepCMap(
        self, 
        hand_cmap_gt: torch.Tensor | None,
        object_cmap_gt: torch.Tensor | None,
        scheduler = None
    ) -> Dict[str, Any] | None:
        self.optimizer.zero_grad(set_to_none=True)
        self.step += 1
        if (hand_cmap_gt is None) == (object_cmap_gt is None):
            raise ValueError("Exactly one of hand_cmap_gt/object_cmap_gt must be provided")
        optimize_target = "hand" if hand_cmap_gt is not None else "object"

        loss = None
        hand_loss = None
        object_loss = None

        if optimize_target == "hand":
            if hand_cmap_gt is None:
                raise ValueError("hand_cmap_gt is None while optimizing hand cmap")
            hand_cmap_mash = self.batch_hoi_mash.toCMap("hand").reshape(self.batch_size, -1)  # [B, P*S*S]
            hand_loss = torch.nn.functional.mse_loss(hand_cmap_mash, hand_cmap_gt)  # [B, P*S*S]
            loss = hand_loss
        else:
            if object_cmap_gt is None:
                raise ValueError("object_cmap_gt is None while optimizing object cmap")
            object_cmap_mash = self.batch_hoi_mash.toCMap("object").reshape(self.batch_size, -1)  # [B, P*S*S]
            object_loss = torch.nn.functional.mse_loss(object_cmap_mash, object_cmap_gt)  # [B, P*S*S]
            loss = object_loss

        if torch.isnan(loss).any():
            print("[ERROR][BatchHOITrainer::trainStepCMap]")
            print("\t loss is nan!")
            return None

        loss.backward()

        self.optimizer.step()
        if scheduler is not None:
            scheduler.step()
        
        loss_dict = {
            "Epoch": self.epoch,
            "CMap/cmap_loss": toNumpy(loss),
        }
        if hand_loss is not None:
            loss_dict["CMap/hand_loss"] = toNumpy(hand_loss)
        if object_loss is not None:
            loss_dict["CMap/object_loss"] = toNumpy(object_loss)

        if self.writer is not None:
            for name, value in loss_dict.items():
                if name != "Epoch":
                    self.writer.add_scalar(name, value, self.step)
            self.writer.add_scalar("LR/lr", self.optimizer.param_groups[0]["lr"], self.step)

        return loss_dict
    
    def trainCMap(
        self,
        hand_cmap_gt: torch.Tensor | None,
        object_cmap_gt: torch.Tensor | None,
        epoch_num: int,
        lr: float
    ) -> float:
        def train_single_cmap(optimize_target: str) -> float:
            self.batch_hoi_mash.setGradState(False)
            if optimize_target == "hand":
                self.batch_hoi_mash.hand_params["cmap_params"].requires_grad_(True)
                self.optimizer = Adam([self.batch_hoi_mash.hand_params["cmap_params"]], lr=lr)
                step_hand_cmap_gt = hand_cmap_gt
                step_object_cmap_gt = None
            else:
                self.batch_hoi_mash.object_params["cmap_params"].requires_grad_(True)
                self.optimizer = Adam([self.batch_hoi_mash.object_params["cmap_params"]], lr=lr)
                step_hand_cmap_gt = None
                step_object_cmap_gt = object_cmap_gt

            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=lr,
                epochs=epoch_num,
                steps_per_epoch=1,
                pct_start=0.1,
            )
            with tqdm(total=epoch_num, desc=f"CMap Training ({optimize_target.capitalize()})") as pbar:
                for _ in range(epoch_num):
                    loss_dict = self.trainStepCMap(step_hand_cmap_gt, step_object_cmap_gt, scheduler)
                    if loss_dict is None:
                        print("[ERROR][BatchHOITrainer::trainCMap]")
                        print("\t trainStepCMap failed!")
                        return False
                    postfix = {"cmap_loss": f"{loss_dict['CMap/cmap_loss']:.5f}"}
                    if "CMap/hand_loss" in loss_dict:
                        postfix["hand_loss"] = f"{loss_dict['CMap/hand_loss']:.5f}"
                    if "CMap/object_loss" in loss_dict:
                        postfix["object_loss"] = f"{loss_dict['CMap/object_loss']:.5f}"
                    pbar.set_postfix(postfix)
                    pbar.update(1)

            return loss_dict["CMap/cmap_loss"]

        if hand_cmap_gt is not None and object_cmap_gt is not None:
            hand_loss = train_single_cmap("hand")
            if hand_loss is False:
                return False
            object_loss = train_single_cmap("object")
            if object_loss is False:
                return False
            return (hand_loss + object_loss) / 2.0

        optimize_target = "hand" if hand_cmap_gt is not None else "object"
        return train_single_cmap(optimize_target)

    def escalateTrainMash(self) -> float:
        lr_decay_rate = 0.2

        self.batch_hoi_mash.setGradState(True)
        self.warmUpEpoch(1.0, 0.5, self.warmup_step_num)
        self.trainEpoch(1.0, 0.5)

        self.batch_hoi_mash.addShDegreeMax()
        self.lr *= lr_decay_rate
        self.optimizer = Adam(
            [
                self.batch_hoi_mash.hand_params["mask_params"],
                self.batch_hoi_mash.hand_params["sh_params"],
                self.batch_hoi_mash.hand_params["ortho_poses"],
                self.batch_hoi_mash.object_params["mask_params"],
                self.batch_hoi_mash.object_params["sh_params"],
                self.batch_hoi_mash.object_params["ortho_poses"],
            ],
            lr=self.lr
        )

        self.batch_hoi_mash.setGradState(True)
        self.warmUpEpoch(1.0, 0.75, self.warmup_step_num)
        self.trainEpoch(1.0, 0.75)

        self.batch_hoi_mash.addShDegreeMax()
        self.lr *= lr_decay_rate
        self.optimizer = Adam(
            [
                self.batch_hoi_mash.hand_params["mask_params"],
                self.batch_hoi_mash.hand_params["sh_params"],
                self.batch_hoi_mash.hand_params["ortho_poses"],
                self.batch_hoi_mash.object_params["mask_params"],
                self.batch_hoi_mash.object_params["sh_params"],
                self.batch_hoi_mash.object_params["ortho_poses"],
            ],
            lr=self.lr
        )

        self.batch_hoi_mash.setGradState(True)
        self.warmUpEpoch(1.0, 1.0, self.warmup_step_num)
        final_loss = self.trainEpoch(1.0, 1.0)

        hand_cmap_gt = self.upSampleCMapGT("hand") # [B, P*N]
        object_cmap_gt = self.upSampleCMapGT("object") # [B, P*N]
        
        cmap_loss = self.trainCMap(hand_cmap_gt, object_cmap_gt, epoch_num=1000, lr=self.lr / lr_decay_rate / lr_decay_rate)

        print(f"cmap_loss: {cmap_loss:.6f}")

        self.closeLog()

        return final_loss
    
    def saveHOIMash(self, save_root: str, data_names: List[str]) -> bool:
        os.makedirs(save_root, exist_ok=True)
        for i in range(self.batch_size):
            params_dict = {
                "positions": toNumpy(self.batch_hoi_mash.positions[i]),
                "hand_params": {
                    "mask_params": toNumpy(self.batch_hoi_mash.hand_params["mask_params"][i]),
                    "sh_params": toNumpy(self.batch_hoi_mash.hand_params["sh_params"][i]),
                    "ortho_poses": toNumpy(self.batch_hoi_mash.hand_params["ortho_poses"][i]),
                    "cmap_params": toNumpy(self.batch_hoi_mash.hand_params["cmap_params"][i]),
                },
                "object_params": {
                    "mask_params": toNumpy(self.batch_hoi_mash.object_params["mask_params"][i]),
                    "sh_params": toNumpy(self.batch_hoi_mash.object_params["sh_params"][i]),
                    "ortho_poses": toNumpy(self.batch_hoi_mash.object_params["ortho_poses"][i]),
                    "cmap_params": toNumpy(self.batch_hoi_mash.object_params["cmap_params"][i]),
                },
            }
            np.save(os.path.join(save_root, data_names[i] + "_hoi.npy"), params_dict, allow_pickle=True)
        
        return True

    def onlyTrainCMap(
        self,
        epoch_num: int = 1000,
        lr: float | None = None,
    ) -> float:
        if self.batch_hand_cmap is None:
            raise ValueError("batch_hand_cmap is None; cannot train hand contact map")
        hand_cmap_gt = self.upSampleCMapGT("hand") # [B, P*N]
        cmap_loss = self.trainCMap(hand_cmap_gt, None, epoch_num=epoch_num, lr=(self.lr if lr is None else lr))
        return cmap_loss


if __name__ == "__main__":
    batch_size = 512

    hand_type = "barrett"

    target_root = f"/mnt/d/yxz/DATASET/HOI-Dataset/HandCMap-Sphere-CMapDataset/{hand_type}/train"

    os.makedirs(target_root, exist_ok=True) 

    mash_root = f"/mnt/d/yxz/DATASET/HOI-Dataset/Sphere-CMapDataset/{hand_type}/train"
    cmap_root = f"/mnt/d/yxz/DATASET/Original-Meshes/Sphere-CMapDataset/{hand_type}/train"

    hand_cmap_dataset = DRODataset4OnlyCMap(
        mash_root=mash_root,
        cmap_root=cmap_root,
    )

    dataloader = DataLoader(
        hand_cmap_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
    )

    for mash_paths, hand_cmap in dataloader:
        mash_path_list = list(mash_paths)
        trainer = BatchHOITrainer.loadFromFile(
            mash_path_list=mash_path_list,
            cmap_batch=hand_cmap,
            anchor_num=256,
            mask_degree_max=3,
            sh_degree_max=2,
            sample_phi_num=32,
            sample_theta_num=32,
            points_per_submesh=1024,
            batch_size=len(mash_path_list),
            dtype=torch.float32,
            device="cuda" if torch.cuda.is_available() else "cpu",
            lr=0.01 * math.sqrt(len(mash_path_list)),
            warmup_step_num=80,
            patience=2,
            save_log_folder_path=None,
            initial_method="middle_plane",
        )
        cmap_loss = trainer.onlyTrainCMap(epoch_num=1000, lr=2e-3)
        print(f"cmap_loss: {cmap_loss:.6f}")

        data_names = [Path(path).stem.rsplit('_', 1)[0] for path in mash_paths]
        
        trainer.saveHOIMash(target_root, data_names)


# if __name__ == "__main__":
#     batch_size = 4
#     dataset = DRODataset(
#         data_root="/data2/yxz/DATASET/DRO-data/DRO-Original/shadow/train",
#         object_template_root="/data2/yxz/DATASET/DRO-data/DRO-Original/object_template",
#         hand_template_path="/data2/yxz/DATASET/template_info/sphere_shadow_template.npy",
#         use_permutation=True
#     )
#     dataloader = DataLoader(
#         dataset, 
#         batch_size=batch_size, 
#         shuffle=False,
#         num_workers=8,
#     )
#     for hand_points, object_points, cmap, data_name in dataloader:
#         start_time = time.time()
#         batch_hoi_trainer = BatchHOITrainer(
#             anchor_num=256,
#             mask_degree_max=3,
#             sh_degree_max=0,
#             sample_phi_num=32,
#             sample_theta_num=32,
#             points_per_submesh=1024,
#             batch_size=len(data_name),
#             dtype=torch.float32,
#             device="cuda",
#             lr=0.01 * math.sqrt(len(data_name)),
#             warmup_step_num=50,
#             patience=2,
#             save_log_folder_path=None,
#             initial_method="sphere_fixed",
#         )
#         batch_hoi_trainer.loadMeshFile(hand_points, object_points, cmap)
#         loss = batch_hoi_trainer.escalateTrainMash()
#         if loss > 0.003:
#             print(f"{data_name} loss: {loss:.6f}")
#         else:
#             batch_hoi_trainer.saveHOIMash("./batch-test", data_name)
#         end_time = time.time()
#         print(f"Total time cost: {end_time - start_time:.6f}")
#         print(f"Average time cost: {(end_time - start_time) / batch_size:.6f}")

#         break
