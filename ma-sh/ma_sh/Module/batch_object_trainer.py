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

from typing import Dict, Any, List

import sys

sys.path.append("../ma-sh")
sys.path.append("../chamfer-distance")

from ma_sh.Config.constant import EPSILON
from ma_sh.Method.data import toNumpy
from ma_sh.Model.batch_object_mash import BatchObjectMash
from ma_sh.Module.data_prepare import DGNTestDataset
from ma_sh.Module.early_stopping import EarlyStopping
from chamfer_distance.Module.chamfer_distances import ChamferDistances

class BatchObjectTrainer:
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

        self.batch_object_mash = BatchObjectMash(
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
                self.batch_object_mash.object_params["mask_params"],
                self.batch_object_mash.object_params["sh_params"],
                self.batch_object_mash.object_params["ortho_poses"]
            ], 
            lr=self.lr
        )

        self.batch_object_gt_points: torch.Tensor | None = None
        self.error = 0
        self.epoch = 0
        self.step = 0
        
        return

    def loadMeshFile(
        self,
        batch_object_points: torch.Tensor,  # [B, P, N, 3]
    ) -> bool:
        assert batch_object_points.shape[0] == self.batch_size
        self.batch_object_gt_points = batch_object_points   # [B, P, N, 3]
        
        if self.initial_method == "middle_plane":
            self.batch_object_mash.middlePlaneInit()
        elif self.initial_method == "middle_plane_fixed":
            self.batch_object_mash.middlePlaneFixInit()
        elif self.initial_method == "sphere_fixed":
            self.batch_object_mash.sphereFixInit()
        else:
            raise ValueError(f"Unknown initial method: {self.initial_method}")
        
        self.matchObjectGT()

        self.batch_object_gt_points = self.batch_object_gt_points.to(self.batch_object_mash.device)
        self.batch_object_mash.shParamInit(self.batch_object_gt_points)

        return True

    def matchObjectGT(self) -> bool:
        batch_object_gt_np = self.batch_object_gt_points.numpy()        # [B, P, N, 3]
        batch_object_patch_centers = batch_object_gt_np.mean(axis=2)    # [B, P, 3]
        positions_np = self.batch_object_mash.positions.cpu().numpy()      # [B, P, 3]
        for i in range(self.batch_size):
            object_cost_matrix = cdist(positions_np[i], batch_object_patch_centers[i], metric="euclidean")
            object_row_ind, object_col_ind = linear_sum_assignment(object_cost_matrix)

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

    def trainStep(
        self,
        fit_loss_weight: float,
        coverage_loss_weight: float,
    ) -> Dict[str, Any] | None:
        self.optimizer.zero_grad()

        batch_object_mash_pts = self.batch_object_mash.toSamplePoints().reshape(self.batch_size, self.submesh_num, self.points_per_submesh, 3)   # [B, P, N, 3]

        if torch.cuda.is_available()  and self.batch_object_gt_points.device != "cpu":
            flat_object_fit_dist2, flat_object_coverage_dist2 = ChamferDistances.namedAlgo("cuda")(
                batch_object_mash_pts.reshape(-1, self.points_per_submesh, 3),
                self.batch_object_gt_points.reshape(-1, self.points_per_submesh, 3),
            )[:2]
        else:
            flat_object_fit_dist2, flat_object_coverage_dist2 = ChamferDistances.namedAlgo("default")(
                batch_object_mash_pts.reshape(-1, self.points_per_submesh, 3),
                self.batch_object_gt_points.reshape(-1, self.points_per_submesh, 3),
            )[:2]
        
        flat_object_dists = torch.sqrt(flat_object_fit_dist2 + EPSILON)
        flat_object_coverage_dists = torch.sqrt(flat_object_coverage_dist2 + EPSILON)

        object_fit_loss = torch.mean(flat_object_dists)
        object_coverage_loss = torch.mean(flat_object_coverage_dists)

        object_weighed_fit_loss = fit_loss_weight * object_fit_loss
        object_weighed_coverage_loss = coverage_loss_weight * object_coverage_loss

        loss = object_weighed_fit_loss + object_weighed_coverage_loss

        if torch.isnan(loss).any():
            print("loss is nan!")
            return None
        
        loss.backward()
        self.optimizer.step()

        object_cd = toNumpy(object_fit_loss) + toNumpy(object_coverage_loss)
        self.error = object_cd

        loss_dict = {
            "Train/epoch": self.epoch,
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

            if self.batch_object_mash.sh_degree_max == 2 and loss < threshold:
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

    def escalateTrainMash(self) -> float:
        lr_decay_rate = 0.2

        self.batch_object_mash.setGradState(True)
        self.warmUpEpoch(1.0, 0.5, self.warmup_step_num)
        self.trainEpoch(1.0, 0.5)

        self.batch_object_mash.addShDegreeMax()
        self.lr *= lr_decay_rate
        self.optimizer = Adam(
            [
                self.batch_object_mash.object_params["mask_params"],
                self.batch_object_mash.object_params["sh_params"],
                self.batch_object_mash.object_params["ortho_poses"],
            ],
            lr=self.lr
        )

        self.batch_object_mash.setGradState(True)
        self.warmUpEpoch(1.0, 0.75, self.warmup_step_num)
        self.trainEpoch(1.0, 0.75)

        self.batch_object_mash.addShDegreeMax()
        self.lr *= lr_decay_rate
        self.optimizer = Adam(
            [
                self.batch_object_mash.object_params["mask_params"],
                self.batch_object_mash.object_params["sh_params"],
                self.batch_object_mash.object_params["ortho_poses"],
            ],
            lr=self.lr
        )

        self.batch_object_mash.setGradState(True)
        self.warmUpEpoch(1.0, 1.0, self.warmup_step_num)
        final_loss = self.trainEpoch(1.0, 1.0)

        self.closeLog()

        return final_loss

    def saveObjectMash(self, save_root: str, data_names: List[str]) -> bool:
        os.makedirs(save_root, exist_ok=True)
        for i in range(self.batch_size):
            params_dict = {
                "positions": toNumpy(self.batch_object_mash.positions[i]),
                "object_params": {
                    "mask_params": toNumpy(self.batch_object_mash.object_params["mask_params"][i]),
                    "sh_params": toNumpy(self.batch_object_mash.object_params["sh_params"][i]),
                    "ortho_poses": toNumpy(self.batch_object_mash.object_params["ortho_poses"][i]),
                }
            }
            np.save(os.path.join(save_root, data_names[i] + "_object.npy"), params_dict, allow_pickle=True)
        
        return True


if __name__ == "__main__":
    batch_size = 128
    dataset = DGNTestDataset(
        data_root="/mnt/d/yxz/DATASET/Original-Meshes/DexGraspNet/test",
        object_template_root="/mnt/d/yxz/DATASET/Original-Meshes/DexGraspNet/object_template",
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8
    )
    for object_points, data_name in dataloader:
        start_time = time.time()
        batch_object_trainer = BatchObjectTrainer(
            anchor_num=256,
            mask_degree_max=3,
            sh_degree_max=0,
            sample_phi_num=32,
            sample_theta_num=32,
            points_per_submesh=1024,
            batch_size=len(data_name),
            dtype=torch.float32,
            device="cuda",
            lr=0.01 * math.sqrt(len(data_name)),
            warmup_step_num=50,
            patience=2,
            save_log_folder_path=None,
            initial_method="middle_plane_fixed",
        )
        batch_object_trainer.loadMeshFile(object_points)
        loss = batch_object_trainer.escalateTrainMash()
        batch_object_trainer.saveObjectMash("./batch-test", data_name)
        
        end_time = time.time()

        print(f"Loss: {loss:.6f}")
        print(f"Total time cost: {end_time - start_time:.6f}")
        print(f"Average time cost: {(end_time - start_time) / batch_size:.6f}")

        break
    


