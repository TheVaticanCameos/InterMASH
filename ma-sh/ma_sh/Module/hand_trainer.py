import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from typing import Union, Dict, Any
from torch.optim import Adam
import trimesh
from tensorboardX import SummaryWriter
from time import time

sys.path.append("../ma-sh")
sys.path.append("../chamfer-distance")
from chamfer_distance.Module.chamfer_distances import ChamferDistances

from ma_sh.Config.constant import EPSILON
from ma_sh.Method.data import toNumpy
from ma_sh.Model.hand_mash import HandMash
from ma_sh.Module.early_stopping import EarlyStopping


class HandTrainer:
    def __init__(
        self,
        anchor_num: int = 256,              # P
        mask_degree_max: int = 3,
        sh_degree_max: int = 0,
        sample_phi_num: int = 32,           # S
        sample_theta_num: int = 32,         # S
        points_per_submesh: int = 1024,     # N
        dtype: torch.dtype = torch.float32,
        device: str = "cuda",
        lr: float = 2e-3,
        warmup_step_num: int = 50,
        patience: int = 2,
        save_log_folder_path: Union[str, None] = None,
        hand_template_info_path: str = None,
        initial_method: str = "middle_plane_fixed",
    ) -> None:
        self.lr = lr
        self.warmup_step_num = warmup_step_num
        self.patience = patience
        self.save_log_folder_path = save_log_folder_path
        self.submesh_num = anchor_num
        self.points_per_submesh = points_per_submesh
        self.initial_method = initial_method

        self.writer = SummaryWriter(log_dir=os.path.join(save_log_folder_path, "tensorboard"), flush_secs=1) if save_log_folder_path else None

        self.hand_mash = HandMash(
            anchor_num=anchor_num,
            mask_degree_max=mask_degree_max,
            sh_degree_max=sh_degree_max,
            sample_phi_num=sample_phi_num,
            sample_theta_num=sample_theta_num,
            dtype=dtype,
            device=device
        )

        self.optimizer = Adam(
            [
                self.hand_mash.hand_params["mask_params"],
                self.hand_mash.hand_params["sh_params"],
                self.hand_mash.hand_params["ortho_poses"]
            ],
            lr=self.lr
        )

        self.hand_gt_points: torch.Tensor | None = None
        self.error = 0
        self.epoch = 0
        self.step = 0

        self.hand_template_info: Dict = np.load(hand_template_info_path, allow_pickle=True).item()

        return
    
    def loadMeshFile(
        self,
        hand_path: str,
    ) -> bool:
        hand_mesh = trimesh.load(hand_path)
        weights = self.hand_template_info["barycentric_coords"]
        triangle_ids = self.hand_template_info["triangle_ids"]
        hand_verts = np.array(hand_mesh.vertices)
        hand_faces = np.array(hand_mesh.faces)
        
        # Get all triangle vertices
        triangles = hand_verts[hand_faces[triangle_ids]]
        # Compute sampled points using vectorized operations
        sampled_points = np.sum(weights[:, :, np.newaxis] * triangles, axis=1)
        
        self.hand_gt_points = torch.from_numpy(sampled_points.reshape(self.submesh_num, self.points_per_submesh, 3)).float()
        self.hand_gt_points = self.hand_gt_points[self.hand_template_info["permutation"], :, :]

        if self.initial_method == "middle_plane":
            self.hand_mash.middlePlaneInit()
        elif self.initial_method == "middle_plane_fixed":
            self.hand_mash.middlePlaneFixInit()
        elif self.initial_method == "sphere_fixed":
            self.hand_mash.sphereFixInit()
        else:
            raise ValueError(f"Unknown initial method: {self.initial_method}")
        
        self.hand_gt_points = self.hand_gt_points.to(self.hand_mash.device)
        self.hand_mash.shParamInit(self.hand_gt_points)

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

        hand_mash_pts = self.hand_mash.toSamplePoints()     # [P, N, 3]

        if torch.cuda.is_available() and self.hand_gt_points.device != "cpu":
            flat_hand_fit_dist2, flat_hand_coverage_dist2 = ChamferDistances.namedAlgo("cuda")(
                hand_mash_pts, 
                self.hand_gt_points
            )[:2]
        else:
            flat_hand_fit_dist2, flat_hand_coverage_dist2 = ChamferDistances.namedAlgo("default")(
                hand_mash_pts, 
                self.hand_gt_points
            )[:2]
        
        flat_hand_dists = torch.sqrt(flat_hand_fit_dist2 + EPSILON)
        flat_hand_coverage_dists = torch.sqrt(flat_hand_coverage_dist2 + EPSILON)

        hand_fit_loss = torch.mean(flat_hand_dists)
        hand_coverage_loss = torch.mean(flat_hand_coverage_dists)

        hand_weighed_fit_loss = fit_loss_weight * hand_fit_loss
        hand_weighed_coverage_loss = coverage_loss_weight * hand_coverage_loss

        loss = hand_weighed_fit_loss + hand_weighed_coverage_loss

        if torch.isnan(loss).any():
            print("loss is nan!")
            return None
        
        loss.backward()
        self.optimizer.step()

        hand_cd = toNumpy(hand_fit_loss) + toNumpy(hand_coverage_loss)
        self.error = hand_cd

        loss_dict = {
            "Train/epoch": self.epoch,
            "Train/step": self.step,
            "Train/error": self.error
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

            if self.hand_mash.sh_degree_max == 2 and loss < threshold:
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

        self.hand_mash.setGradState(True)
        self.warmUpEpoch(1.0, 0.5, self.warmup_step_num)
        self.trainEpoch(1.0, 0.5)

        self.hand_mash.addShDegreeMax()
        self.lr *= lr_decay_rate
        self.optimizer = Adam(
            [
                self.hand_mash.hand_params["mask_params"],
                self.hand_mash.hand_params["sh_params"],
                self.hand_mash.hand_params["ortho_poses"],
            ],
            lr=self.lr
        )

        self.hand_mash.setGradState(True)
        self.warmUpEpoch(1.0, 0.75, self.warmup_step_num)
        self.trainEpoch(1.0, 0.75)

        self.hand_mash.addShDegreeMax()
        self.lr *= lr_decay_rate
        self.optimizer = Adam(
            [
                self.hand_mash.hand_params["mask_params"],
                self.hand_mash.hand_params["sh_params"],
                self.hand_mash.hand_params["ortho_poses"],
            ],
            lr=self.lr
        )

        self.hand_mash.setGradState(True)
        self.warmUpEpoch(1.0, 1.0, self.warmup_step_num)
        final_loss = self.trainEpoch(1.0, 1.0)

        self.closeLog()

        return final_loss
    
    def saveHandMash(self, save_path: str) -> bool:
        self.hand_mash.saveParamsFile(save_path)
        return True


if __name__ == "__main__":
    start_time = time()
    template_info_path = "/data2/yxz/DATASET/template_info/full_barrett_template.npy"
    template_info = np.load(template_info_path, allow_pickle=True).item()
    hand_gt_points = torch.from_numpy(template_info["hand_points"]).float() # [P, N, 3]
    permutation = template_info["permutation"]  # [P]

    hand_gt_points = hand_gt_points[permutation, :, :]
    
    hand_trainer = HandTrainer(
        anchor_num=256,
        mask_degree_max=3,
        sh_degree_max=0,
        sample_phi_num=32,
        sample_theta_num=32,
        points_per_submesh=1024,
        device="cuda",
        dtype=torch.float32,
        lr=0.01,
        warmup_step_num=50,
        patience=2,
        save_log_folder_path=None,
        initial_method="sphere_fixed"
    )
    hand_trainer.loadMeshFile(hand_gt_points)
    hand_trainer.escalateTrainMash()
    hand_trainer.saveHandMash("/data2/yxz/DATASET/template_info/mash_sphere_barrett_template.npy")

    end_time = time()
    print(f"Training time: {end_time - start_time:.6f}")
