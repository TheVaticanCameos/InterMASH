import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from typing import Union, Dict
from torch.optim import Adam
import trimesh
from tensorboardX import SummaryWriter
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from time import time

sys.path.append("../ma-sh")
sys.path.append("../chamfer-distance")
from chamfer_distance.Module.chamfer_distances import ChamferDistances

from ma_sh.Config.constant import EPSILON
from ma_sh.Method.data import toNumpy
from ma_sh.Model.object_mash import ObjectMash
from ma_sh.Module.early_stopping import EarlyStopping
from ma_sh.Module.logger import Logger


class ObjectTrainer:
    def __init__(
        self,
        initial_method: str = "voxel",
        anchor_num: int = 400,
        mask_degree_max: int = 3,
        sh_degree_max: int = 2,
        sample_phi_num: int = 40,
        sample_theta_num: int = 40,
        points_per_submesh: int = 1024,
        dtype: torch.dtype = torch.float32,
        device: str = "cuda",
        lr: float = 2e-3,
        min_lr: float = 1e-3,
        warmup_step_num: int = 80,
        warmup_epoch: int = 4,
        factor: float = 0.8,
        patience: int = 2,
        render: bool = True,
        render_freq: int = 1,
        render_init_only: bool = False,
        save_freq: int = 1,
        save_result_folder_path: Union[str, None] = None,
        save_log_folder_path: Union[str, None] = None,
        hand_template_info_path: str = None,
        object_template_info_path: str | None = None,
        hand_type: str = "mano",
        use_closed_form_cmap: bool = True
    ) -> None:
        self.lr = lr
        self.min_lr = min_lr
        self.warmup_step_num = warmup_step_num
        self.warmup_epoch = warmup_epoch
        self.factor = factor
        self.patience = patience

        self.epoch = 0
        self.step = 0
        self.loss_min = float("inf")

        self.best_params_dict = {}

        self.render = render
        self.render_freq = render_freq
        self.render_init_only = render_init_only
        self.save_freq = save_freq

        self.save_result_folder_path = save_result_folder_path
        self.save_log_folder_path = save_log_folder_path
        self.save_file_idx = 0
        self.logger = Logger()

        self.writer = SummaryWriter(log_dir=os.path.join(save_log_folder_path, "tensorboard"), flush_secs=1) if save_log_folder_path else None

        self.hand_gt_points = None
        self.object_gt_points = None
        self.cmap = None

        self.initial_method = initial_method
        self.hand_type = hand_type
        self.use_closed_form_cmap = use_closed_form_cmap

        self.object_mash = ObjectMash(
            anchor_num,
            mask_degree_max,
            sh_degree_max,
            sample_phi_num,
            sample_theta_num,
            dtype,
            device,
        )

        self.submesh_num = anchor_num
        self.points_per_submesh = points_per_submesh

        self.optimizer = Adam(
            [
                self.object_mash.object_params["mask_params"],
                self.object_mash.object_params["sh_params"],
                self.object_mash.object_params["ortho_poses"]
            ],
            lr=self.lr
        )

        self.min_lr_reach_time = 0

        self.error = 0
        self.object_error = 0

        self.object_template_info: Dict = np.load(object_template_info_path, allow_pickle=True).item()
        self.hand_template_info: Dict = np.load(hand_template_info_path, allow_pickle=True).item()
        
        return

    def loadMeshFile(
        self, 
        hand_path: str,
        object_path: str,
    ) -> bool:
        if not os.path.exists(hand_path): 
            print("[ERROR][ObjectTrainer::loadMeshFile]")
            print("\t hand mesh file not exist!")
            print("\t hand_path:", hand_path)
            return False
        if not os.path.exists(object_path): 
            print("[ERROR][ObjectTrainer::loadMeshFile]")
            print("\t object mesh file not exist!")
            print("\t object_path:", object_path)
            return False
        
        self.hand_gt_points = self.getInitialHandGT(hand_path)
        self.object_gt_points = self.getInitialObjectGT(object_path)

        # initialize mash parameters here 
        if self.initial_method == "middle_plane":
            print("[INFO][ObjectTrainer::loadMeshFile]")
            print("\t initial_method: middle_plane")
            self.object_mash.middlePlaneInit(self.hand_gt_points, self.object_gt_points)
        else:
            print("[ERROR][ObjectTrainer::loadMeshFile]")
            print("\t initial_method not support!")
            print("\t initial_method:", self.initial_method)
            return False

        self.matchObjectGT()

        self.object_mash.shParamInit(self.object_gt_points)

        return True
    
    def getInitialHandGT(self, hand_path: str) -> torch.Tensor:
        hand_mesh = trimesh.load(hand_path)
        weights = self.hand_template_info["barycentric_coords"]
        triangle_ids = self.hand_template_info["triangle_ids"]
        hand_verts = np.array(hand_mesh.vertices)
        hand_faces = np.array(hand_mesh.faces)
        
        # Get all triangle vertices
        triangles = hand_verts[hand_faces[triangle_ids]]
        # Compute sampled points using vectorized operations
        sampled_points = np.sum(weights[:, :, np.newaxis] * triangles, axis=1)
        return torch.from_numpy(sampled_points.reshape(self.submesh_num, self.points_per_submesh, 3))

    def getInitialObjectGT(self, object_path: str) -> torch.Tensor:
        object_mesh = trimesh.load(object_path)
        weights = self.object_template_info["barycentric_coords"]
        triangle_ids = self.object_template_info["triangle_ids"]
        object_verts = np.array(object_mesh.vertices)
        object_faces = np.array(object_mesh.faces)

        # Get all triangle vertices
        triangles = object_verts[object_faces[triangle_ids]]
        # Compute sampled points using vectorized operations
        sampled_points = np.sum(weights[:, :, np.newaxis] * triangles, axis=1)
        return torch.from_numpy(sampled_points.reshape(self.submesh_num, self.points_per_submesh, 3))

    def matchObjectGT(self) -> bool:
        object_gt_numpy = self.object_gt_points.numpy()
        patch_centers = object_gt_numpy.mean(axis=1)
        cost_matrix = cdist(self.object_mash.positions.cpu().numpy(), patch_centers, metric="euclidean")

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        assert (row_ind == np.arange(self.submesh_num)).all(), "LAP ERROR: row_ind must equals to [1,2,..., submesh_num]"
        self.object_gt_points = self.object_gt_points[col_ind]
        
        return True
    
    def saveMashFile(self, save_mash_file_path: str) -> bool:
        self.object_mash.saveParamsFile(save_mash_file_path)
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
        object_gt_points: torch.Tensor,
        fit_loss_weight: float,
        coverage_loss_weight: float,
        scheduler = None
    ) -> Union[Dict, None]:
        self.optimizer.zero_grad()

        object_mash_pts = self.object_mash.toSamplePoints("object").reshape(self.object_mash.anchor_num, -1, 3)

        object_fit_loss = torch.tensor(0.0).type(object_gt_points.dtype).to(object_gt_points.device)
        object_coverage_loss = torch.zeros_like(object_fit_loss)

        if torch.cuda.is_available() and object_gt_points.device != "cpu":
            object_fit_dist2, object_coverage_dist2 = ChamferDistances.namedAlgo("cuda")(
                object_mash_pts, object_gt_points
            )[:2]
        else:
            object_fit_dist2, object_coverage_dist2 = ChamferDistances.namedAlgo("default")(
                object_mash_pts, object_gt_points
            )[:2]
        
        object_fit_dists = torch.sqrt(object_fit_dist2 + EPSILON)
        object_coverage_dists = torch.sqrt(object_coverage_dist2 + EPSILON)

        object_fit_loss = torch.mean(object_fit_dists)
        object_coverage_loss = torch.mean(object_coverage_dists)

        object_weighted_fit_loss = fit_loss_weight * object_fit_loss
        object_weighted_coverage_loss = coverage_loss_weight * object_coverage_loss

        loss = object_weighted_fit_loss + object_weighted_coverage_loss

        if torch.isnan(loss).any():
            print("[ERROR][BaseTrainer::trainStep]")
            print("\t loss is nan!")
            print("\t\t object_fit_loss:", object_fit_loss)
            print("\t\t object_coverage_loss:", object_coverage_loss)
            return None

        loss.backward()

        # self.hoi_mash.clearNanGrads()
        
        self.optimizer.step()
        if scheduler is not None:
            scheduler.step()
        
        object_cd = toNumpy(object_fit_loss) + toNumpy(object_coverage_loss)
        self.error = object_cd
        self.object_error = object_cd

        loss_dict = {
            "Train/epoch": self.epoch,
            "Train/object_weighted_fit_loss": toNumpy(object_weighted_fit_loss),
            "Train/object_weighted_coverage_loss": toNumpy(object_weighted_coverage_loss),
            "Train/object_cd": object_cd,
            "Train/loss": toNumpy(loss),
            "Metric/chamfer_distance": self.error,
        }

        for name, value in loss_dict.items():
            if name != "Train/epoch" and self.writer is not None:
                self.writer.add_scalar(name, value, self.step)
        if self.writer is not None:
            self.writer.add_scalar("Train/lr", self.optimizer.param_groups[0]["lr"], self.step)

        return loss_dict

    def warmUpEpoch(
        self, 
        epoch_lr: float,
        object_gt_points: torch.Tensor,
        fit_loss_weight: float,
        coverage_loss_weight: float,
        train_step_max: int,
    ) -> bool:
        print("[INFO][ObjectTrainer::warmUpEpoch]")
        print("\t start train epoch :", self.epoch, "with lr :", epoch_lr, "...")
        pbar = tqdm(total=train_step_max)

        start_step = self.step

        for i in range(train_step_max):
            current_lr = epoch_lr * (i + 1.0) / train_step_max
            self.updateLr(current_lr)

            loss_dict= self.trainStep(
                object_gt_points, 
                fit_loss_weight, 
                coverage_loss_weight
            )

            if loss_dict is None:
                print("[ERROR][BaseTrainer::warmUpEpoch]")
                print("\t trainStep failed!")
                return False

            assert isinstance(loss_dict, dict)
            
            loss = loss_dict["Train/loss"]

            pbar.set_description("LOSS %.6f" % (loss,))

            self.step += 1
            pbar.update(1)

            if self.step >= start_step + train_step_max:
                break
        
        self.epoch += 1
        return True

    def trainEpoch(
        self,
        epoch_lr: float,
        object_gt_points: torch.Tensor,
        fit_loss_weight: float,
        coverage_loss_weight: float,
        train_step_max: Union[int, None] = None,
    ) -> bool:
        early_stopping = EarlyStopping(self.patience, 1e-6)

        self.updateLr(epoch_lr)

        print("[INFO][ObjectTrainer::trainEpoch]")
        print("\t start train epoch :", self.epoch, "with lr :", epoch_lr, "...")
        if train_step_max is not None:
            pbar = tqdm(total=train_step_max)
        else:
            pbar = tqdm()

        start_step = self.step
        min_loss = float("inf")
        min_loss_reached_time = 0

        while True:
            loss_dict = self.trainStep(
                object_gt_points,
                fit_loss_weight,
                coverage_loss_weight
            )

            if loss_dict is None:
                print("[ERROR][ObjectTrainer::trainEpoch]")
                print("\t trainStep failed!")
                return False

            assert isinstance(loss_dict, dict)
            
            loss = loss_dict["Train/loss"]

            pbar.set_description("LOSS %.6f" % (loss,))

            self.step += 1
            pbar.update(1)

            if train_step_max and self.step >= start_step + train_step_max:
                break
            else:
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
        return True

    def escalateTrainMash(self) -> bool:
        lr_decay_rate= 0.2
        gt_points_dtype = self.object_mash.dtype

        object_gt_points = self.object_gt_points.to(self.object_mash.device, dtype=gt_points_dtype)
        
        print("[INFO][ObjectTrainer::escalateTrainMash]")
        print("\t start escalate train ObjectMash...")
        
        # first step: training with sh_degree_max = 0
        self.object_mash.setGradState(True)
        print(
            "\t degree: mask:",
            self.object_mash.mask_degree_max,
            "sh:",
            self.object_mash.sh_degree_max,
        )

        print("[INFO][ObjectTrainer::escalateTrainMash]")
        print("\t start warmUpEpoch...")
        if not self.warmUpEpoch(self.lr, object_gt_points, 1.0, 0.5, self.warmup_step_num):
            print("[ERROR][Trainer::escalateTrainMash]")
            print("\t warmUpEpoch failed!")
            return False
        
        print("[INFO][ObjectTrainer::escalateTrainMash]")
        print("\t start trainEpoch...")
        if not self.trainEpoch(self.lr, object_gt_points, 1.0, 0.5):
            print("[ERROR][Trainer::escalateTrainMash]")
            print("\t trainEpoch failed!")
            return False
        
        # second step: training with sh_degree_max = 1
        self.object_mash.addShDegreeMax()
        self.lr *= lr_decay_rate
        self.optimizer = Adam(
            [
                self.object_mash.object_params["mask_params"],
                self.object_mash.object_params["sh_params"],
                self.object_mash.object_params["ortho_poses"],
            ],
            lr=self.lr
        )

        self.object_mash.setGradState(True)
        print(
            "\t degree: mask:",
            self.object_mash.mask_degree_max,
            "sh:",
            self.object_mash.sh_degree_max,
        )
        
        print("[INFO][ObjectTrainer::escalateTrainMash]")
        print("\t start warmUpEpoch...")
        if not self.warmUpEpoch(self.lr, object_gt_points, 1.0, 0.75, self.warmup_step_num):
            print("[ERROR][Trainer::escalateTrainMash]")
            print("\t warmUpEpoch failed!")
            return False
        
        print("[INFO][ObjectTrainer::escalateTrainMash]")
        print("\t start trainEpoch...")
        if not self.trainEpoch(self.lr, object_gt_points, 1.0, 0.75):
            print("[ERROR][Trainer::escalateTrainMash]")
            print("\t trainEpoch failed!")
            return False

        # second step: training with sh_degree_max = 2
        self.object_mash.addShDegreeMax()
        self.lr *= lr_decay_rate
        self.optimizer = Adam(
            [
                self.object_mash.object_params["mask_params"],
                self.object_mash.object_params["sh_params"],
                self.object_mash.object_params["ortho_poses"],
            ],
            lr=self.lr
        )

        self.object_mash.setGradState(True)
        print(
            "\t degree: mask:",
            self.object_mash.mask_degree_max,
            "sh:",
            self.object_mash.sh_degree_max,
        )
        
        print("[INFO][ObjectTrainer::escalateTrainMash]")
        print("\t start warmUpEpoch...")
        if not self.warmUpEpoch(self.lr, object_gt_points, 1.0, 1.0, self.warmup_step_num):
            print("[ERROR][Trainer::escalateTrainMash]")
            print("\t warmUpEpoch failed!")
            return False
        
        print("[INFO][ObjectTrainer::escalateTrainMash]")
        print("\t start trainEpoch...")
        if not self.trainEpoch(self.lr, object_gt_points, 1.0, 1.0):
            print("[ERROR][Trainer::escalateTrainMash]")
            print("\t trainEpoch failed!")
            return False
        
        # training finished
        print("[INFO][ObjectTrainer::escalateTrainMash]")
        print("\t escalate train ObjectMash finished!")
        print("\t total error: {:.5f}".format(self.error))
        print("\t object_error: {:.5f}".format(self.object_error))

        # training finished
        print("[INFO][ObjectTrainer::escalateTrainMash]")
        print("\t escalate train ObjectMash finished!")
        print("\t total error: {:.5f}".format(self.error))
        print("\t object_error: {:.5f}".format(self.object_error))

        self.closeLog()

        return True


if __name__ == "__main__":
    start_time = time()
    trainer = ObjectTrainer(
        initial_method="middle_plane",
        anchor_num=256,
        mask_degree_max=3,
        sh_degree_max=0,
        sample_phi_num=32,
        sample_theta_num=32,
        points_per_submesh=1024,
        dtype=torch.float32,
        device="cuda:0",
        lr=0.02,
        min_lr=1e-3,
        warmup_step_num=50,
        warmup_epoch=4,
        factor=0.8,
        patience=2,
        object_template_info_path="/data2/yxz/DATASET/DRO-data/DRO-Original/shadow/object_template/cylinder_medium.npy",
        hand_template_info_path="/data2/yxz/DATASET/template_info/shadow_template.npy"
    )

    trainer.loadMeshFile(
        hand_path="/data2/yxz/DATASET/DRO-data/DRO-Original/shadow/val/shadow_contactdb+cylinder_medium_108_hand.ply",
        object_path="/data2/yxz/DATASET/DRO-data/DRO-Original/shadow/val/shadow_contactdb+cylinder_medium_108_object.ply"
    )
    trainer.escalateTrainMash()
    end_time = time()
    print("\t total time: {:.5f}".format(end_time - start_time))
