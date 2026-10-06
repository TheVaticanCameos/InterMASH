import os
import torch
import numpy as np
from tqdm import tqdm
from typing import Union, Dict
from torch.optim import Adam
import trimesh
from tensorboardX import SummaryWriter
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

import mash_cpp

from chamfer_distance.Module.chamfer_distances import ChamferDistances

from ma_sh.Config.constant import EPSILON
from ma_sh.Method.data import toNumpy
from ma_sh.Model.hoi_mash import HOIMash
from ma_sh.Module.early_stopping import EarlyStopping
from ma_sh.Module.logger import Logger


class HOITrainer(object):
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

        self.hoi_mash = HOIMash(
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
                self.hoi_mash.hand_params["mask_params"],
                self.hoi_mash.hand_params["sh_params"],
                self.hoi_mash.hand_params["ortho_poses"],
                self.hoi_mash.object_params["mask_params"],
                self.hoi_mash.object_params["sh_params"],
                self.hoi_mash.object_params["ortho_poses"]
            ],
            lr=self.lr
        )

        self.min_lr_reach_time = 0

        self.error = 0
        self.hand_error = 0
        self.object_error = 0

        self.hand_template_info: Dict = np.load(hand_template_info_path, allow_pickle=True).item()
        self.object_template_info: Dict = np.load(object_template_info_path, allow_pickle=True).item()
        
        return

    def loadMeshFile(
        self, 
        hand_path: str,
        object_path: str,
        cmap_path: str | None = None
    ) -> bool:
        if not os.path.exists(hand_path): 
            print("[ERROR][MeshTrainer::loadMeshFile]")
            print("\t hand mesh file not exist!")
            print("\t hand_path:", hand_path)
            return False
        if not os.path.exists(object_path): 
            print("[ERROR][MeshTrainer::loadMeshFile]")
            print("\t object mesh file not exist!")
            print("\t object_path:", object_path)
            return False
        
        self.hand_gt_points = self.getInitialHandGT(hand_path)
        self.object_gt_points = self.getInitialObjectGT(object_path)

        if cmap_path is not None:
            data = np.load(cmap_path)
            self.cmap = {
                "pc": data[:, :3],
                "cmap": data[:, 3]
            }

        # initialize mash parameters here 
        if self.initial_method == "random":
            print("[INFO][HOITrainer::loadMeshFile]")
            print("\t initial_method: random")
            self.hoi_mash.randomInit()
        elif self.initial_method == "voxel":
            print("[INFO][HOITrainer::loadMeshFile]")
            print("\t initial_method: voxel")
            self.hoi_mash.voxelInit(self.hand_gt_points, self.object_gt_points)
        elif self.initial_method == "middle_plane":
            print("[INFO][HOITrainer::loadMeshFile]")
            print("\t initial_method: middle_plane")
            self.hoi_mash.middlePlaneInit(self.hand_gt_points, self.object_gt_points)
        elif self.initial_method == "middle_plane_fixed":
            print("[INFO][HOITrainer::loadMeshFile]")
            print("\t initial_method: middle_plane_fixed")
            self.hoi_mash.middlePlaneFixInit(self.hand_gt_points, self.object_gt_points)
        elif self.initial_method == "sphere_fixed":
            print("[INFO][HOITrainer::loadMeshFile]")
            print("\t initial_method: sphere_fixed")
            self.hoi_mash.sphereFixInit()
        else:
            print("[ERROR][HOITrainer::loadMeshFile]")
            print("\t initial_method not support!")
            print("\t initial_method:", self.initial_method)
            return False

        self.matchHandGT()
        self.matchObjectGT()

        self.hoi_mash.shParamInit(self.hand_gt_points, self.object_gt_points)

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

    def matchHandGT(self) -> bool:
        hand_gt_numpy = self.hand_gt_points.numpy()
        patch_centers = hand_gt_numpy.mean(axis=1)
        cost_matrix = cdist(self.hoi_mash.positions.cpu().numpy(), patch_centers, metric="euclidean")

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        assert (row_ind == np.arange(self.submesh_num)).all(), "LAP ERROR: row_ind must equals to [1,2,..., submesh_num]"
        self.hand_gt_points = self.hand_gt_points[col_ind]
        
        return True

    def matchObjectGT(self) -> bool:
        object_gt_numpy = self.object_gt_points.numpy()
        patch_centers = object_gt_numpy.mean(axis=1)
        cost_matrix = cdist(self.hoi_mash.positions.cpu().numpy(), patch_centers, metric="euclidean")

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        assert (row_ind == np.arange(self.submesh_num)).all(), "LAP ERROR: row_ind must equals to [1,2,..., submesh_num]"
        self.object_gt_points = self.object_gt_points[col_ind]
        
        return True

    def regulate(self) -> bool:
        if self.hand_gt_points is None or self.object_gt_points is None:
            print("[ERROR][HOITrainer::regulate]")
            print("\t hand_gt_points or object_gt_points is None!")
            return False

        # Compute the centers of self.hand_gt_points and self.object_gt_points
        hand_center = torch.mean(self.hand_gt_points.reshape(-1, 3), dim=0)
        object_center = torch.mean(self.object_gt_points.reshape(-1, 3), dim=0)

        # Compute the midpoint between the centers
        mid_point = (hand_center + object_center) / 2

        # Translate the midpoint to the origin
        self.hand_gt_points -= mid_point
        self.object_gt_points -= mid_point

        # Compute the vector from the object center to the hand center
        direction = hand_center - object_center
        direction = direction / torch.norm(direction)  # Normalize the vector

        # Compute the rotation axis and angle
        z_axis = torch.tensor([0, 0, 1], dtype=direction.dtype, device=direction.device)
        axis = torch.linalg.cross(direction, z_axis)
        axis_norm = torch.norm(axis)
        if axis_norm < 1e-6:  # The vector is already parallel to the z-axis
            return True

        axis = axis / axis_norm
        cos_theta = torch.dot(direction, z_axis)
        theta = torch.acos(cos_theta)

        # Construct the rotation matrix
        K = torch.tensor([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ], dtype=direction.dtype, device=direction.device)
        rotation_matrix = torch.eye(3, dtype=direction.dtype, device=direction.device) + torch.sin(theta) * K + (1 - torch.cos(theta)) * torch.matmul(K, K)

        # Apply the rotation matrix
        self.hand_gt_points = torch.matmul(self.hand_gt_points, rotation_matrix.T)
        self.object_gt_points = torch.matmul(self.object_gt_points, rotation_matrix.T)

        return True

    def randomRotate(self) -> bool:
        if self.hand_gt_points is None or self.object_gt_points is None:
            print("[ERROR][HOITrainer::randomRotate]")
            print("\t hand_gt_points or object_gt_points is None!")
            return False

        theta = torch.rand(1, device=self.hand_gt_points.device, dtype=self.hand_gt_points.dtype) * 2 * torch.pi

        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        rotation_matrix = torch.tensor([
            [cos_theta, -sin_theta, 0],
            [sin_theta, cos_theta, 0],
            [0, 0, 1]
        ], device=self.hand_gt_points.device, dtype=self.hand_gt_points.dtype).squeeze()

        original_shape_hand = self.hand_gt_points.shape
        original_shape_object = self.object_gt_points.shape

        self.hand_gt_points = torch.matmul(self.hand_gt_points.reshape(-1, 3), rotation_matrix.T).reshape(original_shape_hand)
        self.object_gt_points = torch.matmul(self.object_gt_points.reshape(-1, 3), rotation_matrix.T).reshape(original_shape_object)

        return True

    def erasePositionsZCoord(self) -> bool:
        self.hoi_mash.positions[:, 2] = 0
        return True
    
    def saveMashFile(self, save_mash_file_path: str) -> bool:
        self.hoi_mash.saveParamsFile(save_mash_file_path)
        return True

    def updateLr(self, lr: float) -> bool:
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return True

    def closeLog(self) -> None:
        if self.writer is not None:
            self.writer.close()
        return
    
    def upSampleCMapGT(self) -> torch.Tensor:
        upSampledPC = self.hoi_mash.toSamplePoints("object").view(-1, 3)
        if self.cmap is None or "pc" not in self.cmap or "cmap" not in self.cmap:
            if self.hoi_mash.device == "cpu":
                gt_dtype = self.hoi_mash.dtype
            else:
                gt_dtype = torch.float32
            cmap_gt = torch.zeros(upSampledPC.shape[0], device=self.hoi_mash.device, dtype=gt_dtype)
            self.cmap_gt = cmap_gt
            return cmap_gt

        gt_dtype = self.hoi_mash.dtype

        with torch.no_grad():
            pc_sparse = self.cmap["pc"]
            cmap_sparse = self.cmap["cmap"]

            if isinstance(pc_sparse, torch.Tensor):
                p = pc_sparse.detach().to(self.hoi_mash.device, dtype=gt_dtype)
            else:
                p = torch.from_numpy(np.asarray(pc_sparse)).to(self.hoi_mash.device, dtype=gt_dtype)

            if isinstance(cmap_sparse, torch.Tensor):
                v = cmap_sparse.detach().to(self.hoi_mash.device, dtype=gt_dtype).view(-1)
            else:
                v = torch.from_numpy(np.asarray(cmap_sparse)).to(self.hoi_mash.device, dtype=gt_dtype).view(-1)

            q = upSampledPC.to(self.hoi_mash.device, dtype=gt_dtype)

            n = p.shape[0]
            k = int(min(8, max(1, n)))
            eps = 1e-8

            pred = torch.zeros(q.shape[0], device=self.hoi_mash.device, dtype=gt_dtype)
            # chunk_size = 32768
            chunk_size = 8192
            for s in range(0, q.shape[0], chunk_size):
                e = min(s + chunk_size, q.shape[0])
                qi = q[s:e]
                di = torch.cdist(qi, p)
                k_use = min(k, n)
                dk, idxk = torch.topk(di, k_use, dim=1, largest=False)
                vals = v[idxk]
                w = 1.0 / ((dk + eps) ** 2)
                denom = w.sum(dim=1)
                num = (w * vals).sum(dim=1)
                predi = num / torch.maximum(denom, torch.tensor(eps, device=self.hoi_mash.device, dtype=gt_dtype))
                has_zero = (dk <= eps).any(dim=1)
                if has_zero.any():
                    zero_first = (dk <= eps).float().argmax(dim=1)
                    rows = torch.nonzero(has_zero, as_tuple=True)[0]
                    predi[rows] = vals[rows, zero_first[rows]]
                pred[s:e] = predi

            cmap_gt = pred.detach()
            self.cmap_gt = cmap_gt
            return cmap_gt

    def trainStep(
        self,
        hand_gt_points: torch.Tensor,
        object_gt_points: torch.Tensor,
        fit_loss_weight: float,
        coverage_loss_weight: float,
        scheduler = None
    ) -> Union[Dict, None]:
        self.optimizer.zero_grad()

        hand_mash_pts = self.hoi_mash.toSamplePoints("hand").reshape(self.hoi_mash.anchor_num, -1, 3)
        object_mash_pts = self.hoi_mash.toSamplePoints("object").reshape(self.hoi_mash.anchor_num, -1, 3)

        hand_fit_loss = torch.tensor(0.0).type(hand_gt_points.dtype).to(hand_gt_points.device)
        hand_coverage_loss = torch.zeros_like(hand_fit_loss)
        object_fit_loss = torch.tensor(0.0).type(object_gt_points.dtype).to(object_gt_points.device)
        object_coverage_loss = torch.zeros_like(object_fit_loss)

        if torch.cuda.is_available() and hand_gt_points.device != "cpu" and object_gt_points.device != "cpu":
            hand_fit_dist2, hand_coverage_dist2 = ChamferDistances.namedAlgo("cuda")(
                hand_mash_pts, hand_gt_points
            )[:2]
            object_fit_dist2, object_coverage_dist2 = ChamferDistances.namedAlgo("cuda")(
                object_mash_pts, object_gt_points
            )[:2]
        else:
            hand_fit_dist2, hand_coverage_dist2 = ChamferDistances.namedAlgo("default")(
                hand_mash_pts, hand_gt_points
            )[:2]
            object_fit_dist2, object_coverage_dist2 = ChamferDistances.namedAlgo("default")(
                object_mash_pts, object_gt_points
            )[:2]
        
        hand_fit_dists = torch.sqrt(hand_fit_dist2 + EPSILON)
        hand_coverage_dists = torch.sqrt(hand_coverage_dist2 + EPSILON)
        object_fit_dists = torch.sqrt(object_fit_dist2 + EPSILON)
        object_coverage_dists = torch.sqrt(object_coverage_dist2 + EPSILON)

        hand_fit_loss = torch.mean(hand_fit_dists)
        hand_coverage_loss = torch.mean(hand_coverage_dists)
        object_fit_loss = torch.mean(object_fit_dists)
        object_coverage_loss = torch.mean(object_coverage_dists)

        hand_weighted_fit_loss = fit_loss_weight * hand_fit_loss
        hand_weighted_coverage_loss = coverage_loss_weight * hand_coverage_loss
        object_weighted_fit_loss = fit_loss_weight * object_fit_loss
        object_weighted_coverage_loss = coverage_loss_weight * object_coverage_loss

        loss = hand_weighted_fit_loss + hand_weighted_coverage_loss + object_weighted_fit_loss + object_weighted_coverage_loss

        if torch.isnan(loss).any():
            print("[ERROR][BaseTrainer::trainStep]")
            print("\t loss is nan!")
            print("\t\t hand_fit_loss:", hand_fit_loss)
            print("\t\t hand_coverage_loss:", hand_coverage_loss)
            print("\t\t object_fit_loss:", object_fit_loss)
            print("\t\t object_coverage_loss:", object_coverage_loss)
            return None

        loss.backward()

        # self.hoi_mash.clearNanGrads()
        
        self.optimizer.step()
        if scheduler is not None:
            scheduler.step()

        hand_cd = toNumpy(hand_fit_loss) + toNumpy(hand_coverage_loss)
        object_cd = toNumpy(object_fit_loss) + toNumpy(object_coverage_loss)
        self.error = hand_cd + object_cd
        self.hand_error = hand_cd
        self.object_error = object_cd

        loss_dict = {
            "Train/epoch": self.epoch,
            "Train/hand_weighted_fit_loss": toNumpy(hand_weighted_fit_loss),
            "Train/hand_weighted_coverage_loss": toNumpy(hand_weighted_coverage_loss),
            "Train/object_weighted_fit_loss": toNumpy(object_weighted_fit_loss),
            "Train/object_weighted_coverage_loss": toNumpy(object_weighted_coverage_loss),
            "Train/hand_cd": hand_cd,
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
        hand_gt_points: torch.Tensor, 
        object_gt_points: torch.Tensor,
        fit_loss_weight: float,
        coverage_loss_weight: float,
        train_step_max: int,
    ) -> bool:
        self.hoi_mash.setGradState(True)

        print("[INFO][BaseTrainer::warmUpEpoch]")
        print("\t start train epoch :", self.epoch, "with lr :", epoch_lr, "...")
        pbar = tqdm(total=train_step_max)

        start_step = self.step

        for i in range(train_step_max):
            current_lr = epoch_lr * (i + 1.0) / train_step_max
            self.updateLr(current_lr)

            loss_dict= self.trainStep(
                hand_gt_points, 
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
        hand_gt_points: torch.Tensor,
        object_gt_points: torch.Tensor,
        fit_loss_weight: float,
        coverage_loss_weight: float,
        train_step_max: Union[int, None] = None,
    ) -> bool:
        early_stopping = EarlyStopping(self.patience, 1e-6)

        self.hoi_mash.setGradState(True)

        self.updateLr(epoch_lr)

        print("[INFO][BaseTrainer::trainEpoch]")
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
                hand_gt_points,
                object_gt_points,
                fit_loss_weight,
                coverage_loss_weight
            )

            if loss_dict is None:
                print("[ERROR][BaseTrainer::trainEpoch]")
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
    
    def trainStepCMap(
        self,
        cmap_gt: torch.Tensor,
        scheduler = None
    ) -> Union[Dict, None]:
        self.optimizer.zero_grad(set_to_none=True)

        self.step += 1

        cmap_mash = self.hoi_mash.toCMap().view(-1)
        loss = torch.nn.functional.mse_loss(cmap_mash, cmap_gt)

        if torch.isnan(loss).any():
            print("[ERROR][BaseTrainer::trainStepCMap]")
            print("\t loss is nan!")
            print("\t\t cmap_mash:", cmap_mash)
            print("\t\t cmap_gt:", cmap_gt)
            return None

        loss.backward()

        self.optimizer.step()
        if scheduler is not None:
            scheduler.step()
        
        loss_dict = {
            "Epoch": self.epoch,
            "CMap/cmap_loss": toNumpy(loss),
        }

        if self.writer is not None:
            for name, value in loss_dict.items():
                if name != "Epoch":
                    self.writer.add_scalar(name, value, self.step)
            self.writer.add_scalar("LR/lr", self.optimizer.param_groups[0]["lr"], self.step)

        return loss_dict

    def trainCMap(
        self,
        cmap_gt: torch.Tensor,
        epoch_num: int,
        lr: float,
    ) -> float:
        self.hoi_mash.setGradState(False)
        self.hoi_mash.cmap_params["sh_params"].requires_grad_(True)
        self.optimizer = Adam([
            self.hoi_mash.cmap_params["sh_params"]
        ], lr=lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=lr,
            epochs=epoch_num,
            steps_per_epoch=1,
            pct_start=0.1,
        )
        with tqdm(total=epoch_num, desc="CMap training") as pbar:
            for _ in range(epoch_num):
                loss_dict = self.trainStepCMap(cmap_gt, scheduler)
                if loss_dict is None:
                    print("[ERROR][Trainer::escalateTrainMash]")
                    print("\t trainStepCMap failed!")
                    return False
                
                pbar.set_postfix({"loss": f"{loss_dict['CMap/cmap_loss']:.5f}"})
                pbar.update(1)

        return loss_dict["CMap/cmap_loss"].item()

    def fitCMapLeastSquares(self, cmap_gt: torch.Tensor) -> float:
        weighted = self.hoi_mash.toWeightedSamplePhiThetaMat("object")
        phis = weighted[..., 0]
        thetas = weighted[..., 1]
        deg = int((self.hoi_mash.cmap_params["sh_params"].shape[1]) ** 0.5 - 1)
        base_values = mash_cpp.toSHBaseValues(phis, thetas, deg)
        base_values = base_values.transpose(1, 0).detach()
        if base_values.dim() == 2:
            B = base_values.unsqueeze(0).unsqueeze(2)
            Y = cmap_gt.view(1, -1, 1)
        else:
            B = base_values.reshape(self.hoi_mash.anchor_num, base_values.shape[1], -1).permute(0, 2, 1)
            Y = cmap_gt.view(self.hoi_mash.anchor_num, -1, 1)
        Bt = B.transpose(1, 2)
        BtB = torch.bmm(Bt, B)
        n = BtB.shape[-1]
        I = torch.eye(n, dtype=BtB.dtype, device=BtB.device).unsqueeze(0).expand(BtB.shape[0], n, n)
        X = torch.linalg.solve(BtB + 1e-6 * I, torch.bmm(Bt, Y))
        self.hoi_mash.cmap_params["sh_params"].data = X.squeeze(-1)
        pred = torch.bmm(B, X).squeeze(-1)
        loss = torch.nn.functional.mse_loss(pred, Y.squeeze(-1))
        return loss.item()

    def autoTrainMash(self) -> bool:        
        print("[INFO][HOITrainer::autoTrainMash]")
        print("\t start auto train SimpleMash...")
        print(
            "\t degree: mask:",
            self.hoi_mash.mask_degree_max,
            "sh:",
            self.hoi_mash.sh_degree_max,
        )

        gt_points_dtype = self.hoi_mash.dtype

        hand_gt_points = self.hand_gt_points.to(self.hoi_mash.device, dtype=gt_points_dtype)
        object_gt_points = self.object_gt_points.to(self.hoi_mash.device, dtype=gt_points_dtype)

        print("[INFO][HOITrainer::autoTrainMash]")
        print("\t start warmUpEpoch...") 
        if not self.warmUpEpoch(self.lr, hand_gt_points, object_gt_points, 1.0, 0.5, self.warmup_step_num):
            print("[ERROR][Trainer::autoTrainMash]")
            print("\t warmUpEpoch failed!")
            return False
        
        print("[INFO][HOITrainer::autoTrainMash]")
        print("\t start trainEpoch...")
        if not self.trainEpoch(self.lr, hand_gt_points, object_gt_points, 1.0, 0.5):
            print("[ERROR][Trainer::autoTrainMash]")
            print("\t trainEpoch failed!")
            return False
        
        print("[INFO][HOITrainer::autoTrainMash]")
        print("\t start second warmUpEpoch...")
        if not self.warmUpEpoch(self.lr, hand_gt_points, object_gt_points, 1.0, 1.0, self.warmup_step_num):
            print("[ERROR][Trainer::autoTrainMash]")
            print("\t warmUpEpoch failed!")
            return False

        print("[INFO][HOITrainer::autoTrainMash]")
        print("\t start second trainEpoch...")
        if not self.trainEpoch(self.lr, hand_gt_points, object_gt_points, 1.0, 1.0):
            print("[ERROR][Trainer::autoTrainMash]")
            print("\t trainEpoch failed!")
            return False
        
        print("[INFO][Trainer::autoTrainMash]")
        print("\t training finished! metrics:")
        print("\t total error: {:.5f}".format(self.error))
        print("\t hand_error: {:.5f}".format(self.hand_error))
        print("\t object_error: {:.5f}".format(self.object_error))

        if self.render:
            if not self.renderErrorMap(hand_gt_points, object_gt_points):
                print("[ERROR][Trainer::autoTrainMash]")
                print("\t renderErrorMap failed!")
                return False
        
        self.closeLog()

        return True

    def directTrainMash(self, epoch_num: int = 800) -> bool:
        gt_points_dtype = self.hoi_mash.dtype

        hand_gt_points = self.hand_gt_points.to(self.hoi_mash.device, dtype=gt_points_dtype)
        object_gt_points = self.object_gt_points.to(self.hoi_mash.device, dtype=gt_points_dtype)

        print("[INFO][Trainer::directTrainMash]")
        print("\t start direct train HOIMash...")

        self.hoi_mash.setGradState(True)

        pbar = tqdm(total=epoch_num)

        scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=500, gamma=0.9)

        for _ in range(epoch_num):
            loss_dict = self.trainStep(
                hand_gt_points,
                object_gt_points,
                1.0,
                1.0,
                scheduler
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

        print("[INFO][Trainer::directTrainMash]")
        print("\t direct train HOIMash finished!")
        print("\t total error: {:.5f}".format(self.error))
        print("\t hand_error: {:.5f}".format(self.hand_error))
        print("\t object_error: {:.5f}".format(self.object_error))

        if self.render:
            if not self.renderErrorMap(hand_gt_points, object_gt_points):
                print("[ERROR][Trainer::directTrainMash]")
                print("\t renderErrorMap failed!")
                return False

        self.closeLog()

        return True

    def escalateTrainMash(self) -> bool:
        lr_decay_rate= 0.2
        gt_points_dtype = self.hoi_mash.dtype

        hand_gt_points = self.hand_gt_points.to(self.hoi_mash.device, dtype=gt_points_dtype)
        object_gt_points = self.object_gt_points.to(self.hoi_mash.device, dtype=gt_points_dtype)
        
        print("[INFO][HOITrainer::escalateTrainMash]")
        print("\t start escalate train HOIMash...")
        
        # first step: training with sh_degree_max = 0
        self.hoi_mash.setGradState(True)
        print(
            "\t degree: mask:",
            self.hoi_mash.mask_degree_max,
            "sh:",
            self.hoi_mash.sh_degree_max,
        )

        print("[INFO][HOITrainer::escalateTrainMash]")
        print("\t start warmUpEpoch...")
        if not self.warmUpEpoch(self.lr, hand_gt_points, object_gt_points, 1.0, 0.5, self.warmup_step_num):
            print("[ERROR][Trainer::escalateTrainMash]")
            print("\t warmUpEpoch failed!")
            return False
        
        print("[INFO][HOITrainer::escalateTrainMash]")
        print("\t start trainEpoch...")
        if not self.trainEpoch(self.lr, hand_gt_points, object_gt_points, 1.0, 0.5):
            print("[ERROR][Trainer::escalateTrainMash]")
            print("\t trainEpoch failed!")
            return False
        
        # second step: training with sh_degree_max = 1
        self.hoi_mash.addShDegreeMax()
        self.lr *= lr_decay_rate
        self.optimizer = Adam(
            [
                self.hoi_mash.hand_params["mask_params"],
                self.hoi_mash.hand_params["sh_params"],
                self.hoi_mash.hand_params["ortho_poses"],
                self.hoi_mash.object_params["mask_params"],
                self.hoi_mash.object_params["sh_params"],
                self.hoi_mash.object_params["ortho_poses"]
            ],
            lr=self.lr
        )

        self.hoi_mash.setGradState(True)
        print(
            "\t degree: mask:",
            self.hoi_mash.mask_degree_max,
            "sh:",
            self.hoi_mash.sh_degree_max,
        )
        
        print("[INFO][HOITrainer::escalateTrainMash]")
        print("\t start warmUpEpoch...")
        if not self.warmUpEpoch(self.lr, hand_gt_points, object_gt_points, 1.0, 0.75, self.warmup_step_num):
            print("[ERROR][Trainer::escalateTrainMash]")
            print("\t warmUpEpoch failed!")
            return False
        
        print("[INFO][HOITrainer::escalateTrainMash]")
        print("\t start trainEpoch...")
        if not self.trainEpoch(self.lr, hand_gt_points, object_gt_points, 1.0, 0.75):
            print("[ERROR][Trainer::escalateTrainMash]")
            print("\t trainEpoch failed!")
            return False

        # second step: training with sh_degree_max = 2
        self.hoi_mash.addShDegreeMax()
        self.lr *= lr_decay_rate
        self.optimizer = Adam(
            [
                self.hoi_mash.hand_params["mask_params"],
                self.hoi_mash.hand_params["sh_params"],
                self.hoi_mash.hand_params["ortho_poses"],
                self.hoi_mash.object_params["mask_params"],
                self.hoi_mash.object_params["sh_params"],
                self.hoi_mash.object_params["ortho_poses"]
            ],
            lr=self.lr
        )

        self.hoi_mash.setGradState(True)
        print(
            "\t degree: mask:",
            self.hoi_mash.mask_degree_max,
            "sh:",
            self.hoi_mash.sh_degree_max,
        )
        
        print("[INFO][HOITrainer::escalateTrainMash]")
        print("\t start warmUpEpoch...")
        if not self.warmUpEpoch(self.lr, hand_gt_points, object_gt_points, 1.0, 1.0, self.warmup_step_num):
            print("[ERROR][Trainer::escalateTrainMash]")
            print("\t warmUpEpoch failed!")
            return False
        
        print("[INFO][HOITrainer::escalateTrainMash]")
        print("\t start trainEpoch...")
        if not self.trainEpoch(self.lr, hand_gt_points, object_gt_points, 1.0, 1.0):
            print("[ERROR][Trainer::escalateTrainMash]")
            print("\t trainEpoch failed!")
            return False
        
        # training finished
        print("[INFO][Trainer::escalateTrainMash]")
        print("\t escalate train HOIMash finished!")
        print("\t total error: {:.5f}".format(self.error))
        print("\t hand_error: {:.5f}".format(self.hand_error))
        print("\t object_error: {:.5f}".format(self.object_error))

        # cmap_gt = self.upSampleCMapGT()
        # print("[INFO][HOITrainer::escalateTrainMash]")
        # if self.use_closed_form_cmap:
        #     print("\t start fitCMap...")
        #     cmap_loss = self.fitCMapLeastSquares(cmap_gt)
        #     cmap_loss = self.fitCMapLeastSquares(cmap_gt)
        # else:
        #     print("\t start trainEpochCMap...")
        #     cmap_loss = self.trainCMap(cmap_gt, 1000, 8e-3)
        # self.optimizer = Adam(
        #     [
        #         self.hoi_mash.hand_params["mask_params"],
        #         self.hoi_mash.hand_params["sh_params"],
        #         self.hoi_mash.hand_params["ortho_poses"],
        #         self.hoi_mash.object_params["mask_params"],
        #         self.hoi_mash.object_params["sh_params"],
        #         self.hoi_mash.object_params["ortho_poses"]
        #     ],
        #     lr=self.lr
        # )

        # self.hand_error = np.sqrt(self.hand_error)
        # self.object_error = np.sqrt(self.object_error)
        
        # self.error = self.hand_error + self.object_error

        # training finished
        print("[INFO][Trainer::escalateTrainMash]")
        print("\t escalate train HOIMash finished!")
        print("\t total error: {:.5f}".format(self.error))
        print("\t hand_error: {:.5f}".format(self.hand_error))
        print("\t object_error: {:.5f}".format(self.object_error))
        # print("\t cmap_loss: {:.5f}".format(cmap_loss))

        self.closeLog()

        return True
