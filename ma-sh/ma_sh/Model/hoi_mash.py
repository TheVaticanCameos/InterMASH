import math
import torch
import numpy as np
import open3d as o3d
from copy import deepcopy
from typing import Tuple

import mash_cpp

from ma_sh.Config.constant import W0, MAX_SH_DEGREE
from ma_sh.Method.data import toNumpy
from ma_sh.Method.pcd import getPointCloud
from ma_sh.Method.render import renderGeometries, renderGeometriesToFile
from ma_sh.Method.rotate import compute_rotation_matrix_from_ortho6d
from sklearn.decomposition import PCA


PI = math.pi
two_PI = 2.0 * PI
CMAP_SH_DEGREE_MAX = 1


class HOIMash(object):
    """
    Class for hand-object interaction mash representation.
    There are hand mash and object mash, which have the same anchor positions, but different other mash parameters.
    """
    def __init__(
        self,
        anchor_num: int = 400,
        mask_degree_max: int = 3,
        sh_degree_max: int = 2,
        sample_phi_num: int = 40,
        sample_theta_num: int = 40,
        dtype: torch.dtype = torch.float32,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        sample_method: str = "fibonacci"
    ) -> None:
        self.anchor_num: int = anchor_num
        self.mask_degree_max: int = mask_degree_max
        self.sh_degree_max: int = sh_degree_max
        self.sample_phi_num: int = sample_phi_num
        self.sample_theta_num: int = sample_theta_num
        self.dtype = dtype
        self.device = device
        self.sample_method = sample_method

        self.positions = torch.zeros(
            [anchor_num, 3], dtype=self.dtype, device=self.device
        )
        self.hand_params = {
            "mask_params": torch.zeros(
                [anchor_num, 2 * mask_degree_max + 1], dtype=self.dtype, device=self.device
                ),
            "sh_params": torch.zeros(
                [anchor_num, (sh_degree_max + 1) ** 2], dtype=self.dtype, device=self.device
                ),
            "ortho_poses": torch.zeros(
                [anchor_num, 6], dtype=self.dtype, device=self.device
                ),
            "cmap_params": torch.zeros(
                [anchor_num, (CMAP_SH_DEGREE_MAX + 1) ** 2], dtype=self.dtype, device=self.device
                )
        }
        self.object_params = {
            "mask_params": torch.zeros(
                [anchor_num, 2 * mask_degree_max + 1], dtype=self.dtype, device=self.device
                ),
            "sh_params": torch.zeros(
                [anchor_num, (sh_degree_max + 1) ** 2], dtype=self.dtype, device=self.device
                ),
            "ortho_poses": torch.zeros(
                [anchor_num, 6], dtype=self.dtype, device=self.device
                ),
            "cmap_params": torch.zeros(
                [anchor_num, (CMAP_SH_DEGREE_MAX + 1) ** 2], dtype=self.dtype, device=self.device
                )
        }

        self.updatePreLoadDatas()
        
        return

    def addShDegreeMax(self) -> bool:
        old_sh_degree_max = self.sh_degree_max
        self.sh_degree_max += 1
        if self.sh_degree_max > MAX_SH_DEGREE:
            print("[ERROR][HOIMash::addShDegreeMax]")
            print("\t sh_degree_max exceeds MAX_SH_DEGREE!")
            return False
        old_hand_sh_params = self.hand_params["sh_params"]
        old_object_sh_params = self.object_params["sh_params"]
        self.hand_params["sh_params"] = torch.zeros(
            [self.anchor_num, (self.sh_degree_max + 1) ** 2], dtype=self.dtype, device=self.device
            )
        self.object_params["sh_params"] = torch.zeros(
            [self.anchor_num, (self.sh_degree_max + 1) ** 2], dtype=self.dtype, device=self.device
            )
        
        self.hand_params["sh_params"][:, : (old_sh_degree_max + 1) ** 2] = old_hand_sh_params
        self.object_params["sh_params"][:, : (old_sh_degree_max + 1) ** 2] = old_object_sh_params
        
        self.hand_params["mask_params"] = self.hand_params["mask_params"].detach()
        self.hand_params["sh_params"] = self.hand_params["sh_params"].detach()
        self.hand_params["ortho_poses"] = self.hand_params["ortho_poses"].detach()
        
        self.object_params["mask_params"] = self.object_params["mask_params"].detach()
        self.object_params["sh_params"] = self.object_params["sh_params"].detach()
        self.object_params["ortho_poses"] = self.object_params["ortho_poses"].detach()
        
        return True

    def updatePreLoadDatas(self) -> bool:
        sample_phis = torch.linspace(
            two_PI / self.sample_phi_num,
            two_PI,
            self.sample_phi_num,
            dtype=self.dtype,
            device=self.device,
        )
        sample_thetas = torch.linspace(
            PI / self.sample_theta_num,
            PI,
            self.sample_theta_num,
            dtype=self.dtype,
            device=self.device,
        )

        phi_grid, theta_grid = torch.meshgrid(sample_phis, sample_thetas, indexing="ij")
        sample_phi_theta_mat = torch.stack([phi_grid, theta_grid], dim=-1)

        self.expanded_sample_phi_theta_mat = sample_phi_theta_mat.unsqueeze(0).expand(
            self.anchor_num, -1, -1, -1
        )

        if self.mask_degree_max > 0:
            degrees = torch.arange(
                1, self.mask_degree_max + 1, dtype=self.dtype, device=self.device
            )
            angles = degrees.unsqueeze(1) * sample_phis.unsqueeze(0)
            self.cos_terms = torch.cos(angles)
            self.sin_terms = torch.sin(angles)
        return True

    def setGradState(self, need_grad: bool) -> bool:
        self.positions.requires_grad_(need_grad)
        for param in self.hand_params.values():
            param.requires_grad_(need_grad)
        for param in self.object_params.values():
            param.requires_grad_(need_grad)
        return True

    def editGrads(self, edit_fn) -> bool:
        if self.positions.grad is not None:
            self.positions.grad = edit_fn(self.positions.grad)

        if self.hand_params["mask_params"].grad is not None:
            self.hand_params["mask_params"].grad = edit_fn(self.hand_params["mask_params"].grad)
        if self.hand_params["sh_params"].grad is not None:
            self.hand_params["sh_params"].grad = edit_fn(self.hand_params["sh_params"].grad)
        if self.hand_params["ortho_poses"].grad is not None:
            self.hand_params["ortho_poses"].grad = edit_fn(self.hand_params["ortho_poses"].grad)
        
        if self.object_params["mask_params"].grad is not None:
            self.object_params["mask_params"].grad = edit_fn(self.object_params["mask_params"].grad)
        if self.object_params["sh_params"].grad is not None:
            self.object_params["sh_params"].grad = edit_fn(self.object_params["sh_params"].grad)
        if self.object_params["ortho_poses"].grad is not None:
            self.object_params["ortho_poses"].grad = edit_fn(self.object_params["ortho_poses"].grad)
        
        return True

    def clearNanGrads(self) -> bool:
        def edit_fn(grad: torch.Tensor) -> torch.Tensor:
            nan_mask = torch.isnan(grad)
            grad[nan_mask] = 0
            return grad
        
        return self.editGrads(edit_fn)

    def nearestInitOrthoPoses(
        self, 
        hand_gt_points: torch.Tensor, 
        object_gt_points: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hand_pcd = o3d.geometry.PointCloud()
        hand_pcd.points = o3d.utility.Vector3dVector(hand_gt_points.cpu().numpy())
        hand_pcd_tree = o3d.geometry.KDTreeFlann(hand_pcd)
        
        hand_nearest_points = torch.zeros_like(self.positions)
        hand_nearest_dists = torch.zeros(size=[self.positions.shape[0],], dtype=self.positions.dtype, device=self.positions.device)
        for i, point in enumerate(self.positions.cpu().numpy()):
            [_, idx, dist] = hand_pcd_tree.search_knn_vector_3d(point, 1)
            nearest_point = np.asarray(hand_pcd.points)[idx[0]]
            hand_nearest_points[i] = torch.tensor(nearest_point, dtype=self.dtype, device=self.device)
            hand_nearest_dists[i] = dist[0]
        
        object_pcd = o3d.geometry.PointCloud()
        object_pcd.points = o3d.utility.Vector3dVector(object_gt_points.cpu().numpy())
        object_pcd_tree = o3d.geometry.KDTreeFlann(object_pcd)

        object_nearest_points = torch.zeros_like(self.positions)
        object_nearest_dists = torch.zeros(size=[self.positions.shape[0],], dtype=self.positions.dtype, device=self.positions.device)
        for i, point in enumerate(self.positions.cpu().numpy()):
            [_, idx, dist] = object_pcd_tree.search_knn_vector_3d(point, 1)
            nearest_point = np.asarray(object_pcd.points)[idx[0]]
            object_nearest_points[i] = torch.tensor(nearest_point, dtype=self.dtype, device=self.device)
            object_nearest_dists[i] = dist[0]

        hand_face_forward_vectors = hand_nearest_points - self.positions
        hand_rotate_vectors = mash_cpp.toRotateVectorsByFaceForwardVectors(hand_face_forward_vectors)
        hand_rotate_matrixs = mash_cpp.toRotateMatrixs(hand_rotate_vectors)
        hand_ortho_poses = hand_rotate_matrixs[:, :2, :].view(-1, 6)
        self.hand_params["ortho_poses"].data = (
            hand_ortho_poses.detach().clone().type(self.dtype).to(self.device)
        )
        
        object_face_forward_vectors = object_nearest_points - self.positions
        object_rotate_vectors = mash_cpp.toRotateVectorsByFaceForwardVectors(object_face_forward_vectors)
        object_rotate_matrixs = mash_cpp.toRotateMatrixs(object_rotate_vectors)
        object_ortho_poses = object_rotate_matrixs[:, :2, :].view(-1, 6)
        self.object_params["ortho_poses"].data = (
            object_ortho_poses.detach().clone().type(self.dtype).to(self.device)
        )

        return (hand_nearest_dists, object_nearest_dists)

    def centerInitOrthoPoses(
        self, 
        hand_gt_points: torch.Tensor,   # [P, N, 3]
        object_gt_points: torch.Tensor  # [P, N, 3]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hand_patch_center = hand_gt_points.mean(dim=1).to(self.device)
        object_patch_center = object_gt_points.mean(dim=1).to(self.device)

        hand_face_forward_vectors = hand_patch_center - self.positions
        hand_rotate_vectors = mash_cpp.toRotateVectorsByFaceForwardVectors(hand_face_forward_vectors)
        hand_rotate_matrixs = mash_cpp.toRotateMatrixs(hand_rotate_vectors)
        hand_ortho_poses = hand_rotate_matrixs[:, :2, :].view(-1, 6)
        self.hand_params["ortho_poses"].data = (
            hand_ortho_poses.detach().clone().type(self.dtype).to(self.device)
        )
        
        object_face_forward_vectors = object_patch_center - self.positions
        object_rotate_vectors = mash_cpp.toRotateVectorsByFaceForwardVectors(object_face_forward_vectors)
        object_rotate_matrixs = mash_cpp.toRotateMatrixs(object_rotate_vectors)
        object_ortho_poses = object_rotate_matrixs[:, :2, :].view(-1, 6)
        self.object_params["ortho_poses"].data = (
            object_ortho_poses.detach().clone().type(self.dtype).to(self.device)
        )

        hand_dists = torch.norm(hand_patch_center - self.positions, dim=1)
        object_dists = torch.norm(object_patch_center - self.positions, dim=1)

        return hand_dists, object_dists

    def randomInit(self) -> bool:
        self.positions.data = torch.randn_like(self.positions)
        self.hand_params["sh_params"].data = torch.randn_like(self.hand_params["sh_params"])
        self.object_params["sh_params"].data = torch.randn_like(self.object_params["sh_params"])
        self.hand_params["ortho_poses"].data = torch.randn_like(self.hand_params["ortho_poses"])
        self.object_params["ortho_poses"].data = torch.randn_like(self.object_params["ortho_poses"])
        self.hand_params["mask_params"].data = torch.randn_like(self.hand_params["mask_params"])
        self.object_params["mask_params"].data = torch.randn_like(self.object_params["mask_params"])
        return True

    def voxelInit(
        self, 
        hand_gt_points: torch.Tensor, 
        object_gt_points: torch.Tensor, 
        resolution: int = 32,
        scale: int = 4
    ) -> bool:
        hand_gt_points = hand_gt_points.reshape(-1, 3)
        object_gt_points = object_gt_points.reshape(-1, 3)
        
        min_coords, _ = torch.min(object_gt_points, dim=0)
        max_coords, _ = torch.max(object_gt_points, dim=0)

        center = (min_coords + max_coords) / 2
        half_size = (max_coords - min_coords) / 2
        new_half_size = half_size * scale
        new_min_coords = center - new_half_size
        new_max_coords = center + new_half_size

        x_coords = torch.linspace(new_min_coords[0], new_max_coords[0], resolution, dtype=self.dtype, device=self.device)
        y_coords = torch.linspace(new_min_coords[1], new_max_coords[1], resolution, dtype=self.dtype, device=self.device)
        z_coords = torch.linspace(new_min_coords[2], new_max_coords[2], resolution, dtype=self.dtype, device=self.device)

        x_grid, y_grid, z_grid = torch.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
        all_grid_points = torch.stack([x_grid.flatten(), y_grid.flatten(), z_grid.flatten()], dim=-1)

        assert all_grid_points.shape[0] >= self.anchor_num, f"all_grid_points.shape[0] = {all_grid_points.shape[0]}, self.anchor_num = {self.anchor_num}"

        indices = torch.randperm(all_grid_points.shape[0], device=self.device)[:self.anchor_num]
        selected_points = all_grid_points[indices]
        
        self.positions.data = selected_points

        hand_nearest_dists, object_nearest_dists = self.nearestInitOrthoPoses(hand_gt_points, object_gt_points)

        self.hand_params["sh_params"].data[:, 0] = hand_nearest_dists / W0
        self.object_params["sh_params"].data[:, 0] = object_nearest_dists / W0
        
        return True

    def middlePlaneInitOld(
        self,
        hand_gt_points: torch.Tensor,
        object_gt_points: torch.Tensor,
        r: float = 0.3
    ) -> bool:
        hand_gt_points = hand_gt_points.reshape(-1, 3)
        object_gt_points = object_gt_points.reshape(-1, 3)
        
        radius = torch.tensor([r], device=self.device)

        hand_center = hand_gt_points.mean(dim=0)
        object_center = object_gt_points.mean(dim=0)
        center = (hand_center + object_center) / 2
        center = center.to(self.device)

        normal = object_center - hand_center
        normal = normal / torch.norm(normal)
        normal= normal.to(self.device)

        arbitrary_vec = torch.tensor([1.0, 0.0, 0.0], dtype=self.dtype, device=self.device)
        
        if torch.allclose(normal, arbitrary_vec, atol=1e-6):
            arbitrary_vec = torch.tensor([0.0, 1.0, 0.0], dtype=self.dtype, device=self.device)

        basis1 = torch.linalg.cross(normal, arbitrary_vec)
        basis1 = basis1 / torch.norm(basis1)
        
        basis2 = torch.linalg.cross(normal, basis1)

        angles = torch.rand(self.anchor_num, dtype=self.dtype, device=self.device) * 2 * torch.pi
        radii = torch.sqrt(torch.rand(self.anchor_num, dtype=self.dtype, device=self.device)) * radius

        points_in_plane = (
            center.unsqueeze(0)
            + (basis1.unsqueeze(0) * torch.cos(angles).unsqueeze(-1) * radii.unsqueeze(-1))
            + (basis2.unsqueeze(0) * torch.sin(angles).unsqueeze(-1) * radii.unsqueeze(-1))
        )

        self.positions.data = points_in_plane

        hand_nearest_dists, object_nearest_dists = self.nearestInitOrthoPoses(hand_gt_points, object_gt_points)

        self.hand_params["sh_params"].data[:, 0] = hand_nearest_dists / W0
        self.object_params["sh_params"].data[:, 0] = object_nearest_dists / W0

        return True

    def middlePlaneInit(
        self,
        hand_gt_points: torch.Tensor,
        object_gt_points: torch.Tensor,
        r: float = 0.2,
        offset: float = 0.03
    ) -> bool:
        # Sample uniformly from a disk of radius r in the xz plane
        angles = torch.rand(self.anchor_num, dtype=self.dtype, device=self.device) * 2 * torch.pi
        radii = torch.sqrt(torch.rand(self.anchor_num, dtype=self.dtype, device=self.device)) * r

        # Compute the coordinate offsets of sampled points in the xz plane
        x_offset = torch.cos(angles) * radii
        z_offset = torch.sin(angles) * radii

        # Construct sampled point coordinates with the same y coordinate as the center
        points_in_disk = torch.stack([
            x_offset,
            offset * torch.ones_like(x_offset),
            z_offset
        ], dim=-1)

        self.positions.data = points_in_disk

        return True

    def middlePlaneFixInit(
        self,
        hand_gt_points: torch.Tensor,
        object_gt_points: torch.Tensor,
        r: float = 0.2,
        offset: float = 0.03
    ) -> bool:
        hand_gt_points = hand_gt_points.reshape(-1, 3)
        object_gt_points = object_gt_points.reshape(-1, 3)

        N = self.anchor_num
        i = torch.arange(N, dtype=self.dtype, device=self.device)
        golden_angle = torch.tensor(math.pi * (3.0 - math.sqrt(5.0)), dtype=self.dtype, device=self.device)
        angles = (i * golden_angle) % (2 * torch.pi)
        radii = torch.sqrt((i + 0.5) / float(N)) * r

        # Compute the coordinate offsets of sampled points in the xz plane
        x_offset = torch.cos(angles) * radii
        z_offset = torch.sin(angles) * radii

        # Construct sampled point coordinates with the same y coordinate as the center
        points_in_disk = torch.stack([
            x_offset,
            offset * torch.ones_like(x_offset),
            z_offset
        ], dim=-1)

        self.positions.data = points_in_disk

        return True
    
    def sphereFixInit(
        self,
        r: float = 0.2
    ) -> bool:
        N = self.anchor_num
        i = torch.arange(N, dtype=self.dtype, device=self.device)
        golden_angle = torch.tensor(math.pi * (3.0 - math.sqrt(5.0)), dtype=self.dtype, device=self.device)
        phis = (i * golden_angle) % (2 * torch.pi)
        ys = 1.0 - 2.0 * ((i + 0.5) / float(N))
        r_xy = torch.sqrt(torch.clamp(1.0 - ys.square(), min=0.0)) * r
        xs = torch.cos(phis) * r_xy
        zs = torch.sin(phis) * r_xy
        ys = ys * r
        points_on_sphere = torch.stack([xs, ys, zs], dim=-1)
        self.positions = points_on_sphere
        return True

    def shParamInit(
        self,
        hand_gt_points: torch.Tensor,   # [P, N, 3]
        object_gt_points: torch.Tensor  # [P, N, 3]
    ) -> bool:
        hand_dists, object_dists = self.centerInitOrthoPoses(hand_gt_points, object_gt_points)

        self.hand_params["sh_params"].data[:, 0] = hand_dists
        self.object_params["sh_params"].data[:, 0] = object_dists

        return True

    def toMaskThetas(self, target: str) -> torch.Tensor:
        if target == "hand":
            mask_thetas = self.hand_params["mask_params"][:, 0:1].expand(-1, self.sample_phi_num)   # [P, S]
            if self.mask_degree_max > 0:
                cos_params = self.hand_params["mask_params"][:, 1::2]   # [P, 3]
                sin_params = self.hand_params["mask_params"][:, 2::2]   # [P, 3]
                mask_thetas = torch.addmm(mask_thetas, cos_params, self.cos_terms)
                mask_thetas = torch.addmm(mask_thetas, sin_params, self.sin_terms)
            
            return torch.sigmoid(mask_thetas)
                
        elif target == "object":
            mask_thetas = self.object_params["mask_params"][:, 0:1].expand(-1, self.sample_phi_num)
            if self.mask_degree_max > 0:
                cos_params = self.object_params["mask_params"][:, 1::2]
                sin_params = self.object_params["mask_params"][:, 2::2]
                mask_thetas = torch.addmm(mask_thetas, cos_params, self.cos_terms)
                mask_thetas = torch.addmm(mask_thetas, sin_params, self.sin_terms)
            
            return torch.sigmoid(mask_thetas)

        else:
            raise ValueError(f"target {target} is not supported")

    def toWeightedSamplePhiThetaMat(self, target: str) -> torch.Tensor:
        theta_weights = self.toMaskThetas(target)   # [P, S]

        phis = self.expanded_sample_phi_theta_mat[..., 0]   # [P, S, S]
        thetas = self.expanded_sample_phi_theta_mat[..., 1] # [P, S, S]

        weighted_thetas = thetas * theta_weights.unsqueeze(-1)  # [P, S, S]

        weighted_sample_phi_theta_mat = torch.stack([phis, weighted_thetas], dim=-1)

        return weighted_sample_phi_theta_mat

    def toDirections(self, phis: torch.Tensor, thetas: torch.Tensor) -> torch.Tensor:
        sin_theta, cos_theta = torch.sin(thetas), torch.cos(thetas)
        sin_phi, cos_phi = torch.sin(phis), torch.cos(phis)

        x = sin_theta * cos_phi
        y = sin_theta * sin_phi
        z = cos_theta

        return torch.stack([x, y, z], dim=-1)

    def toDistances(self, phis: torch.Tensor, thetas: torch.Tensor, target: str) -> torch.Tensor:
        if target == "hand":
            base_values = mash_cpp.toSHBaseValues(phis, thetas, self.sh_degree_max) # [1, P, S, S]
            base_values = base_values.transpose(1, 0)   # [P, 1, S, S]

            if base_values.dim() == 2:
                sample_distances = torch.sum(self.hand_params["sh_params"] * base_values, dim=1)
            else:
                sample_distances = torch.einsum(
                    "bn,bn...->b...", self.hand_params["sh_params"], base_values
                )   # [P, S, S]

            return sample_distances
        
        elif target == "object":
            base_values = mash_cpp.toSHBaseValues(phis, thetas, self.sh_degree_max)
            base_values = base_values.transpose(1, 0)

            if base_values.dim() == 2:
                sample_distances = torch.sum(self.object_params["sh_params"] * base_values, dim=1)
            else:
                sample_distances = torch.einsum(
                    "bn,bn...->b...", self.object_params["sh_params"], base_values
                )

            return sample_distances

        else:
            raise ValueError(f"target {target} is not supported")

    def toSampleMoveVectors(self, target: str) -> torch.Tensor:
        weighted_sample_phi_theta_mat = self.toWeightedSamplePhiThetaMat(target)
        
        phis, thetas = weighted_sample_phi_theta_mat.split(1, dim=-1)
        phis = phis.squeeze(-1)     # [P, S, S]
        thetas = thetas.squeeze(-1) # [P, S, S]

        sample_directions = self.toDirections(phis, thetas)         # [P, S, S, 3]
        sample_distances = self.toDistances(phis, thetas, target)   # [P, S, S]

        sample_move_vectors = sample_directions * sample_distances.unsqueeze(-1)
        return sample_move_vectors

    def toInvSampleMoveVectors(self, target: str) -> torch.Tensor:
        sample_move_vectors = self.toSampleMoveVectors(target).view(self.anchor_num, -1, 3)
        
        if target == "hand":
            inv_radius = W0 * self.hand_params["sh_params"][:, 0]
        elif target == "object":
            inv_radius = W0 * self.object_params["sh_params"][:, 0]
        else:
            raise ValueError(f"target {target} is not supported")

        inv_centers = torch.zeros(
            self.anchor_num,
            1,
            3,
            dtype=sample_move_vectors.dtype,
            device=sample_move_vectors.device,
        )
        inv_centers[:, 0, 2] = -inv_radius

        in_inv_points = sample_move_vectors - inv_centers

        in_inv_point_norms_sq = torch.sum(in_inv_points.square(), dim=2, keepdim=True)

        in_inv_point_weights = torch.rsqrt(in_inv_point_norms_sq)

        in_inv_point_directions = in_inv_points * in_inv_point_weights

        inv_radius_sq_4 = (4.0 * inv_radius.square()).view(-1, 1, 1)

        in_inv_point_lengths = inv_radius_sq_4 * in_inv_point_weights

        inv_points = torch.addcmul(
            inv_centers, in_inv_point_lengths, in_inv_point_directions
        )

        return inv_points

    def toSamplePoints(self, target: str, method: str = "casual") -> torch.Tensor:
        if method == "casual":
            sample_move_vectors = self.toInvSampleMoveVectors(target)
        elif method == "fibonacci":
            sample_num = int(self.sample_phi_num * self.sample_theta_num)
            i = torch.arange(sample_num, dtype=self.dtype, device=self.device)
            golden_angle = torch.tensor(
                math.pi * (3.0 - math.sqrt(5.0)), dtype=self.dtype, device=self.device
            )
            phis = torch.remainder(i * golden_angle, two_PI).unsqueeze(0).expand(
                self.anchor_num, -1
            )

            if target == "hand":
                mask_params = self.hand_params["mask_params"]
            elif target == "object":
                mask_params = self.object_params["mask_params"]
            else:
                raise ValueError(f"target {target} is not supported")

            mask_thetas = mask_params[:, 0:1].expand(-1, sample_num)
            if self.mask_degree_max > 0:
                degrees = torch.arange(
                    1, self.mask_degree_max + 1, dtype=self.dtype, device=self.device
                )
                angles = degrees.unsqueeze(1) * phis[0].unsqueeze(0)
                cos_terms = torch.cos(angles)
                sin_terms = torch.sin(angles)
                cos_params = mask_params[:, 1::2]
                sin_params = mask_params[:, 2::2]
                mask_thetas = torch.addmm(mask_thetas, cos_params, cos_terms)
                mask_thetas = torch.addmm(mask_thetas, sin_params, sin_terms)
            theta_weights = torch.sigmoid(mask_thetas).clamp(0.0, 1.0)

            thetas_max = PI * theta_weights
            cos_thetas_max = torch.cos(thetas_max)
            u = ((i + 0.5) / float(sample_num)).unsqueeze(0).expand(self.anchor_num, -1)
            cos_thetas = 1.0 - u * (1.0 - cos_thetas_max)
            cos_thetas = cos_thetas.clamp(-1.0, 1.0)
            thetas = torch.arccos(cos_thetas)

            sample_directions = self.toDirections(phis, thetas)
            sample_distances = self.toDistances(phis, thetas, target)
            sample_move_vectors = sample_directions * sample_distances.unsqueeze(-1)

            if target == "hand":
                inv_radius = W0 * self.hand_params["sh_params"][:, 0]
            elif target == "object":
                inv_radius = W0 * self.object_params["sh_params"][:, 0]
            else:
                raise ValueError(f"target {target} is not supported")

            inv_centers = torch.zeros(
                self.anchor_num,
                1,
                3,
                dtype=sample_move_vectors.dtype,
                device=sample_move_vectors.device,
            )
            inv_centers[:, 0, 2] = -inv_radius

            in_inv_points = sample_move_vectors - inv_centers
            in_inv_point_norms_sq = torch.sum(in_inv_points.square(), dim=2, keepdim=True)
            in_inv_point_weights = torch.rsqrt(in_inv_point_norms_sq)
            in_inv_point_directions = in_inv_points * in_inv_point_weights
            inv_radius_sq_4 = (4.0 * inv_radius.square()).view(-1, 1, 1)
            in_inv_point_lengths = inv_radius_sq_4 * in_inv_point_weights

            sample_move_vectors = torch.addcmul(
                inv_centers, in_inv_point_lengths, in_inv_point_directions
            )
        else:
            raise ValueError(f"method {method} is not supported")

        if target == "hand":
            rotate_mats = compute_rotation_matrix_from_ortho6d(self.hand_params["ortho_poses"])     # [P, 3, 3]
        elif target == "object":
            rotate_mats = compute_rotation_matrix_from_ortho6d(self.object_params["ortho_poses"])   # [P, 3, 3]
        else:
            raise ValueError(f"target {target} is not supported")
        
        rotated_sample_move_vectors = torch.einsum(
            "b...i,bij->b...j", sample_move_vectors, rotate_mats
        )   # [P, N, 3]

        positions_expanded = self.positions.view(
            self.anchor_num, *((1,) * (sample_move_vectors.dim() - 2)), 3
        )   # [P, 1, 3]

        sample_points = positions_expanded + rotated_sample_move_vectors

        return sample_points    # [P, N, 3]

    def toCMap(self, target: str, method: str = "casual") -> torch.Tensor:
        if method == "casual":
            weighted_sample_phi_theta_mat = self.toWeightedSamplePhiThetaMat(target)
            phis, thetas = weighted_sample_phi_theta_mat.split(1, dim=-1)
            phis = phis.squeeze(-1)
            thetas = thetas.squeeze(-1)
        elif method == "fibonacci":
            sample_num = int(self.sample_phi_num * self.sample_theta_num)
            i = torch.arange(sample_num, dtype=self.dtype, device=self.device)
            golden_angle = torch.tensor(
                math.pi * (3.0 - math.sqrt(5.0)), dtype=self.dtype, device=self.device
            )
            phis = torch.remainder(i * golden_angle, two_PI).unsqueeze(0).expand(
                self.anchor_num, -1
            )

            if target == "hand":
                mask_params = self.hand_params["mask_params"]
            elif target == "object":
                mask_params = self.object_params["mask_params"]
            else:
                raise ValueError(f"target {target} is not supported")

            mask_thetas = mask_params[:, 0:1].expand(-1, sample_num)
            if self.mask_degree_max > 0:
                degrees = torch.arange(
                    1, self.mask_degree_max + 1, dtype=self.dtype, device=self.device
                )
                angles = degrees.unsqueeze(1) * phis[0].unsqueeze(0)
                cos_terms = torch.cos(angles)
                sin_terms = torch.sin(angles)
                cos_params = mask_params[:, 1::2]
                sin_params = mask_params[:, 2::2]
                mask_thetas = torch.addmm(mask_thetas, cos_params, cos_terms)
                mask_thetas = torch.addmm(mask_thetas, sin_params, sin_terms)
            theta_weights = torch.sigmoid(mask_thetas).clamp(0.0, 1.0)

            thetas_max = PI * theta_weights
            cos_thetas_max = torch.cos(thetas_max)
            u = ((i + 0.5) / float(sample_num)).unsqueeze(0).expand(self.anchor_num, -1)
            cos_thetas = 1.0 - u * (1.0 - cos_thetas_max)
            cos_thetas = cos_thetas.clamp(-1.0, 1.0)
            thetas = torch.arccos(cos_thetas)
        else:
            raise ValueError(f"method {method} is not supported")

        base_values = mash_cpp.toSHBaseValues(phis, thetas, CMAP_SH_DEGREE_MAX)
        base_values = base_values.transpose(1, 0)
        base_values = base_values.detach()

        if base_values.dim() == 2:
            if target == "hand":
                cmap = torch.sum(self.hand_params["cmap_params"] * base_values, dim=1)
            elif target == "object":
                cmap = torch.sum(self.object_params["cmap_params"] * base_values, dim=1)
            else:
                raise ValueError(f"target {target} is not supported")
        else:
            if target == "hand":
                cmap = torch.einsum(
                    "bn,bn...->b...", self.hand_params["cmap_params"], base_values
                )
            elif target == "object":
                cmap = torch.einsum(
                    "bn,bn...->b...", self.object_params["cmap_params"], base_values
                )
            else:
                raise ValueError(f"target {target} is not supported")

        if method == "fibonacci" and cmap.dim() == 2 and cmap.shape[1] == self.sample_phi_num * self.sample_theta_num:
            cmap = cmap.view(self.anchor_num, self.sample_phi_num, self.sample_theta_num)

        return cmap

    def saveParamsFile(self, save_params_file_path: str) -> bool:
        params_dict = self.toParamsDict()
        np.save(save_params_file_path, params_dict, allow_pickle=True)
        return True

    def toParamsDict(self) -> dict:
        params_dict = {
            "positions": toNumpy(self.positions),
            "hand_params": {
                "mask_params": toNumpy(self.hand_params["mask_params"]),
                "sh_params": toNumpy(self.hand_params["sh_params"]),
                "ortho_poses": toNumpy(self.hand_params["ortho_poses"]),
                "cmap_params": toNumpy(self.hand_params["cmap_params"]),
            },
            "object_params": {
                "mask_params": toNumpy(self.object_params["mask_params"]),
                "sh_params": toNumpy(self.object_params["sh_params"]),
                "ortho_poses": toNumpy(self.object_params["ortho_poses"]),
                "cmap_params": toNumpy(self.object_params["cmap_params"]),
            }
        }
        return params_dict

    @classmethod
    def fromParamsFile(
        cls,
        params_file_path: str,
        sample_phi_num: int = 10,
        sample_theta_num: int = 10,
        dtype=torch.float32,
        device: str = "cpu",
    ):
        params_dict = np.load(params_file_path, allow_pickle=True).item()

        return cls.fromParamsDict(
            params_dict,
            sample_phi_num,
            sample_theta_num,
            dtype,
            device
        )

    @classmethod
    def fromParamsDict(
        cls,
        params_dict: dict,
        sample_phi_num: int = 10,
        sample_theta_num: int = 10,
        dtype=torch.float32,
        device: str = "cpu"
    ):
        anchor_num = params_dict["positions"].shape[0]
        mask_degree_max = int((params_dict["hand_params"]["mask_params"].shape[1] - 1) / 2)
        sh_degree_max = int(math.sqrt(params_dict["hand_params"]["sh_params"].shape[1] - 1))

        hoi_mash= cls(
            anchor_num,
            mask_degree_max,
            sh_degree_max,
            sample_phi_num,
            sample_theta_num,
            dtype,
            device
        )

        hoi_mash.loadParamsDict(params_dict)

        return hoi_mash

    def loadParamsDict(self, params_dict: dict) -> bool:
        self.positions = torch.from_numpy(params_dict["positions"]).to(self.dtype).to(self.device)

        self.hand_params["mask_params"] = torch.from_numpy(params_dict["hand_params"]["mask_params"]).to(self.dtype).to(self.device)
        self.hand_params["sh_params"] = torch.from_numpy(params_dict["hand_params"]["sh_params"]).to(self.dtype).to(self.device)
        self.hand_params["ortho_poses"] = torch.from_numpy(params_dict["hand_params"]["ortho_poses"]).to(self.dtype).to(self.device)
        if "cmap_params" in params_dict["hand_params"]:
            self.hand_params["cmap_params"] = torch.from_numpy(params_dict["hand_params"]["cmap_params"]).to(self.dtype).to(self.device)
        
        if "object_params" in params_dict.keys():
            self.object_params["mask_params"] = torch.from_numpy(params_dict["object_params"]["mask_params"]).to(self.dtype).to(self.device)
            self.object_params["sh_params"] = torch.from_numpy(params_dict["object_params"]["sh_params"]).to(self.dtype).to(self.device)
            self.object_params["ortho_poses"] = torch.from_numpy(params_dict["object_params"]["ortho_poses"]).to(self.dtype).to(self.device)
            if "cmap_params" in params_dict["object_params"]:
                self.object_params["cmap_params"] = torch.from_numpy(params_dict["object_params"]["cmap_params"]).to(self.dtype).to(self.device)
        
        return True

    def loadParamsDictLegacy(self, params_dict: dict) -> bool:
        self.positions = torch.from_numpy(params_dict["positions"]).to(self.dtype).to(self.device)
        self.hand_params["mask_params"] = torch.from_numpy(params_dict["hand_params"]["mask_params"]).to(self.dtype).to(self.device)
        self.hand_params["sh_params"] = torch.from_numpy(params_dict["hand_params"]["sh_params"]).to(self.dtype).to(self.device)
        self.hand_params["ortho_poses"] = torch.from_numpy(params_dict["hand_params"]["ortho_poses"]).to(self.dtype).to(self.device)
        self.object_params["mask_params"] = torch.from_numpy(params_dict["object_params"]["mask_params"]).to(self.dtype).to(self.device)
        self.object_params["sh_params"] = torch.from_numpy(params_dict["object_params"]["sh_params"]).to(self.dtype).to(self.device)
        self.object_params["ortho_poses"] = torch.from_numpy(params_dict["object_params"]["ortho_poses"]).to(self.dtype).to(self.device)
        if "cmap_params" in params_dict:
            self.cmap_params["sh_params"] = torch.from_numpy(params_dict["cmap_params"]["sh_params"]).to(self.dtype).to(self.device)
        return True

    def renderSamplePoints(
        self, 
        show_positions: bool = False, 
        save_path: str = None, 
        show_hand_line: bool = False, 
        show_object_line: bool = False,
        show_cmap: bool = False
    ) -> bool:
        hand_sample_pts = toNumpy(self.toSamplePoints("hand", self.sample_method))
        hand_pt_centers = np.mean(hand_sample_pts, axis=1)
        hand_sample_pts = hand_sample_pts.reshape(-1, 3)
        
        object_sample_pts = toNumpy(self.toSamplePoints("object", self.sample_method))
        object_pt_centers = np.mean(object_sample_pts, axis=1)
        object_sample_pts = object_sample_pts.reshape(-1, 3)
        if show_cmap:
            import matplotlib.cm as cm
            has_hand_cmap = isinstance(getattr(self, "hand_params", None), dict) and "cmap_params" in self.hand_params and self.hand_params["cmap_params"] is not None
            has_object_cmap = isinstance(getattr(self, "object_params", None), dict) and "cmap_params" in self.object_params and self.object_params["cmap_params"] is not None

            def to_colors(cmap_values: np.ndarray) -> np.ndarray:
                cmap_min = cmap_values.min() if cmap_values.size > 0 else 0.0
                cmap_max = cmap_values.max() if cmap_values.size > 0 else 1.0
                if cmap_max > cmap_min:
                    cmap_norm = (cmap_values - cmap_min) / (cmap_max - cmap_min)
                else:
                    cmap_norm = np.zeros_like(cmap_values)
                return cm.get_cmap("coolwarm")(cmap_norm)[:, :3]

            hand_sample_pcd = getPointCloud(hand_sample_pts)
            if has_hand_cmap:
                try:
                    hand_cmap_values = toNumpy(self.toCMap("hand", self.sample_method)).reshape(-1)
                    hand_colors = to_colors(hand_cmap_values)
                    if hand_colors.shape[0] == hand_sample_pts.shape[0]:
                        hand_sample_pcd.colors = o3d.utility.Vector3dVector(hand_colors.astype(np.float64))
                    else:
                        hand_sample_pcd.paint_uniform_color([1.0, 0.0, 0.0])
                except Exception:
                    hand_sample_pcd.paint_uniform_color([1.0, 0.0, 0.0])
            else:
                hand_sample_pcd.paint_uniform_color([1.0, 0.0, 0.0])

            object_sample_pcd = getPointCloud(object_sample_pts)
            if has_object_cmap:
                try:
                    object_cmap_values = toNumpy(self.toCMap("object", self.sample_method)).reshape(-1)
                    object_colors = to_colors(object_cmap_values)
                    if object_colors.shape[0] == object_sample_pts.shape[0]:
                        object_sample_pcd.colors = o3d.utility.Vector3dVector(object_colors.astype(np.float64))
                    else:
                        object_sample_pcd.paint_uniform_color([0.0, 1.0, 0.0])
                except Exception:
                    object_sample_pcd.paint_uniform_color([0.0, 1.0, 0.0])
            else:
                object_sample_pcd.paint_uniform_color([0.0, 1.0, 0.0])
        else:
            hand_sample_pcd = getPointCloud(hand_sample_pts)
            hand_sample_pcd.paint_uniform_color([1., 0., 0.])
            object_sample_pcd = getPointCloud(object_sample_pts)
            object_sample_pcd.paint_uniform_color([0., 1., 0.])
        
        geometries = [hand_sample_pcd, object_sample_pcd]
        
        positions_pts = toNumpy(self.positions)
        
        if show_hand_line:
            line_set_hand = o3d.geometry.LineSet()
            points = np.vstack([hand_pt_centers, positions_pts])
            line_set_hand.points = o3d.utility.Vector3dVector(points)
            lines = np.array([[i, i + len(hand_pt_centers)] for i in range(len(hand_pt_centers))])
            line_set_hand.lines = o3d.utility.Vector2iVector(lines)
            line_set_hand.paint_uniform_color([0., 0., 0.3])
            
            geometries.append(line_set_hand)

            hand_centers_pcd = getPointCloud(hand_pt_centers)
            hand_centers_pcd.paint_uniform_color([0., 0., 1.])
        
        if show_object_line:
            line_set_object = o3d.geometry.LineSet()
            points = np.vstack([object_pt_centers, positions_pts])
            line_set_object.points = o3d.utility.Vector3dVector(points)
            lines = np.array([[i, i + len(object_pt_centers)] for i in range(len(object_pt_centers))])
            line_set_object.lines = o3d.utility.Vector2iVector(lines)
            line_set_object.paint_uniform_color([0., 0., 0.3])
            
            geometries.append(line_set_object)
            
            object_centers_pcd = getPointCloud(object_pt_centers)
            object_centers_pcd.paint_uniform_color([0., 0., 1.])

        if show_positions:
            positions_pcd = getPointCloud(positions_pts)
            positions_pcd.paint_uniform_color([0., 0., 0.])
            geometries.append(positions_pcd)
        
        renderGeometriesToFile(geometries, save_path) if save_path else renderGeometries(geometries)
        return True

    def renderPositions(self, feat_name: str = "ortho_poses", target_name: str = "object") -> bool:
        # Extract sh_params from hand_params and convert them to a NumPy array
        feat = toNumpy(self.hand_params[feat_name]) if target_name == "hand" else toNumpy(self.object_params[feat_name])
        
        # Apply principal component analysis to sh_params
        pca = PCA(n_components=3)
        pca_result = pca.fit_transform(feat)
        
        # Normalize the principal components to [0, 1] and map them to RGB space
        pca_min = pca_result.min(axis=0)
        pca_max = pca_result.max(axis=0)
        rgb_colors = (pca_result - pca_min) / (pca_max - pca_min)
        
        positions = toNumpy(self.positions)
        geometries = []
        # Create a small sphere at each position
        for i, pos in enumerate(positions):
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
            sphere.translate(pos)
            sphere.paint_uniform_color(rgb_colors[i])
            geometries.append(sphere)
        
        renderGeometries(geometries)
        return True
    
    def renderPCA_all(self, feat_name: str = "ortho_poses", target_name: str = "object") -> bool:
        # Extract sh_params from hand_params and convert them to a NumPy array
        feat = np.stack((toNumpy(self.hand_params["ortho_poses"], toNumpy(self.hand_params["sh_params"]), toNumpy(self.hand_params)), toNumpy), axis=0)
        
        # Apply principal component analysis to sh_params
        pca = PCA(n_components=3)
        pca_result = pca.fit_transform(feat)
        
        # Normalize the principal components to [0, 1] and map them to RGB space
        pca_min = pca_result.min(axis=0)
        pca_max = pca_result.max(axis=0)
        rgb_colors = (pca_result - pca_min) / (pca_max - pca_min)
        
        positions = toNumpy(self.positions)
        geometries = []
        # Create a small sphere at each position
        for i, pos in enumerate(positions):
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
            sphere.translate(pos)
            sphere.paint_uniform_color(rgb_colors[i])
            geometries.append(sphere)
        
        renderGeometries(geometries)
        return True
    
    def clone(self) -> "HOIMash":
        return deepcopy(self)
