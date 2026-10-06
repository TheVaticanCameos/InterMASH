import math
import torch

import sys

sys.path.append("../ma-sh")


import mash_cpp
from ma_sh.Config.constant import MAX_SH_DEGREE, W0
from ma_sh.Method.rotate import compute_rotation_matrix_from_ortho6d


PI = math.pi
two_PI = 2.0 * PI
CMAP_SH_DEGREE_MAX = 1


class BatchObjectMash:
    """
    Class for batched ObjectMash
    """
    def __init__(
        self,
        anchor_num: int  = 256,     # N
        mask_degree_max: int = 3,
        sh_degree_max: int = 2,
        sample_phi_num: int = 32,   # S
        sample_theta_num: int = 32, # S
        dtype: torch.dtype = torch.float32,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = 1,        # B
    ) -> None:
        self.anchor_num: int = anchor_num
        self.mask_degree_max: int = mask_degree_max
        self.sh_degree_max: int = sh_degree_max
        self.sample_phi_num: int = sample_phi_num
        self.sample_theta_num: int = sample_theta_num
        self.dtype = dtype
        self.device = device
        self.batch_size: int = batch_size

        self.positions = torch.zeros(
            [self.batch_size, self.anchor_num, 3],
            dtype=self.dtype,
            device=self.device,
        )   # [B, N, 3]

        self.object_params = {
            "mask_params": torch.zeros(
                [self.batch_size, self.anchor_num, 2 * self.mask_degree_max + 1],
                dtype=self.dtype,
                device=self.device,
            ),  # [B, N, 2 * mask_degree_max + 1]
            "sh_params": torch.zeros(
                [self.batch_size, self.anchor_num, (self.sh_degree_max + 1) ** 2],
                dtype=self.dtype,
                device=self.device,
            ),  # [B, N, (sh_degree_max + 1) ** 2]
            "ortho_poses": torch.zeros(
                [self.batch_size, self.anchor_num, 6],
                dtype=self.dtype,
                device=self.device,
            ),  # [B, N, 6]
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
        old_object_sh_params = self.object_params["sh_params"]
        self.object_params["sh_params"] = torch.zeros(
            [self.batch_size, self.anchor_num, (self.sh_degree_max + 1) ** 2],
            dtype=self.dtype,
            device=self.device,
        )
        self.object_params["sh_params"][:, :, :(old_sh_degree_max + 1) ** 2] = old_object_sh_params

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

        phi_grid, theta_grid = torch.meshgrid(sample_phis, sample_thetas, indexing="ij")    # [S, S], [S, S]
        sample_phi_theta_mat = torch.stack([phi_grid, theta_grid], dim=-1)  # [S, S, 2]

        self.expanded_sample_phi_theta_mat = sample_phi_theta_mat.unsqueeze(0).expand(
            self.anchor_num, -1, -1, -1
        )   # [N, S, S, 2]

        if self.mask_degree_max > 0:
            degrees = torch.arange(
                1, self.mask_degree_max + 1, dtype=self.dtype, device=self.device
            )
            angles = degrees.unsqueeze(1) * sample_phis.unsqueeze(0)
            self.cos_terms = torch.cos(angles)  # [3, S]
            self.sin_terms = torch.sin(angles)  # [3, S]
        
        return True

    def setGradState(self, need_grad: bool) -> bool:
        self.positions.requires_grad_(need_grad)
        for param in self.object_params.values():
            param.requires_grad_(need_grad)
        return True

    def editGrads(self, edit_fn) -> bool:
        if self.positions.grad is not None:
            self.positions.grad = edit_fn(self.positions.grad)
        
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

    def centerInitOrthoPoses(
        self,
        object_gt_points: torch.Tensor, # [B, P, N, 3]
    ) -> torch.Tensor: # [B, P]
        object_patch_center = object_gt_points.mean(dim=2)    # [B, P, 3]

        object_face_forward_vectors = object_patch_center - self.positions  # [B, P, 3]
        object_rotate_vectors = mash_cpp.toRotateVectorsByFaceForwardVectors(object_face_forward_vectors.view(-1, 3))
        object_rotate_matrixs = mash_cpp.toRotateMatrixs(object_rotate_vectors).reshape(self.batch_size, self.anchor_num, 3, 3) # [B, P, 3, 3]
        object_ortho_poses = object_rotate_matrixs[:, :, :2, :].view(self.batch_size, -1, 6)
        self.object_params["ortho_poses"].data = object_ortho_poses.detach().clone().type(self.dtype).to(self.device)

        object_dists = torch.norm(object_face_forward_vectors, dim=-1)  # [B, P]
        
        return object_dists
    
    def middlePlaneInit(
        self,
        r: float = 0.2,
        offset: float = 0.03
    ) -> bool:
        angles = torch.rand(size=(self.batch_size, self.anchor_num), dtype=self.dtype, device=self.device) * 2 * torch.pi
        radii = torch.sqrt(torch.rand(size=(self.batch_size, self.anchor_num), dtype=self.dtype, device=self.device)) * r
        x_offset = torch.cos(angles) * radii    # [B, P]
        z_offset = torch.sin(angles) * radii    # [B, P]

        points_in_disk = torch.stack([
            x_offset,
            offset * torch.ones_like(x_offset),
            z_offset
        ], dim=-1)  # [B, P, 3]

        self.positions = points_in_disk

        return True

    def middlePlaneFixInit(
        self,
        r: float = 0.2,
        offset: float = 0.03
    ) -> bool:
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
        ], dim=-1)  # [P, 3]

        self.positions = points_in_disk.expand(self.batch_size, -1, -1) # [B, P, 3]

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
        self.positions = points_on_sphere.expand(self.batch_size, -1, -1)
        return True

    def shParamInit(
        self,
        object_gt_points: torch.Tensor, # [B, P, N, 3]
    ) -> bool:
        object_dists = self.centerInitOrthoPoses(object_gt_points)
        self.object_params["sh_params"].data[:, :, 0] = object_dists
        return True

    def toMaskThetas(self) -> torch.Tensor:    # 【B, P, S】
        mask_thetas = self.object_params["mask_params"][:, :, 0:1].expand(-1, -1, self.sample_phi_num)
        if self.mask_degree_max > 0:
            cos_params = self.object_params["mask_params"][:, :, 1::2]    # [B, P, 3]
            sin_params = self.object_params["mask_params"][:, :, 2::2]    # [B, P, 3]
            mask_thetas = mask_thetas + torch.matmul(cos_params, self.cos_terms)    # [B, P, S]
            mask_thetas = mask_thetas + torch.matmul(sin_params, self.sin_terms)    # [B, P, S]
        return torch.sigmoid(mask_thetas)

    def toWeightedSamplePhiThetaMat(self) -> torch.Tensor:
        theta_weights = self.toMaskThetas()   # [B, P, S]

        phis = self.expanded_sample_phi_theta_mat[:, :, :, 0]   # [P, S, S]
        thetas = self.expanded_sample_phi_theta_mat[:, :, :, 1] # [P, S, S]

        phis = phis.unsqueeze(0).expand(self.batch_size, -1, -1, -1)    # [B, P, S, S]
        thetas = thetas.unsqueeze(0).expand(self.batch_size, -1, -1, -1)# [B, P, S, S]

        weighted_thetas = thetas * theta_weights.unsqueeze(-1)  # [B, P, S, S]

        weighted_sample_phi_theta_mat = torch.stack([phis, weighted_thetas], dim=-1)    # [B, P, S, S, 2]

        return weighted_sample_phi_theta_mat

    def toDirections(
        self, 
        phis: torch.Tensor,     # [B, P, S, S]
        thetas: torch.Tensor    # [B, P, S, S]
    ) -> torch.Tensor:          # [B, P, S, S, 3]
        sin_theta, cos_theta = torch.sin(thetas), torch.cos(thetas)
        sin_phi, cos_phi = torch.sin(phis), torch.cos(phis)

        x = sin_theta * cos_phi
        y = sin_theta * sin_phi
        z = cos_theta

        return torch.stack([x, y, z], dim=-1)

    def toDistances(
        self, 
        phis: torch.Tensor,     # [B, P, S, S]
        thetas: torch.Tensor,   # [B, P, S, S]
    ) -> torch.Tensor:          # [B, P, S, S]
        # flatten batch and anchor to feed non-batch cpp
        BP = phis.shape[0] * phis.shape[1]
        phis_flat = phis.reshape(BP, phis.shape[2], phis.shape[3])      # [B*P, S, S]
        thetas_flat = thetas.reshape(BP, thetas.shape[2], thetas.shape[3])  # [B*P, S, S]

        # cpp returns base values stacked over SH basis
        base_values = mash_cpp.toSHBaseValues(phis_flat, thetas_flat, self.sh_degree_max)  # [(D+1)^2, B*P, S, S]
        
        base_values = base_values.transpose(1, 0)    # [B*P, (D+1)^2, S, S]

        # reshape params to match B*P dimension
        sh_params = self.object_params["sh_params"].reshape(BP, -1)  # [B*P, (D+1)^2]

        # accumulate over SH basis
        sample_distances_flat = torch.einsum("bn,bn...->b...", sh_params, base_values)  # [B*P, S, S]

        # reshape back to [B, P, S, S]
        sample_distances = sample_distances_flat.reshape(phis.shape[0], phis.shape[1], phis.shape[2], phis.shape[3])  # [B, P, S, S]

        return sample_distances

    def toSampleMoveVectors(self) -> torch.Tensor: # [B, P, S, S, 3]
        weighted_sample_phi_theta_mat = self.toWeightedSamplePhiThetaMat()    # [B, P, S, S, 2]
        phis, thetas = weighted_sample_phi_theta_mat.split(1, dim=-1)
        phis = phis.squeeze(-1)     # [B, P, S, S]
        thetas = thetas.squeeze(-1) # [B, P, S, S]

        sample_directions = self.toDirections(phis, thetas) # [B, P, S, S, 3]
        sample_distances = self.toDistances(phis, thetas)   # [B, P, S, S]

        sample_move_vectors = sample_directions * sample_distances.unsqueeze(-1)    # [B, P, S, S, 3]
        return sample_move_vectors

    def toInvSampleMoveVectors(self) -> torch.Tensor:
        sample_move_vectors = self.toSampleMoveVectors()  # [B, P, S, S, 3]
        sample_move_vectors = sample_move_vectors.reshape(
            self.batch_size, self.anchor_num, -1, 3
        )  # [B, P, S*S, 3]

        inv_radius = W0 * self.object_params["sh_params"][:, :, 0]  # [B, P]

        inv_centers = torch.zeros(
            self.batch_size, self.anchor_num, 1, 3, dtype=sample_move_vectors.dtype, device=sample_move_vectors.device
        )  # [B, P, 1, 3]
        inv_centers[:, :, 0, 2] = -inv_radius

        in_inv_points = sample_move_vectors - inv_centers  # [B, P, S*S, 3]

        in_inv_point_norms_sq = torch.sum(
            in_inv_points.square(), dim=3, keepdim=True
        )  # [B, P, S*S, 1]

        in_inv_point_weights = torch.rsqrt(in_inv_point_norms_sq)  # [B, P, S*S, 1]

        in_inv_point_directions = in_inv_points * in_inv_point_weights  # [B, P, S*S, 3]

        inv_radius_sq_4 = (4.0 * inv_radius.square()).view(
            self.batch_size, self.anchor_num, 1, 1
        )  # [B, P, 1, 1]

        in_inv_point_lengths = inv_radius_sq_4 * in_inv_point_weights  # [B, P, S*S, 1]

        inv_points = torch.addcmul(
            inv_centers, in_inv_point_lengths, in_inv_point_directions
        )  # [B, P, S*S, 3]

        inv_points = inv_points.view(
            self.batch_size, self.anchor_num, self.sample_phi_num, self.sample_theta_num, 3
        )  # [B, P, S, S, 3]

        return inv_points

    def toSamplePoints(self) -> torch.Tensor:    # [B, P, S, S, 3]
        sample_move_vectors = self.toInvSampleMoveVectors()    # [B, P, S, S, 3]

        ortho = self.object_params["ortho_poses"].reshape(-1, 6)  # [B*P, 6]
        rotate_mats = compute_rotation_matrix_from_ortho6d(ortho).reshape(self.batch_size, self.anchor_num, 3, 3)  # [B, P, 3, 3]

        rotated_sample_move_vectors = torch.einsum(
            "bp...i,bpij->bp...j", sample_move_vectors, rotate_mats
        )  # [B, P, S, S, 3]

        positions_expanded = self.positions.view(
            self.batch_size, self.anchor_num, *((1,) * (sample_move_vectors.dim() - 3)), 3
        )  # [B, P, 1, 1, 3]

        sample_points = positions_expanded + rotated_sample_move_vectors  # [B, P, S, S, 3]

        return sample_points

    
if __name__ == "__main__":
    batch_object_mash = BatchObjectMash(
        anchor_num=256,
        mask_degree_max=3,
        sh_degree_max=0,
        sample_phi_num=32,
        sample_theta_num=32,
        dtype=torch.float32,
        device="cuda" if torch.cuda.is_available() else "cpu",
        batch_size=2,
    )
    batch_object_mash.centerInitOrthoPoses(
        object_gt_points=torch.randn([2, 256, 1024, 3], dtype=torch.float32, device="cuda"),
    )
    batch_object_mash.middlePlaneFixInit()
    batch_object_mash.shParamInit(
        object_gt_points=torch.randn([2, 256, 1024, 3], dtype=torch.float32, device="cuda"),
    )
    sample_points = batch_object_mash.toSamplePoints()
    print(sample_points.shape)
    print(sample_points)
