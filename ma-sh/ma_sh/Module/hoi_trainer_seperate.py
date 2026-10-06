import os
import torch
import numpy as np
from tqdm import tqdm
from typing import Union, Dict
from torch.optim import Adam
import trimesh
from tensorboardX import SummaryWriter
from scipy.spatial.distance import cdist
from scipy.spatial import KDTree
from scipy.optimize import linear_sum_assignment

from mesh_cut.Module.average_mesh_cutter import AverageMeshCutter

from chamfer_distance.Module.chamfer_distances import ChamferDistances

from ma_sh.Config.constant import EPSILON, SHADOW_HAND_SKIPPED_FACE_IDS
from ma_sh.Method.data import toNumpy
from ma_sh.Model.hoi_mash import HOIMash
from ma_sh.Module.early_stopping import EarlyStopping
from ma_sh.Module.logger import Logger


class HOITrainerSeperate(object):
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
        template_info_path: str = None,
        hand_type: str = "mano",
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

        self.hand_template_info: Dict = np.load(template_info_path, allow_pickle=True).item()

        return
    
    # NOTE: DON'T DELETE THIS ! IT WORKS !
    # def prepareTemplate(self) -> Dict:
    #     if self.hand_type == "mano":
    #         template_path = "/data2/yxz/DATASET/GrabDataset/hand_rest.ply"
    #     elif self.hand_type == "shadow":
    #         template_path = "/data2/yxz/DATASET/DexGraspNet/Processed/shadow_template_hand.ply"
    #     else:
    #         raise ValueError(f"Hand type {self.hand_type} is not supported!")
        
    #     template_mesh = trimesh.load(template_path)
    #     hand_verts = np.array(template_mesh.vertices)
    #     hand_faces = np.array(template_mesh.faces)

    #     average_mesh_cutter = AverageMeshCutter(
    #         template_path,
    #         dist_max=1.0 / 100,
    #         is_unique_label=True,
    #         print_progress=True
    #     )
    #     average_mesh_cutter.cutMesh(
    #         self.submesh_num, 
    #         self.points_per_submesh
    #     )
    #     template_points = average_mesh_cutter.sub_mesh_sample_points.cpu().numpy().reshape(-1, 3)

    #     closest_points, distances, triangle_ids = template_mesh.nearest.on_surface(template_points)
    #     target_triangles = template_mesh.triangles[triangle_ids]
    #     barycentric_coords = trimesh.triangles.points_to_barycentric(
    #         target_triangles, 
    #         closest_points
    #     )

    #     kdtree = KDTree(template_points)
    #     dist, idx = kdtree.query(hand_verts)
    #     patch_ids = idx // self.points_per_submesh

    #     template_npy = {
    #         "verts": hand_verts,
    #         "faces": hand_faces,
    #         "patch_ids": patch_ids,
    #         "submesh_num": self.submesh_num,
    #         "points_per_submesh": self.points_per_submesh,
    #         "hand_points": template_points,
    #         "triangle_ids": triangle_ids,
    #         "barycentric_coords": barycentric_coords
    #     }
        
    #     if self.hand_type == "mano":
    #         np.save("/data2/yxz/DATASET/GrabDataset/hand_template.npy", template_npy)
    #     # np.save("shadow_template_hand.npy", template_npy)
        
    #     return template_npy

    def prepareTemplate(self) -> Dict:
        if self.hand_type == "mano":
            template_path = "/data2/yxz/DATASET/GrabDataset/hand_rest.ply"
        elif self.hand_type == "shadow":
            template_path = "/data2/yxz/DATASET/DexGraspNet/Processed/shadow_template_hand.ply"
        else:
            raise ValueError(f"Hand type {self.hand_type} is not supported!")
        
        template_mesh = trimesh.load(template_path)
        hand_verts = np.array(template_mesh.vertices)
        hand_faces = np.array(template_mesh.faces)

        average_mesh_cutter = AverageMeshCutter(
            template_path,
            dist_max=1.0 / 100,
            is_unique_label=True,
            print_progress=True
        )
        average_mesh_cutter.cutMesh(
            self.submesh_num, 
            self.points_per_submesh
        )
        template_points = average_mesh_cutter.sub_mesh_sample_points.cpu().numpy().reshape(-1, 3)

        # Filter points based on SHADOW_HAND_SKIPPED_FACE_IDS
        if self.hand_type == "shadow":
            # Get triangle IDs for each sampled point
            closest_points, distances, triangle_ids = template_mesh.nearest.on_surface(template_points)
            closest_points = closest_points.reshape((self.submesh_num, self.points_per_submesh, 3))
            triangle_ids = triangle_ids.reshape((self.submesh_num, self.points_per_submesh))
            # Create a mask to filter out points belonging to skipped face IDs
            valid_mask = np.isin(triangle_ids, SHADOW_HAND_SKIPPED_FACE_IDS, invert=True)
            template_points = template_points.reshape((self.submesh_num, self.points_per_submesh, 3))

            # Replace invalid points with random points from the same group
            for i in range(self.submesh_num):
                for j in range(self.points_per_submesh):
                    if not valid_mask[i, j]:
                        # Randomly select a valid point from the same group as a replacement
                        valid_indices = np.where(valid_mask[i])[0]
                        if len(valid_indices) > 0:
                            replace_idx = np.random.choice(valid_indices)
                            template_points[i, j] = template_points[i, replace_idx]
                        else:
                            # If the entire group is invalid, select a random point from that group without further processing
                            replace_idx = np.random.randint(self.points_per_submesh)
                            template_points[i, j] = template_points[i, replace_idx]

        # # Visualize template_points with Open3D, coloring groups according to submesh_num
        # import open3d as o3d
        # import matplotlib.cm as cm

        # # Reshape template_points to (submesh_num, points_per_submesh, 3)
        # template_points_reshaped = template_points.reshape(self.submesh_num, self.points_per_submesh, 3)

        # # Create a color map
        # colors = cm.tab20(np.linspace(0, 1, self.submesh_num))[:, :3]  # Keep RGB and discard alpha

        # # Build a colored point cloud
        # all_points = []
        # all_colors = []
        # for i in range(self.submesh_num):
        #     group_points = template_points_reshaped[i]
        #     group_color = colors[i]
        #     all_points.append(group_points)
        #     all_colors.append(np.tile(group_color, (self.points_per_submesh, 1)))

        # all_points = np.vstack(all_points)
        # all_colors = np.vstack(all_colors)

        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(all_points)
        # pcd.colors = o3d.utility.Vector3dVector(all_colors)

        # o3d.visualization.draw_geometries([pcd], window_name="Template Points by Submesh")

        template_points = template_points.reshape(-1, 3)
        
        # Recompute triangle IDs and barycentric coordinates after filtering
        closest_points, distances, triangle_ids = template_mesh.nearest.on_surface(template_points)
        target_triangles = template_mesh.triangles[triangle_ids]
        barycentric_coords = trimesh.triangles.points_to_barycentric(
            target_triangles, 
            closest_points
        )

        kdtree = KDTree(template_points)
        dist, idx = kdtree.query(hand_verts)
        patch_ids = idx // self.points_per_submesh

        template_npy = {
            "verts": hand_verts,
            "faces": hand_faces,
            "patch_ids": patch_ids,
            "submesh_num": self.submesh_num,
            "points_per_submesh": self.points_per_submesh,
            "hand_points": template_points,
            "triangle_ids": triangle_ids,
            "barycentric_coords": barycentric_coords
        }

        if self.hand_type == "mano":
            np.save("/data2/yxz/DATASET/GrabDataset/hand_template.npy", template_npy)
        elif self.hand_type == "shadow":
            np.save("/data2/yxz/DATASET/DexGraspNet/Processed/shadow_template_hand.npy", template_npy)
        
        return template_npy

    # def loadMeshFile(
    #     self, 
    #     hand_path: str,
    #     object_path: str
    # ) -> bool:
    #     if not os.path.exists(hand_path): 
    #         print("[ERROR][MeshTrainer::loadMeshFile]")
    #         print("\t hand mesh file not exist!")
    #         print("\t hand_path:", hand_path)
    #         return False
    #     if not os.path.exists(object_path): 
    #         print("[ERROR][MeshTrainer::loadMeshFile]")
    #         print("\t object mesh file not exist!")
    #         print("\t object_path:", object_path)
    #         return False
        
    #     self.hand_gt_points = self.getInitialHandGT(hand_path)
    #     self.object_gt_points = self.getInitialObjectGT(object_path)

    #     # initialize mash parameters here 
    #     # [TODO] check anchor distribution size !
    #     if self.initial_method == "random":
    #         print("[INFO][HOITrainer::loadMeshFile]")
    #         print("\t initial_method: random")
    #         self.hoi_mash.randomInit()
    #     elif self.initial_method == "voxel":
    #         print("[INFO][HOITrainer::loadMeshFile]")
    #         print("\t initial_method: voxel")
    #         self.hoi_mash.voxelInit(self.hand_gt_points, self.object_gt_points)
    #     elif self.initial_method == "middle_plane":
    #         print("[INFO][HOITrainer::loadMeshFile]")
    #         print("\t initial_method: middle_plane")
    #         self.hoi_mash.middlePlaneInit(self.hand_gt_points, self.object_gt_points)   # TODO: HERE TO CHECK ANCHOR DISTRIBUTION SIZE !
    #     else:
    #         print("[ERROR][HOITrainer::loadMeshFile]")
    #         print("\t initial_method not support!")
    #         print("\t initial_method:", self.initial_method)
    #         return False

    #     # self.matchHandGT()
    #     # self.matchObjectGT()

    #     self.hoi_mash.shParamInit(self.hand_gt_points, self.object_gt_points)

    #     return True

    def loadMeshFile(
        self, 
        hand_path: str,
        object_path: str,
        cmap_path: str = None
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
        # [TODO] check anchor distribution size !
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
    
    # def getInitialHandGT(self, hand_path: str) -> torch.Tensor:
    #     hand_mesh = trimesh.load(hand_path)
    #     sampled_points, face_indices = trimesh.sample.sample_surface(
    #         hand_mesh,
    #         count=self.submesh_num * self.points_per_submesh,
    #         seed=None
    #     )
    #     sampled_points = np.array(sampled_points)
    #     face_indices = np.array(face_indices)
    #     pc_in_face_ids = self.hand_template_info["faces"][face_indices]
    #     patch_ids_of_face = self.hand_template_info["patch_ids"][pc_in_face_ids]
    #     patch_ids_pc = np.take_along_axis(patch_ids_of_face, np.random.randint(0, 3, size=(patch_ids_of_face.shape[0], 1)), axis=1).flatten()

    #     sorted_indices = np.argsort(patch_ids_pc)
    #     sorted_points = sampled_points[sorted_indices]
    #     sampled_points = sorted_points.reshape(self.submesh_num, self.points_per_submesh, 3)
        
    #     return torch.from_numpy(sampled_points).float()

    # def getInitialObjectGT(self, object_path: str) -> torch.Tensor: # FIXME
    #     object_mesh = trimesh.load(object_path)
    #     sampled_points, face_indices = trimesh.sample.sample_surface(
    #         object_mesh,
    #         count=self.submesh_num * self.points_per_submesh,
    #         seed=None
    #     )
    #     return torch.tensor(sampled_points).float().reshape(self.submesh_num, self.points_per_submesh, 3)

    def getInitialObjectGT(self, object_path: str) -> torch.Tensor:
        average_mesh_cutter = AverageMeshCutter(
            object_path,
            dist_max=1.0 / 100,
            is_unique_label=True,
            print_progress=True
        )
        average_mesh_cutter.cutMesh(
            self.submesh_num, 
            self.points_per_submesh
        )
        return torch.from_numpy(average_mesh_cutter.sub_mesh_sample_points.cpu().numpy().reshape(self.submesh_num, self.points_per_submesh, 3)).float()

    def matchHandGT(self) -> bool:
        hand_template_points = self.hand_template_info["hand_points"]   # (B, N, 3)
        patch_centers = hand_template_points.mean(axis=1)   # (B, 3)
        cost_matrix = cdist(self.hoi_mash.positions.cpu().numpy(), patch_centers, metric="euclidean")

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        assert (row_ind == np.arange(self.submesh_num)).all(), "LAP ERROR: row_ind must equals to [1,2,..., submesh_num]"
        
        self.hand_gt_points = self.hand_gt_points[col_ind]
        
        # # Visualize hand_template_points with Open3D using distinct patch colors and connect patch centers to the corresponding self.hoi_mash.positions
        # import matplotlib.cm as cm

        # # Generate colors
        # colors = cm.tab20(np.linspace(0, 1, self.submesh_num))[:, :3]

        # # Prepare point clouds and colors
        # vis_points = []
        # vis_colors = []
        # lines = []
        # line_colors = []

        # # Index offset for the existing points
        # idx_offset = 0

        # for i in range(self.submesh_num):
        #     group_points = hand_template_points[i]  # (N, 3)
        #     group_color = colors[i]
        #     vis_points.append(group_points)
        #     vis_colors.append(np.tile(group_color, (group_points.shape[0], 1)))

        #     # Connect the patch center to the corresponding hoi_mash position
        #     center = patch_centers[i]
        #     target = self.hoi_mash.positions[i].cpu().numpy()
        #     lines.append([idx_offset + group_points.shape[0], idx_offset + group_points.shape[0] + 1])
        #     line_colors.append(group_color)

        #     # Add the center and target points
        #     vis_points.append(center.reshape(1, 3))
        #     vis_colors.append(group_color.reshape(1, 3))
        #     vis_points.append(target.reshape(1, 3))
        #     vis_colors.append(group_color.reshape(1, 3))

        #     idx_offset += group_points.shape[0] + 2

        # # Merge
        # all_points = np.vstack(vis_points)
        # all_colors = np.vstack(vis_colors)

        # # Build the point cloud
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(all_points)
        # pcd.colors = o3d.utility.Vector3dVector(all_colors)

        # # Build connecting lines
        # line_set = o3d.geometry.LineSet()
        # line_set.points = o3d.utility.Vector3dVector(all_points)
        # line_set.lines = o3d.utility.Vector2iVector(lines)
        # line_set.colors = o3d.utility.Vector3dVector(line_colors)

        # # Visualize
        # o3d.visualization.draw_geometries([pcd, line_set], window_name="Hand Template Points with Patch Centers and Links")

        # import pdb
        # pdb.set_trace()

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
        if hasattr(self, "writer") and self.writer is not None:
            self.writer.close()
        return

    def upSampleCMapGT(self) -> torch.Tensor:
        # hand_pc = self.hoi_mash.toSamplePoints("hand").view(-1, 3)
        upSampledPC = self.hoi_mash.toSamplePoints("object").view(-1, 3)
        if self.cmap is None or "pc" not in self.cmap or "cmap" not in self.cmap:
            if self.hoi_mash.device == "cpu":
                gt_dtype = self.hoi_mash.dtype
            else:
                gt_dtype = torch.float32
            cmap_gt = torch.zeros(upSampledPC.shape[0], device=self.hoi_mash.device, dtype=gt_dtype)
            self.cmap_gt = cmap_gt
            return cmap_gt
        
        pc_sparse = self.cmap["pc"]
        cmap_sparse = self.cmap["cmap"]
        if isinstance(pc_sparse, torch.Tensor):
            pc_sparse_np = pc_sparse.detach().cpu().numpy()
        else:
            pc_sparse_np = np.asarray(pc_sparse)
        if isinstance(cmap_sparse, torch.Tensor):
            cmap_sparse_np = cmap_sparse.detach().cpu().numpy()
        else:
            cmap_sparse_np = np.asarray(cmap_sparse)
        query_np = upSampledPC.detach().cpu().numpy()
        k = int(min(8, max(1, pc_sparse_np.shape[0])))
        tree = KDTree(pc_sparse_np)
        dists, idxs = tree.query(query_np, k=k)
        if k == 1:
            dists = dists[:, None]
            idxs = idxs[:, None]
        values = cmap_sparse_np[idxs]
        eps = 1e-8
        weights = 1.0 / ((dists + eps) ** 2)
        has_zero = (dists <= eps).any(axis=1)
        denom = weights.sum(axis=1)
        num = (weights * values).sum(axis=1)
        pred = num / np.maximum(denom, eps)
        if has_zero.any():
            zero_first = (dists <= eps).argmax(axis=1)
            rows = np.arange(query_np.shape[0])[has_zero]
            pred[has_zero] = values[rows, zero_first[has_zero]]
        if self.hoi_mash.device == "cpu":
            gt_dtype = self.hoi_mash.dtype
        else:
            gt_dtype = torch.float32
        cmap_gt = torch.from_numpy(pred).to(self.hoi_mash.device, dtype=gt_dtype)
        self.cmap_gt = cmap_gt

        return cmap_gt

    def trainStepHand(
        self,
        hand_gt_points: torch.Tensor,
        fit_loss_weight: float,
        coverage_loss_weight: float,
        scheduler = None
    ) -> Union[Dict, None]:
        self.optimizer.zero_grad()

        hand_mash_pts = self.hoi_mash.toSamplePoints("hand").reshape(self.hoi_mash.anchor_num, -1, 3)
        
        hand_fit_loss = torch.tensor(0.0).type(hand_gt_points.dtype).to(hand_gt_points.device)
        hand_coverage_loss = torch.zeros_like(hand_fit_loss)
        
        if torch.cuda.is_available() and hand_gt_points.device != "cpu":
            hand_fit_dist2, hand_coverage_dist2 = ChamferDistances.namedAlgo("cuda")(
                hand_mash_pts, hand_gt_points
            )[:2]
        else:
            hand_fit_dist2, hand_coverage_dist2 = ChamferDistances.namedAlgo("default")(
                hand_mash_pts, hand_gt_points
            )[:2]
        
        hand_fit_dists = torch.sqrt(hand_fit_dist2 + EPSILON)
        hand_coverage_dists = torch.sqrt(hand_coverage_dist2 + EPSILON)

        hand_fit_loss = torch.mean(hand_fit_dists)
        hand_coverage_loss = torch.mean(hand_coverage_dists)

        hand_weighted_fit_loss = fit_loss_weight * hand_fit_loss
        hand_weighted_coverage_loss = coverage_loss_weight * hand_coverage_loss

        loss = hand_weighted_fit_loss + hand_weighted_coverage_loss

        if torch.isnan(loss).any():
            print("[ERROR][BaseTrainer::trainStep]")
            print("\t loss is nan!")
            print("\t\t hand_fit_loss:", hand_fit_loss)
            print("\t\t hand_coverage_loss:", hand_coverage_loss)
            return None

        loss.backward()

        # self.hoi_mash.clearNanGrads()
        
        self.optimizer.step()
        if scheduler is not None:
            scheduler.step()

        hand_cd = toNumpy(hand_fit_loss) + toNumpy(hand_coverage_loss)
        self.hand_error = hand_cd

        loss_dict = {
            "Epoch": self.epoch,
            "Fit Loss/hand_weighted_fit_loss": toNumpy(hand_weighted_fit_loss),
            "Coverage Loss/hand_weighted_coverage_loss": toNumpy(hand_weighted_coverage_loss),
            "CD/hand_cd": hand_cd,
        }

        if self.writer is not None:
            for name, value in loss_dict.items():
                if name != "Epoch":
                    self.writer.add_scalar(name, value, self.step)
            self.writer.add_scalar("LR/lr", self.optimizer.param_groups[0]["lr"], self.step)

        return loss_dict

    def trainStepObject(
        self,
        object_gt_points: torch.Tensor,
        fit_loss_weight: float,
        coverage_loss_weight: float,
        scheduler = None
    ) -> Union[Dict, None]:
        self.optimizer.zero_grad()

        object_mash_pts = self.hoi_mash.toSamplePoints("object").reshape(self.hoi_mash.anchor_num, -1, 3)

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
        self.object_error = object_cd

        loss_dict = {
            "Epoch": self.epoch,
            "Fit Loss/object_weighted_fit_loss": toNumpy(object_weighted_fit_loss),
            "Coverage Loss/object_weighted_coverage_loss": toNumpy(object_weighted_coverage_loss),
            "CD/object_cd": object_cd,
        }

        if self.writer is not None:
            for name, value in loss_dict.items():
                if name != "Epoch":
                    self.writer.add_scalar(name, value, self.step)
            self.writer.add_scalar("LR/lr", self.optimizer.param_groups[0]["lr"], self.step)

        return loss_dict

    def trainStepCMap(
        self,
        cmap_gt: torch.Tensor,
        scheduler = None
    ) -> Union[Dict, None]:
        self.optimizer.zero_grad()

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

    def warmUpEpochHand(
        self, 
        epoch_lr: float,
        hand_gt_points: torch.Tensor, 
        fit_loss_weight: float,
        coverage_loss_weight: float,
        train_step_max: int,
    ) -> bool:
        self.hoi_mash.setGradState(True)

        print("[INFO][BaseTrainer::warmUpEpochHand]")
        print("\t start train epoch :", self.epoch, "with lr :", epoch_lr, "...")
        pbar = tqdm(total=train_step_max)

        start_step = self.step

        for i in range(train_step_max):
            current_lr = epoch_lr * (i + 1.0) / train_step_max
            self.updateLr(current_lr)

            loss_dict= self.trainStepHand(
                hand_gt_points, 
                fit_loss_weight, 
                coverage_loss_weight
            )

            if loss_dict is None:
                print("[ERROR][BaseTrainer::warmUpEpochHand]")
                print("\t trainStep failed!")
                return False

            assert isinstance(loss_dict, dict)
            
            loss = loss_dict["CD/hand_cd"]

            pbar.set_description("LOSS %.6f" % (loss,))

            self.step += 1
            pbar.update(1)

            if self.step >= start_step + train_step_max:
                break
        
        self.epoch += 1
        return True

    def warmUpEpochObject(
        self, 
        epoch_lr: float,
        object_gt_points: torch.Tensor, 
        fit_loss_weight: float,
        coverage_loss_weight: float,
        train_step_max: int,
    ) -> bool:
        self.hoi_mash.setGradState(True)

        print("[INFO][BaseTrainer::warmUpEpochObject]")
        print("\t start train epoch :", self.epoch, "with lr :", epoch_lr, "...")
        pbar = tqdm(total=train_step_max)

        start_step = self.step

        for i in range(train_step_max):
            current_lr = epoch_lr * (i + 1.0) / train_step_max
            self.updateLr(current_lr)

            loss_dict= self.trainStepObject(
                object_gt_points, 
                fit_loss_weight, 
                coverage_loss_weight
            )

            if loss_dict is None:
                print("[ERROR][BaseTrainer::warmUpEpochObject]")
                print("\t trainStep failed!")
                return False

            assert isinstance(loss_dict, dict)
            
            loss = loss_dict["CD/object_cd"]

            pbar.set_description("LOSS %.6f" % (loss,))

            self.step += 1
            pbar.update(1)

            if self.step >= start_step + train_step_max:
                break
        
        self.epoch += 1
        return True

    def trainEpochHand(
        self,
        epoch_lr: float,
        hand_gt_points: torch.Tensor,
        fit_loss_weight: float,
        coverage_loss_weight: float,
        train_step_max: Union[int, None] = None,
    ) -> bool:
        early_stopping = EarlyStopping(self.patience, 1e-6)
        
        self.hoi_mash.setGradState(True)

        self.updateLr(epoch_lr)

        print("[INFO][BaseTrainer::trainEpochHand]")
        print("\t start train epoch :", self.epoch, "with lr :", epoch_lr, "...")
        if train_step_max is not None:
            pbar = tqdm(total=train_step_max)
        else:
            pbar = tqdm()

        start_step = self.step
        min_loss = float("inf")
        min_loss_reached_time = 0

        while True:
            loss_dict = self.trainStepHand(
                hand_gt_points,
                fit_loss_weight,
                coverage_loss_weight
            )

            if loss_dict is None:
                print("[ERROR][BaseTrainer::trainEpochHand]")
                print("\t trainStep failed!")
                return False

            assert isinstance(loss_dict, dict)
            
            loss = loss_dict["CD/hand_cd"]

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

    def trainEpochObject(
        self,
        epoch_lr: float,
        object_gt_points: torch.Tensor,
        fit_loss_weight: float,
        coverage_loss_weight: float,
        train_step_max: Union[int, None] = None,
    ) -> bool:
        early_stopping = EarlyStopping(self.patience, 1e-6)
        
        self.hoi_mash.setGradState(True)

        self.updateLr(epoch_lr)

        print("[INFO][BaseTrainer::trainEpochObject]")
        print("\t start train epoch :", self.epoch, "with lr :", epoch_lr, "...")
        if train_step_max is not None:
            pbar = tqdm(total=train_step_max)
        else:
            pbar = tqdm()

        start_step = self.step
        min_loss = float("inf")
        min_loss_reached_time = 0

        while True:
            loss_dict = self.trainStepObject(
                object_gt_points,
                fit_loss_weight,
                coverage_loss_weight
            )

            if loss_dict is None:
                print("[ERROR][BaseTrainer::trainEpochObject]")
                print("\t trainStep failed!")
                return False

            assert isinstance(loss_dict, dict)
            
            loss = loss_dict["CD/object_cd"]

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

    def trainCMap(
        self,
        cmap_gt: torch.Tensor,
        epoch_num: int,
        lr: float,
    ) -> float:
        # Freeze all parameters
        self.hoi_mash.setGradState(False)
        # Unfreeze only cmap's sh_params
        self.hoi_mash.cmap_params["sh_params"].requires_grad_(True)
        # Optimize only cmap's sh_params
        self.optimizer = Adam([
            self.hoi_mash.cmap_params["sh_params"]
        ], lr=lr)
        with tqdm(total=epoch_num, desc="CMap training") as pbar:
            for _ in range(epoch_num):
                loss_dict = self.trainStepCMap(cmap_gt)
                if loss_dict is None:
                    print("[ERROR][Trainer::escalateTrainMash]")
                    print("\t trainStepCMap failed!")
                    return False
                
                pbar.set_postfix({"loss": f"{loss_dict['CMap/cmap_loss']:.5f}"})
                pbar.update(1)

        return loss_dict["CMap/cmap_loss"].item()

    def autoTrainMash(self) -> bool:        
        print("[INFO][HOITrainer::autoTrainMash]")
        print("\t start auto train SimpleMash...")
        print(
            "\t degree: mask:",
            self.hoi_mash.mask_degree_max,
            "sh:",
            self.hoi_mash.sh_degree_max,
        )

        if self.hoi_mash.device == "cpu":
            gt_points_dtype = self.hoi_mash.dtype
        else:
            gt_points_dtype = torch.float32

        hand_gt_points = self.hand_gt_points.to(self.hoi_mash.device, dtype=gt_points_dtype)
        object_gt_points = self.object_gt_points.to(self.hoi_mash.device, dtype=gt_points_dtype)

        # PART 1: training MASH for hand
        print("[INFO][HOITrainer::autoTrainMash]")
        print("\t start warmUpEpochHand...") 
        if not self.warmUpEpochHand(self.lr, hand_gt_points, 1.0, 0.5, self.warmup_step_num):
            print("[ERROR][HOITrainer::autoTrainMash]")
            print("\t warmUpEpochHand failed!")
            return False
        
        print("[INFO][HOITrainer::autoTrainMash]")
        print("\t start trainEpochHand...")
        if not self.trainEpochHand(self.lr, hand_gt_points, 1.0, 0.5):
            print("[ERROR][HOITrainer::autoTrainMash]")
            print("\t trainEpochHand failed!")
            return False
        
        print("[INFO][HOITrainer::autoTrainMash]")
        print("\t start second warmUpEpochHand...")
        if not self.warmUpEpochHand(self.lr, hand_gt_points, 1.0, 1.0, self.warmup_step_num):
            print("[ERROR][HOITrainer::autoTrainMash]")
            print("\t warmUpEpochHand failed!")
            return False

        print("[INFO][HOITrainer::autoTrainMash]")
        print("\t start second trainEpochHand...")
        if not self.trainEpochHand(self.lr, hand_gt_points, 1.0, 1.0):
            print("[ERROR][HOITrainer::autoTrainMash]")
            print("\t trainEpochHand failed!")
            return False
        
        print("[INFO][HOITrainer::autoTrainMash]")
        print("\t training finished! metrics:")
        print("\t hand_error: {:.5f}".format(self.hand_error))

        # PART 2: training MASH for object
        print("[INFO][HOITrainer::autoTrainMash]")
        print("\t start warmUpEpochObject...") 
        if not self.warmUpEpochObject(self.lr, object_gt_points, 1.0, 0.5, self.warmup_step_num):
            print("[ERROR][HOITrainer::autoTrainMash]")
            print("\t warmUpEpochObject failed!")
            return False
        
        print("[INFO][HOITrainer::autoTrainMash]")
        print("\t start trainEpochObject...")
        if not self.trainEpochObject(self.lr, object_gt_points, 1.0, 0.5):
            print("[ERROR][HOITrainer::autoTrainMash]")
            print("\t trainEpochObject failed!")
            return False
        
        print("[INFO][HOITrainer::autoTrainMash]")
        print("\t start second warmUpEpochObject...")
        if not self.warmUpEpochObject(self.lr, object_gt_points, 1.0, 1.0, self.warmup_step_num):
            print("[ERROR][HOITrainer::autoTrainMash]")
            print("\t warmUpEpochObject failed!")
            return False

        print("[INFO][HOITrainer::autoTrainMash]")
        print("\t start second trainEpochObject...")
        if not self.trainEpochObject(self.lr, object_gt_points, 1.0, 1.0):
            print("[ERROR][HOITrainer::autoTrainMash]")
            print("\t trainEpochObject failed!")
            return False
        
        cmap_gt = self.upSampleCMapGT()
        print("[INFO][HOITrainer::autoTrainMash]")
        print("\t start trainEpochCMap...")
        cmap_loss = self.trainCMap(cmap_gt, 6000, self.lr)
        
        self.error = self.hand_error + self.object_error

        # training finished
        print("[INFO][HOITrainer::autoTrainMash]")
        print("\t auto train HOIMash finished!")
        print("\t total error: {:.5f}".format(self.error))
        print("\t hand_error: {:.5f}".format(self.hand_error))
        print("\t object_error: {:.5f}".format(self.object_error))
        print("\t cmap_loss: {:.5f}".format(cmap_loss))

        self.closeLog()

        return True

    def directTrainMash(self, epoch_num: int = 800) -> bool:
        if self.hoi_mash.device == "cpu":
            gt_points_dtype = self.hoi_mash.dtype
        else:
            gt_points_dtype = torch.float32

        hand_gt_points = self.hand_gt_points.to(self.hoi_mash.device, dtype=gt_points_dtype)
        object_gt_points = self.object_gt_points.to(self.hoi_mash.device, dtype=gt_points_dtype)

        print("[INFO][Trainer::directTrainMash]")
        print("\t start direct train HOIMash...")

        self.hoi_mash.setGradState(True)

        pbar = tqdm(total=epoch_num)

        # PART 1: training MASH for hand
        scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=500, gamma=0.9)

        for _ in range(epoch_num):
            loss_dict = self.trainStepHand(
                hand_gt_points,
                1.0,
                1.0,
                scheduler
            )

            if loss_dict is None:
                print("[ERROR][BaseTrainer::warmUpEpoch]")
                print("\t trainStep failed!")
                return False

            assert isinstance(loss_dict, dict)
            
            loss = loss_dict["CD/hand_cd"]

            pbar.set_description("LOSS %.6f" % (loss,))

            self.step += 1
            pbar.update(1)


        # PART 2: training MASH for object
        scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=500, gamma=0.9)

        for _ in range(epoch_num):
            loss_dict = self.trainStepObject(
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
            
            loss = loss_dict["CD/object_cd"]

            pbar.set_description("LOSS %.6f" % (loss,))

            self.step += 1
            pbar.update(1)

        cmap_gt = self.upSampleCMapGT()
        print("[INFO][HOITrainer::escalateTrainMash]")
        print("\t start trainEpochCMap...")
        cmap_loss = self.trainCMap(cmap_gt, 6000, self.lr)
        
        self.error = self.hand_error + self.object_error

        # training finished
        print("[INFO][Trainer::escalateTrainMash]")
        print("\t escalate train HOIMash finished!")
        print("\t total error: {:.5f}".format(self.error))
        print("\t hand_error: {:.5f}".format(self.hand_error))
        print("\t object_error: {:.5f}".format(self.object_error))
        print("\t cmap_loss: {:.5f}".format(cmap_loss))

        self.closeLog()

        return True

    def escalateTrainMash(self) -> bool:
        lr_decay_rate= 0.2
        if self.hoi_mash.device == "cpu":
            gt_points_dtype = self.hoi_mash.dtype
        else:
            gt_points_dtype = torch.float32

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

        # for hand
        print("[INFO][HOITrainer::escalateTrainMash]")
        print("\t start warmUpEpochHand...")
        if not self.warmUpEpochHand(self.lr, hand_gt_points, 1.0, 0.5, self.warmup_step_num):
            print("[ERROR][Trainer::escalateTrainMash]")
            print("\t warmUpEpochHand failed!")
            return False
        
        print("[INFO][HOITrainer::escalateTrainMash]")
        print("\t start trainEpochHand...")
        if not self.trainEpochHand(self.lr, hand_gt_points, 1.0, 0.5):
            print("[ERROR][Trainer::escalateTrainMash]")
            print("\t trainEpochHand failed!")
            return False

        # for object
        print("[INFO][HOITrainer::escalateTrainMash]")
        print("\t start warmUpEpochObject...")
        if not self.warmUpEpochObject(self.lr, object_gt_points, 1.0, 0.5, self.warmup_step_num):
            print("[ERROR][Trainer::escalateTrainMash]")
            print("\t warmUpEpochObject failed!")
            return False
        
        print("[INFO][HOITrainer::escalateTrainMash]")
        print("\t start trainEpochObject...")
        if not self.trainEpochObject(self.lr, object_gt_points, 1.0, 0.5):
            print("[ERROR][Trainer::escalateTrainMash]")
            print("\t trainEpochObject failed!")
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
        
        # for hand
        print("[INFO][HOITrainer::escalateTrainMash]")
        print("\t start warmUpEpochHand...")
        if not self.warmUpEpochHand(self.lr, hand_gt_points, 1.0, 0.75, self.warmup_step_num):
            print("[ERROR][Trainer::escalateTrainMash]")
            print("\t warmUpEpochHand failed!")
            return False
        
        print("[INFO][HOITrainer::escalateTrainMash]")
        print("\t start trainEpochHand...")
        if not self.trainEpochHand(self.lr, hand_gt_points, 1.0, 0.75):
            print("[ERROR][Trainer::escalateTrainMash]")
            print("\t trainEpochHand failed!")
            return False

        # for object
        print("[INFO][HOITrainer::escalateTrainMash]")
        print("\t start warmUpEpochObject...")
        if not self.warmUpEpochObject(self.lr, object_gt_points, 1.0, 0.75, self.warmup_step_num):
            print("[ERROR][Trainer::escalateTrainMash]")
            print("\t warmUpEpochObject failed!")
            return False
        
        print("[INFO][HOITrainer::escalateTrainMash]")
        print("\t start trainEpochObject...")
        if not self.trainEpochObject(self.lr, object_gt_points, 1.0, 0.75):
            print("[ERROR][Trainer::escalateTrainMash]")
            print("\t trainEpochObject failed!")
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
        
        # for hand
        print("[INFO][HOITrainer::escalateTrainMash]")
        print("\t start warmUpEpochHand...")
        if not self.warmUpEpochHand(self.lr, hand_gt_points, 1.0, 1.0, self.warmup_step_num):
            print("[ERROR][Trainer::escalateTrainMash]")
            print("\t warmUpEpochHand failed!")
            return False
        
        print("[INFO][HOITrainer::escalateTrainMash]")
        print("\t start trainEpochHand...")
        if not self.trainEpochHand(self.lr, hand_gt_points, 1.0, 1.0):
            print("[ERROR][Trainer::escalateTrainMash]")
            print("\t trainEpochHand failed!")
            return False

        # for object
        print("[INFO][HOITrainer::escalateTrainMash]")
        print("\t start warmUpEpochObject...")
        if not self.warmUpEpochObject(self.lr, object_gt_points, 1.0, 1.0, self.warmup_step_num):
            print("[ERROR][Trainer::escalateTrainMash]")
            print("\t warmUpEpochObject failed!")
            return False
        
        print("[INFO][HOITrainer::escalateTrainMash]")
        print("\t start trainEpochObject...")
        if not self.trainEpochObject(self.lr, object_gt_points, 1.0, 1.0):
            print("[ERROR][Trainer::escalateTrainMash]")
            print("\t trainEpochObject failed!")
            return False

        cmap_gt = self.upSampleCMapGT()
        print("[INFO][HOITrainer::escalateTrainMash]")
        print("\t start trainEpochCMap...")
        cmap_loss = self.trainCMap(cmap_gt, 6000, self.lr)
        
        self.error = self.hand_error + self.object_error

        # training finished
        print("[INFO][Trainer::escalateTrainMash]")
        print("\t escalate train HOIMash finished!")
        print("\t total error: {:.5f}".format(self.error))
        print("\t hand_error: {:.5f}".format(self.hand_error))
        print("\t object_error: {:.5f}".format(self.object_error))
        print("\t cmap_loss: {:.5f}".format(cmap_loss))

        self.closeLog()

        return True
