import os
import numpy as np
import torch
import pytorch_kinematics as pk
from pytorch_kinematics.urdf_parser_py.urdf import URDF
import transforms3d
import trimesh
from tqdm import tqdm
import open3d as o3d
from typing import Dict, Optional, List, Tuple
from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import knn_points
import xml.etree.ElementTree as ET

from mash_hoi.source.Utils.mesh_utils import parse_origin, as_mesh, apply_transform, create_primitive_mesh, compute_rotation_matrix_from_ortho6d
from mash_hoi.source.Utils.key_points import BARRETT_KEY_POINTS
from mash_hoi.source.Utils.template_base_link_submesh_ids import BARRETT_BASE_LINK_SUBMESH_IDS


def visualize_fitter(
    src: torch.Tensor,              # [P * max_pts, 3] 
    tgt: torch.Tensor,              # [P * max_pts, 3]
    object_points: torch.Tensor,    # [max_obj, 3]
    contact_tensor: torch.Tensor | None = None,   # [max_obj]
    contact_threshold: float = 0.005,
) -> None:
    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = _to_o3d_vector3d(tgt.cpu().numpy())
    target_pcd.paint_uniform_color([1, 0, 0])

    if contact_tensor is not None:
        contact_tensor = contact_tensor.cpu().numpy()
    else:
        contact_tensor = np.zeros(object_points.shape[0])

    current_verts = src.cpu().numpy()
    current_pcd = o3d.geometry.PointCloud()
    current_pcd.points = _to_o3d_vector3d(current_verts)
    current_pcd.paint_uniform_color([0, 1, 0])

    obj_np = object_points.cpu().numpy()
    s_np = contact_tensor.cpu().numpy()
    colors = np.zeros((obj_np.shape[0], 3))
    colors[:, 0] = s_np
    colors[:, 2] = 1 - s_np
    colors[:, 1] = 0.2 * (1 - np.abs(s_np - 0.5) * 2)
    obj_vis = o3d.geometry.PointCloud()
    obj_vis.points = _to_o3d_vector3d(obj_np)
    obj_vis.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

    # Draw a yellow sphere at the origin with radius contact_threshold
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.5 * contact_threshold)
    sphere.translate([0, 0, 0])
    sphere.paint_uniform_color([1, 1, 0])

    geometries = [target_pcd, current_pcd]
    if obj_vis is not None:
        geometries.append(obj_vis)
    geometries.append(sphere)

    o3d.visualization.draw_geometries(geometries)


def _to_o3d_vector3d(points: np.ndarray) -> o3d.utility.Vector3dVector:
    """
    Safely convert the input array to an Open3D Vector3dVector:
    - Accept arrays of shape [N, 3] or [..., 3] and reshape them to [N, 3]
    - Convert the dtype to float64 (Open3D defaults to double)
    """
    arr = np.asarray(points)
    if arr.ndim == 3 and arr.shape[-1] == 3:
        arr = arr.reshape(-1, 3)
    elif arr.ndim == 2 and arr.shape[1] == 3:
        pass
    else:
        raise ValueError("Points must have shape [N, 3] or [..., 3] with last dim=3.")

    if arr.dtype != np.float64:
        arr = arr.astype(np.float64)
    return o3d.utility.Vector3dVector(arr)


def toCorrectTensor(tensor: torch.Tensor, device: str) -> torch.Tensor:
    if tensor.dtype != torch.float32:
        tensor = tensor.float()
    return tensor.to(device=device)


class BarrettHandFitter:
    def __init__(
        self,
        urdf_path: str,
        part_path: str,
        device: str,
        submesh_num: int = 256,
        points_per_submesh: int = 1024,
        barycentric_coords: Optional[np.ndarray] = None,
        triangle_ids: Optional[np.ndarray] = None,
        permutation: Optional[np.ndarray] = None,
        use_cmap: bool = True,
        use_joint_limits: bool = True,
        use_srf: bool = True,
    ) -> None:
        self.device = torch.device(device)
        self.urdf_path = urdf_path
        self.robot = pk.build_chain_from_urdf(open(urdf_path).read()).to(device=self.device, dtype=torch.float32)
        self.current_status: Dict[str, torch.Tensor] = {}
        self.mesh_verts: Dict[str, np.ndarray] = {}
        self.mesh_faces: Dict[str, np.ndarray] = {}
        self.visual = URDF.from_xml_string(open(urdf_path).read())
        self.part_path = part_path
        self.scale = 1.0
        R_x = trimesh.transformations.rotation_matrix(np.radians(90), [1, 0, 0])
        R_y = trimesh.transformations.rotation_matrix(np.radians(90), [0, 1, 0])
        self.R = np.dot(R_y, R_x)
        self.bse_link_face_ids = []
        self._prepare_mesh()

        self.use_cmap = use_cmap
        
        self.use_joint_limits = use_joint_limits
        self.use_srf = use_srf

        self.submesh_num = submesh_num
        self.points_per_submesh = points_per_submesh
        self.barycentric_coords = (
            torch.from_numpy(barycentric_coords).to(device=self.device, dtype=torch.float32)
            if barycentric_coords is not None else None
        )
        self.triangle_ids = (
            torch.from_numpy(triangle_ids).to(device=self.device, dtype=torch.int32)
            if triangle_ids is not None else None
        )
        self.permutation = (
            torch.from_numpy(permutation).to(device=self.device, dtype=torch.int32)
            if permutation is not None else None
        )
        self.inv_permutation = torch.argsort(self.permutation)
        self.joint_lower: torch.Tensor
        self.joint_upper: torch.Tensor
        lowers: List[float] = []
        uppers: List[float] = []
        for j in self.visual.joints:
            if getattr(j, "type", None) == "fixed":
                continue
            lim = getattr(j, "limit", None)
            if lim is None:
                lowers.append(float("-inf"))
                uppers.append(float("inf"))
            else:
                lo = lim.lower if lim.lower is not None else float("-inf")
                up = lim.upper if lim.upper is not None else float("inf")
                lowers.append(float(lo))
                uppers.append(float(up))
        if len(lowers) == 0:
            lowers = [float("-inf")] * 16
            uppers = [float("inf")] * 16
        self.joint_lower = torch.tensor(lowers, device=self.device, dtype=torch.float32)
        self.joint_upper = torch.tensor(uppers, device=self.device, dtype=torch.float32)

    def _prepare_mesh(self) -> bool:
        urdf_dir = os.path.dirname(self.urdf_path)
        tree = ET.parse(self.urdf_path)
        root = tree.getroot()
        
        for link in root.findall("link"):
            link_name = link.attrib["name"]
            if link_name in self.robot.get_link_names():
                geom_index = "visual"
                link_mesh = []
                for visual in link.findall(".//" + geom_index):
                    geometry = visual.find("geometry")
                    xyz, rotation = parse_origin(visual)
                    try:
                        if geometry[0].tag.endswith("mesh"):
                            mesh_filename = geometry[0].attrib["filename"]
                            full_mesh_path = os.path.join(urdf_dir, mesh_filename)
                            mesh = as_mesh(trimesh.load(full_mesh_path))
                            scale = np.fromstring(geometry[0].attrib.get("scale", "1 1 1"), sep=" ")
                            mesh.apply_scale(scale)
                            mesh = apply_transform(mesh, xyz, rotation)
                            link_mesh.append(mesh)
                        else:  # Handle primitive shapes
                            mesh = create_primitive_mesh(geometry[0], xyz, rotation)
                            scale = np.fromstring(geometry[0].attrib.get("scale", "1 1 1"), sep=" ")
                            mesh.apply_scale(scale)
                            link_mesh.append(mesh)
                    except Exception as e:
                        print(f"Failed to load geometry for {link_name}: {e}")
                if len(link_mesh) == 0:
                    continue
                elif len(link_mesh) > 1:
                    link_trimesh = as_mesh(trimesh.Scene(link_mesh))
                elif len(link_mesh) == 1:
                    link_trimesh = link_mesh[0]

            self.mesh_verts[link_name] = np.array(link_trimesh.vertices)
            self.mesh_faces[link_name] = np.array(link_trimesh.faces)
        
        if "bh_base_link" in self.mesh_faces:
            self.base_link_face_ids = list(range(self.mesh_faces["bh_base_link"].shape[0]))
        else:
            self.base_link_face_ids = []
        
        return True

    def get_meshes_from_joint(self, joint_angles: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Input:
            joint_angles: [B, 8]
        Output:
            all_vertices: [B, N, 3]
            all_faces: [F, 3]
        """
        B = joint_angles.shape[0]
        verts_list_b = []
        faces_list = []
        face_link_ids_list = []
        self.current_status = self.robot.forward_kinematics(torch.cat([torch.zeros(B, 6, device=self.device), joint_angles], axis=1))
        for i, link_name in enumerate(self.mesh_verts):
            if link_name in ['forearm']:
                continue
            trans_matrix = self.current_status[link_name].get_matrix()  # [B, 4, 4]

            v = torch.from_numpy(self.mesh_verts[link_name]).to(device=self.device, dtype=torch.float32)  # [V, 3]
            v_h = torch.concatenate([v, torch.ones([v.shape[0], 1], device=self.device)], axis=-1)  # [V, 4]
            v_h_b = v_h.unsqueeze(0).expand(B, -1, -1)  # [B, V, 4]
            transformed = torch.bmm(trans_matrix, v_h_b.transpose(1, 2))  # [B, 4, V]
            transformed_v = transformed.transpose(1, 2)[:, :, :3]  # [B, V, 3]
            transformed_v = transformed_v * self.scale
            f = torch.from_numpy(self.mesh_faces[link_name]).to(device=self.device, dtype=torch.int32)  # [F, 3]

            link_idx = len(verts_list_b)
            verts_list_b.append(transformed_v)
            faces_list.append(f)
            face_link_ids_list.append(torch.full((f.shape[0],), link_idx, device=self.device, dtype=torch.long))

        # Merge vertices and faces from all links
        vertex_offset = 0
        all_faces = []
        for i in range(len(faces_list)):
            all_faces.append(faces_list[i] + vertex_offset)
            vertex_offset += verts_list_b[i].shape[1]

        all_vertices = torch.cat(verts_list_b, dim=1)  # [B, sum(V_i), 3]
        all_faces = torch.cat(all_faces, dim=0)       # [sum(F_i), 3]
        self._face_link_ids = torch.cat(face_link_ids_list, dim=0)

        return all_vertices, all_faces

    def initialize_parameters(self, transform_inv: torch.Tensor) -> Tuple[torch.Tensor]:
        """
        Input:
            transform_inv: [B, 4, 4]
        Output:
            joint_angles: [B, 8]
            global_orient: [B, 3]
            global_transl: [B, 3]
            global_rot6d: [B, 6]
        """
        B = transform_inv.shape[0]
        joint_angles = torch.zeros(B, 8, device=self.device, requires_grad=True)
        # Broadcast the fixed rotation matrix across the batch dimension
        R_base = self.R
        R_base = torch.tensor(R_base[:3, :3], dtype=torch.float32, device=self.device)  # [3, 3]
        # Compute batched rotations: transform_inv[:, :3, :3] @ R_base
        R = torch.bmm(transform_inv[:, :3, :3], R_base.unsqueeze(0).expand(B, -1, -1))  # [B, 3, 3]
        R64 = R.double()
        euler_list = []
        for i in range(B):
            euler_i = transforms3d.euler.mat2euler(R64[i].detach().cpu().numpy(), axes='sxyz')
            euler_list.append(euler_i)
        global_orient = torch.tensor(euler_list, dtype=torch.float32, device=self.device)
        global_orient.requires_grad_(True)
        global_rot6d = torch.cat([R[:, :, 0], R[:, :, 1]], dim=1)
        global_rot6d.requires_grad_(True)
        # Get initial mesh vertices (batch support included)
        init_vertices, _ = self.get_meshes_from_joint(joint_angles)  # [B, N, 3]
        # Compute centers for the batch and add translations
        init_center = -torch.mean(init_vertices, dim=1) + transform_inv[:, :3, 3]  # [B, 3]
        global_transl = init_center.clone().detach().requires_grad_(True)
        return joint_angles, global_orient, global_transl, global_rot6d

    def estimate_global_rotation(
        self, 
        source_centers: torch.Tensor,   # [B, P, 3]
        target_centers: torch.Tensor    # [B, P, 3]
    ) -> torch.Tensor:                  # [B, 3, 3]
        source_centers = source_centers[:, BARRETT_BASE_LINK_SUBMESH_IDS, :]
        target_centers = target_centers[:, self.inv_permutation, :][:, BARRETT_BASE_LINK_SUBMESH_IDS, :]
        
        source_mean = torch.mean(source_centers, dim=1, keepdim=True)  # [B, 1, 3]
        target_mean = torch.mean(target_centers, dim=1, keepdim=True)  # [B, 1, 3]
        source_centers_centered = source_centers - source_mean  # [B, P, 3]
        target_centers_centered = target_centers - target_mean  # [B, P, 3]
        H = torch.bmm(source_centers_centered.transpose(1, 2), target_centers_centered)  # [B, 3, 3]
        U, S, V = torch.svd(H)
        R = torch.bmm(V, U.transpose(1, 2))
        det_R = torch.det(R)
        D = torch.eye(3, device=self.device, dtype=R.dtype).unsqueeze(0).repeat(R.shape[0], 1, 1)
        D[:, 2, 2] = torch.where(det_R < 0, -1.0, 1.0)
        R = torch.bmm(torch.bmm(V, D), U.transpose(1, 2))
        return R

    def fit(
        self,
        target_points: torch.Tensor,      # [B, P, N, 3]
        object_points: torch.Tensor,      # [B, P, N, 3]
        contact_strengths: torch.Tensor,  # [B, P, N]
        transform_inv: torch.Tensor,      # [B, 4, 4]
        object_normals: Optional[torch.Tensor] = None, # [B, P, N, 3]
        num_iterations: int = 1000,
        lr: float = 0.012,
        lambda_geom: float = 1.0,
        lambda_contact: float = 2.0,
        lambda_reg: float = 0.1,
        lambda_limits: float = 1.0,
        lambda_natural: float = 0.1,
        lambda_srf: float = 0.5,
        lambda_spf: float = 0.0,
        lambda_erf: float = 0.0,
        d_threshold: float = 0.015,
        contact_threshold: float = 0.005,
        verbose: bool = True,
        vis: bool = False,
        vis_id: int = 0,
        stop_error: float = 0.006,
        reg_start_iter: int = 200,
    ) -> Dict[str, np.ndarray]:
        B, P, N, _ = target_points.shape

        assert vis_id < B

        target_points_tensor = toCorrectTensor(target_points, self.device)          # [B, P, N, 3]
        object_points_tensor = toCorrectTensor(object_points, self.device)          # [B, P, N, 3]
        object_centers = torch.mean(object_points_tensor.view(B, -1, 3), dim=1)     # [B, 3]
        contact_strengths_tensor = toCorrectTensor(contact_strengths, self.device)  # [B, P, N]
        transform_inv_tensor = toCorrectTensor(transform_inv, self.device)          # [B, 4, 4]
        
        rotation = transform_inv_tensor[:, :3, :3].transpose(1, 2)  # [B, 3, 3]
        translation = transform_inv_tensor[:, :3, 3]                # [B, 3]

        target_points_tensor = torch.bmm(target_points_tensor.view(B, -1, 3), rotation) + translation.unsqueeze(1)  # [B, NP, 3]
        object_points_tensor = torch.bmm(object_points_tensor.view(B, -1, 3), rotation) + translation.unsqueeze(1)  # [B, NP, 3]
        if object_normals is not None:
            object_normals_tensor = toCorrectTensor(object_normals, self.device)
            object_normals_tensor = torch.bmm(object_normals_tensor.view(B, -1, 3), rotation) # [B, NP, 3]
        else:
            object_normals_tensor = None

        joint_angles, global_orient, global_transl, global_rot6d = self.initialize_parameters(transform_inv_tensor)

        target_centers = torch.mean(target_points_tensor.view(B, P, N, 3), dim=2)  # [B, P, 3]
        all_vertices, all_faces = self.get_meshes_from_joint(joint_angles)  # [B, V, 3], [F, 3]
        # all_faces: [F, 3], triangle_ids: [PN] -> face_indices: [PN, 3]
        face_indices = all_faces[self.triangle_ids]  # [PN, 3]
        # all_vertices: [B, V, 3]
        batch_indices = torch.arange(B, device=self.device).view(B, 1, 1).expand(B, self.triangle_ids.shape[0], 3)  # [B, PN, 3]
        triangle_verts = all_vertices[batch_indices, face_indices]  # [B, PN, 3, 3]

        # barycentric_coords: [PN, 3] -> [1, PN, 3, 1]
        weights = self.barycentric_coords.unsqueeze(0).unsqueeze(-1)    # [1, PN, 3, 1]
        sampled_points = torch.sum(triangle_verts * weights, dim=2)     # [B, PN, 3]

        source_centers = torch.mean(sampled_points.view(B, P, N, 3), dim=2)  # [B, P, 3]
                
        if self.permutation is not None:
            sampled_points = sampled_points.reshape(B, P, N, 3)[:, self.permutation, :, :]

        estimated_global_rotation = self.estimate_global_rotation(source_centers, target_centers)  # [B, 3, 3]

        # estimated_global_translation = 2.0 * (torch.mean(target_centers, dim=1) - object_centers)
        estimated_global_translation = torch.bmm(torch.mean(target_centers, dim=1).unsqueeze(1), estimated_global_rotation) - torch.mean(source_centers, dim=1).unsqueeze(1)  # [B, 1, 3]
        estimated_global_translation = estimated_global_translation.squeeze(1)  # [B, 3]

        global_rot6d = (
            torch.cat([estimated_global_rotation[:, :, 0], estimated_global_rotation[:, :, 1]], dim=1)
            .clone()
            .detach()
            .requires_grad_(True)
        )
        global_transl = estimated_global_translation.clone().detach().requires_grad_(True)


        optimizer = torch.optim.Adam([
            {"params": joint_angles, "lr": lr},         # [B, 24]
            {"params": global_orient, "lr": lr},        # [B, 3]
            {"params": global_rot6d, "lr": lr},         # [B, 6]
            {"params": global_transl, "lr": lr * 10},   # [B, 3]
        ])
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            total_steps=num_iterations,
            pct_start=0.1
        )

        K = len(BARRETT_KEY_POINTS)
        first_point_ids = [pair[0] for pair in BARRETT_KEY_POINTS]
        second_point_ids = [pair[1] for pair in BARRETT_KEY_POINTS]

        pbar = tqdm(range(num_iterations))
        for iteration in pbar:
            optimizer.zero_grad()
            
            all_vertices, all_faces = self.get_meshes_from_joint(joint_angles)  # [B, V, 3], [F, 3]
            Rm = compute_rotation_matrix_from_ortho6d(global_rot6d)             # [B, 3, 3]
            all_vertices = torch.bmm(all_vertices + global_transl.unsqueeze(1), Rm.transpose(1, 2)) # [B, V, 3]
            
            key_points = 0.5 * (all_vertices[:, first_point_ids, :] + all_vertices[:, second_point_ids, :])

            # all_faces: [F, 3], triangle_ids: [PN] -> face_indices: [PN, 3]
            face_indices = all_faces[self.triangle_ids]  # [PN, 3]

            # all_vertices: [B, V, 3]
            batch_indices = torch.arange(B, device=self.device).view(B, 1, 1).expand(B, self.triangle_ids.shape[0], 3)  # [B, PN, 3]
            triangle_verts = all_vertices[batch_indices, face_indices]  # [B, PN, 3, 3]

            # barycentric_coords: [PN, 3] -> [1, PN, 3, 1]
            weights = self.barycentric_coords.unsqueeze(0).unsqueeze(-1)    # [1, PN, 3, 1]
            sampled_points = torch.sum(triangle_verts * weights, dim=2)     # [B, PN, 3]
            if self.permutation is not None:
                sampled_points = sampled_points.reshape(B, P, N, 3)[:, self.permutation, :, :]
            sampled_points = sampled_points.reshape(B*P, N, 3)              # [BP, N, 3]

            src = sampled_points
            tgt = target_points_tensor.reshape(B*P, N, 3)
            max_pts = 32
            if src.shape[1] > max_pts:
                perm_src = torch.randperm(src.shape[1], device=self.device)[:max_pts]
                src = src[:, perm_src, :]
            if tgt.shape[1] > max_pts:
                perm_tgt = torch.randperm(tgt.shape[1], device=self.device)[:max_pts]
                tgt = tgt[:, perm_tgt, :]

            chamfer_dist, _ = chamfer_distance(
                src,
                tgt,
                norm=1
            )
            
            obj = object_points_tensor
            obj_normals = object_normals_tensor
            s = contact_strengths_tensor.reshape(B, -1) if self.use_cmap else None
            max_obj = 5000
            if obj.shape[1] > max_obj:
                perm_obj = torch.randperm(obj.shape[1], device=self.device)[:max_obj]
                obj = obj[:, perm_obj, :]
                if s is not None:
                    s = s[:, perm_obj]
                if obj_normals is not None:
                    obj_normals = obj_normals[:, perm_obj, :]
            if not self.use_cmap:
                s = torch.zeros(B, obj.shape[1], device=self.device)
                contact_loss = torch.tensor(0.0, device=self.device)
                contact_weight = 0.0
            else:
                if iteration < reg_start_iter:
                    contact_loss = torch.tensor(0.0, device=self.device)
                    contact_weight = 0.0
                else:
                    d2_oh, _, _ = knn_points(obj, src.reshape(B, -1, 3), K=1, return_nn=False)
                    d2 = d2_oh.squeeze(-1)
                    d = torch.sqrt(torch.clamp(d2, min=1e-12))
                    contact_curr = 1.0 - 2.0 * (torch.sigmoid(100 * d) - 0.5)
                    contact_loss = torch.mean((contact_curr - s) ** 2)
                    effective_steps = max(num_iterations - reg_start_iter, 1)
                    contact_weight = lambda_contact * min((iteration - reg_start_iter) / effective_steps, 1.0)

            if self.use_srf:
                K = key_points.shape[1]
                idx_i, idx_j = torch.triu_indices(K, K, offset=1, device=self.device)
                dmat = torch.cdist(key_points, key_points, p=2)
                dij = dmat[:, idx_i, idx_j]
                srf_loss = torch.relu(d_threshold - dij).mean()
            else:
                srf_loss = torch.tensor(0.0, device=self.device)

            if iteration < reg_start_iter:
                spf_loss = torch.tensor(0.0, device=self.device)
                spf_weight = 0.0
            else:
                hand_points_flat = sampled_points.reshape(B, -1, 3)
                d_ho, _, _ = knn_points(hand_points_flat, obj, K=1, norm=1)
                
                mask_spf = d_ho < contact_threshold
                if mask_spf.any():
                    spf_loss = d_ho[mask_spf].mean()
                else:
                    spf_loss = torch.tensor(0.0, device=self.device)
                
                effective_steps = max(num_iterations - reg_start_iter, 1)
                spf_weight = lambda_spf * min((iteration - reg_start_iter) / effective_steps, 1.0)

            # ERF Logic
            if lambda_erf > 0 and obj_normals is not None:
                hand_points_flat = sampled_points.reshape(B, -1, 3)
                d2_ho, idx_ho, _ = knn_points(hand_points_flat, obj, K=1)
                idx_ho = idx_ho.squeeze(-1)
                
                batch_indices = torch.arange(B, device=self.device).view(B, 1).expand(B, idx_ho.shape[1])
                nearest_normals = obj_normals[batch_indices, idx_ho]
                nearest_points = obj[batch_indices, idx_ho]
                
                vec = hand_points_flat - nearest_points
                dot = (vec * nearest_normals).sum(dim=-1)
                
                mask_erf = dot < 0
                if mask_erf.any():
                    d_ho_val = torch.sqrt(d2_ho).squeeze(-1)
                    erf_loss = d_ho_val[mask_erf].mean()
                else:
                    erf_loss = torch.tensor(0.0, device=self.device)
            else:
                erf_loss = torch.tensor(0.0, device=self.device)

            D = joint_angles.shape[1]
            if self.use_joint_limits and self.joint_lower.numel() >= D:    
                lower = self.joint_lower[6:].unsqueeze(0)
                upper = self.joint_upper[6:].unsqueeze(0)
                lim_pen = torch.relu(joint_angles - upper) + torch.relu(lower - joint_angles)
                reg_limits = lim_pen.mean()                
            else:
                reg_limits = torch.tensor(0.0, device=self.device)

            reg_natural = (joint_angles ** 2).mean()
            reg_loss = lambda_limits * reg_limits + lambda_natural * reg_natural
            loss = (
                lambda_geom * chamfer_dist
                + contact_weight * contact_loss
                + lambda_reg * reg_loss
                + lambda_srf * srf_loss
                + spf_weight * spf_loss
                + lambda_erf * erf_loss
            )

            loss.backward()
            optimizer.step()
            scheduler.step()

            if verbose:
                pbar.set_postfix({
                    "cd": f'{chamfer_dist.item():.8f}',
                    "contact": f'{contact_loss.item():.8f}',
                    "srf": f'{srf_loss.item():.8f}',
                    "spf": f'{spf_loss.item():.8f}',
                    "erf": f'{erf_loss.item():.8f}',
                    "limits": f'{reg_limits.item():.8f}',
                    "natural": f'{reg_natural.item():.8f}',
                    "loss": f'{loss.item():.8f}',
                })
            
            if vis and iteration % 100 == 0:
                with torch.no_grad():
                    try:                        
                        visualize_fitter(src=src.view(B, -1, 3)[vis_id], tgt=tgt.view(B, -1, 3)[vis_id], object_points=obj[vis_id], contact_tensor=s[vis_id], contact_threshold=contact_threshold)
                    except Exception as e:
                        if verbose:
                            print(f"Open3D visualization failed: {e}")
            
            if chamfer_dist.item() < stop_error:
                break
        
        if vis:
            with torch.no_grad():
                visualize_fitter(src=src.view(B, -1, 3)[vis_id], tgt=tgt.view(B, -1, 3)[vis_id], object_points=obj[vis_id], contact_tensor=s[vis_id], contact_threshold=contact_threshold)

        hand_verts = all_vertices.detach().cpu().numpy()    # [B, V, 3]
        hand_faces = all_faces.cpu().numpy()                # [F, 3]
        R_batch = compute_rotation_matrix_from_ortho6d(global_rot6d).detach().cpu().numpy()
        ga_list = []
        for i in range(B):
            axis, angle = transforms3d.axangles.mat2axangle(R_batch[i])
            ga_list.append(axis * angle)
        global_axis_angle = np.asarray(ga_list, dtype=np.float32)
        
        return {
            'joint_angles': joint_angles.detach().cpu().numpy(),    # [B, 24]
            'global_orient': global_axis_angle,                     # [B, 3]
            'global_transl': global_transl.detach().cpu().numpy(),  # [B, 3]
            'global_rot6d': global_rot6d.detach().cpu().numpy(),    # [B, 6]
            'rot_mat': R_batch,                                     # [B, 3, 3]
            'hand_verts': hand_verts,                               # [B, N, 3]
            'hand_faces': hand_faces,                               # [F, 3]
            'cd_loss': chamfer_dist.item(),
        }
    
    def pure_geom_fit(
        self,
        target_points: torch.Tensor,      # [B, P, N, 3]
        object_points: torch.Tensor,      # [B, P, N, 3]
        transform_inv: torch.Tensor,      # [B, 4, 4]
        num_iterations: int = 1000,
        lr: float = 0.012,
        verbose: bool = True,
        vis: bool = False,
        vis_id: int = 0,
        stop_error: float = 0.006,
    ) -> Dict[str, np.ndarray]:
        B, P, N, _ = target_points.shape

        assert vis_id < B

        target_points_tensor = toCorrectTensor(target_points, self.device)          # [B, P, N, 3]
        object_points_tensor = toCorrectTensor(object_points, self.device)          # [B, P, N, 3]
        transform_inv_tensor = toCorrectTensor(transform_inv, self.device)          # [B, 4, 4]
        
        rotation = transform_inv_tensor[:, :3, :3].transpose(1, 2)  # [B, 3, 3]
        translation = transform_inv_tensor[:, :3, 3]                # [B, 3]

        target_points_tensor = torch.bmm(target_points_tensor.view(B, -1, 3), rotation) + translation.unsqueeze(1)  # [B, NP, 3]
        object_points_tensor = torch.bmm(object_points_tensor.view(B, -1, 3), rotation) + translation.unsqueeze(1)  # [B, NP, 3]

        joint_angles, global_orient, global_transl, global_rot6d = self.initialize_parameters(transform_inv_tensor)

        target_centers = torch.mean(target_points_tensor.view(B, P, N, 3), dim=2)  # [B, P, 3]
        all_vertices, all_faces = self.get_meshes_from_joint(joint_angles)  # [B, V, 3], [F, 3]
        # all_faces: [F, 3], triangle_ids: [PN] -> face_indices: [PN, 3]
        face_indices = all_faces[self.triangle_ids]  # [PN, 3]
        # all_vertices: [B, V, 3]
        batch_indices = torch.arange(B, device=self.device).view(B, 1, 1).expand(B, self.triangle_ids.shape[0], 3)  # [B, PN, 3]
        triangle_verts = all_vertices[batch_indices, face_indices]  # [B, PN, 3, 3]

        # barycentric_coords: [PN, 3] -> [1, PN, 3, 1]
        weights = self.barycentric_coords.unsqueeze(0).unsqueeze(-1)    # [1, PN, 3, 1]
        sampled_points = torch.sum(triangle_verts * weights, dim=2)     # [B, PN, 3]
        if self.permutation is not None:
            sampled_points = sampled_points.reshape(B, P, N, 3)[:, self.permutation, :, :]
        
        source_centers = torch.mean(sampled_points, dim=2)  # [B, P, 3]

        estimated_global_rotation = self.estimate_global_rotation(source_centers, target_centers)  # [B, 3, 3]

        estimated_global_translation = torch.bmm(torch.mean(target_centers, dim=1).unsqueeze(1), estimated_global_rotation) - torch.mean(source_centers, dim=1).unsqueeze(1)  # [B, 1, 3]
        estimated_global_translation = estimated_global_translation.squeeze(1)  # [B, 3]

        global_rot6d = (
            torch.cat([estimated_global_rotation[:, :, 0], estimated_global_rotation[:, :, 1]], dim=1)
            .clone()
            .detach()
            .requires_grad_(True)
        )
        global_transl = estimated_global_translation.clone().detach().requires_grad_(True)
        

        optimizer = torch.optim.Adam([
            {"params": joint_angles, "lr": lr},         # [B, 24]
            {"params": global_orient, "lr": lr},        # [B, 3]
            {"params": global_rot6d, "lr": lr},         # [B, 6]
            {"params": global_transl, "lr": lr * 10},   # [B, 3]
        ])
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            total_steps=num_iterations,
            pct_start=0.1
        )

        pbar = tqdm(range(num_iterations))
        for iteration in pbar:
            optimizer.zero_grad()
            
            all_vertices, all_faces = self.get_meshes_from_joint(joint_angles)  # [B, V, 3], [F, 3]
            Rm = compute_rotation_matrix_from_ortho6d(global_rot6d)             # [B, 3, 3]
            all_vertices = torch.bmm(all_vertices + global_transl.unsqueeze(1), Rm.transpose(1, 2)) # [B, V, 3]
            
            # all_faces: [F, 3], triangle_ids: [PN] -> face_indices: [PN, 3]
            face_indices = all_faces[self.triangle_ids]  # [PN, 3]

            # all_vertices: [B, V, 3]
            batch_indices = torch.arange(B, device=self.device).view(B, 1, 1).expand(B, self.triangle_ids.shape[0], 3)  # [B, PN, 3]
            triangle_verts = all_vertices[batch_indices, face_indices]  # [B, PN, 3, 3]

            # barycentric_coords: [PN, 3] -> [1, PN, 3, 1]
            weights = self.barycentric_coords.unsqueeze(0).unsqueeze(-1)    # [1, PN, 3, 1]
            sampled_points = torch.sum(triangle_verts * weights, dim=2)     # [B, PN, 3]
            if self.permutation is not None:
                sampled_points = sampled_points.reshape(B, P, N, 3)[:, self.permutation, :, :]
            sampled_points = sampled_points.reshape(B*P, N, 3)              # [BP, N, 3]

            src = sampled_points
            tgt = target_points_tensor.reshape(B*P, N, 3)
            max_pts = 512
            if src.shape[1] > max_pts:
                perm_src = torch.randperm(src.shape[1], device=self.device)[:max_pts]
                src = src[:, perm_src, :]
            if tgt.shape[1] > max_pts:
                perm_tgt = torch.randperm(tgt.shape[1], device=self.device)[:max_pts]
                tgt = tgt[:, perm_tgt, :]

            chamfer_dist, _ = chamfer_distance(
                src,
                tgt,
                norm=1
            )
            
            obj = object_points_tensor
            
            loss = chamfer_dist
            loss.backward()
            optimizer.step()
            scheduler.step()

            if verbose:
                pbar.set_postfix({
                    "cd": f'{chamfer_dist.item():.8f}',
                })
            
            if vis and iteration % 100 == 0:
                with torch.no_grad():
                    try:                        
                        visualize_fitter(src=src.view(B, -1, 3)[vis_id], tgt=tgt.view(B, -1, 3)[vis_id], object_points=obj[vis_id])
                    except Exception as e:
                        if verbose:
                            print(f"Open3D visualization failed: {e}")
            
            if chamfer_dist.item() < stop_error:
                break
        
        if vis:
            with torch.no_grad():
                visualize_fitter(src=src.view(B, -1, 3)[vis_id], tgt=tgt.view(B, -1, 3)[vis_id], object_points=obj[vis_id])

        hand_verts = all_vertices.detach().cpu().numpy()    # [B, V, 3]
        hand_faces = all_faces.cpu().numpy()                # [F, 3]
        R_batch = compute_rotation_matrix_from_ortho6d(global_rot6d).detach().cpu().numpy()
        ga_list = []
        for i in range(B):
            axis, angle = transforms3d.axangles.mat2axangle(R_batch[i])
            ga_list.append(axis * angle)
        global_axis_angle = np.asarray(ga_list, dtype=np.float32)
        
        return {
            'joint_angles': joint_angles.detach().cpu().numpy(),    # [B, 24]
            'global_orient': global_axis_angle,                     # [B, 3]
            'global_transl': global_transl.detach().cpu().numpy(),  # [B, 3]
            'global_rot6d': global_rot6d.detach().cpu().numpy(),    # [B, 6]
            'rot_mat': R_batch,                                     # [B, 3, 3]
            'hand_verts': hand_verts,                               # [B, N, 3]
            'hand_faces': hand_faces,                               # [F, 3]
            'cd_loss': chamfer_dist.item(),
        }

