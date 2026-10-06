import sys
import os
import numpy as np
import torch
import transforms3d
import trimesh
from tqdm import tqdm
import open3d as o3d
from typing import Dict, Optional, List, Tuple
from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import knn_points
import trimesh


from ma_sh.Model import HOIMash
from mash_hoi.source.Utils.mesh_utils import compute_rotation_matrix_from_ortho6d


LAST_FACES = np.array(
    [[ 92,  38, 122],
     [234,  92, 122],
     [239, 234, 122],
     [279, 239, 122],
     [215, 279, 122],
     [215, 122, 118],
     [215, 118, 117],
     [215, 117, 119],
     [215, 119, 120],
     [215, 120, 108],
     [215, 108,  79],
     [215,  79,  78],
     [215,  78, 121],
     [214, 215, 121]]
)


def _patch_numpy_legacy_aliases() -> None:
    legacy = {
        "bool": bool,
        "int": int,
        "float": float,
        "complex": complex,
        "object": object,
        "str": str,
    }
    for name, value in legacy.items():
        if not hasattr(np, name):
            setattr(np, name, value)


def _ensure_manopth_importable(manopth_path: str) -> None:
    _patch_numpy_legacy_aliases()
    if manopth_path not in sys.path:
        sys.path.insert(0, manopth_path)
    try:
        import manopth.manolayer
        return
    except Exception:
        pass
    to_delete = [k for k in sys.modules.keys() if k == "manopth" or k.startswith("manopth.")]
    for k in to_delete:
        del sys.modules[k]
    if manopth_path not in sys.path:
        sys.path.insert(0, manopth_path)
    import importlib

    importlib.invalidate_caches()
    import manopth.manolayer


def visualize_fitter(
    src: torch.Tensor,              # [P * max_pts, 3] 
    tgt: torch.Tensor,              # [P * max_pts, 3]
    object_points: torch.Tensor,    # [max_obj, 3]
    contact_tensor: torch.Tensor | None = None,   # [max_obj]
    contact_threshold: float = 0.005,
) -> None:
    if contact_tensor is None:
        contact_tensor = torch.zeros(object_points.shape[0])
    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = _to_o3d_vector3d(tgt.cpu().numpy())
    target_pcd.paint_uniform_color([1, 0, 0])

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

    # Draw a yellow sphere at the origin with diameter contact_threshold
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.5 * contact_threshold)
    sphere.translate([0, 0, 0])
    sphere.paint_uniform_color([1, 1, 0])

    # Place a small red sphere at the src center
    src_center = src.mean(dim=0).cpu().numpy()
    src_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
    src_sphere.translate(src_center)
    src_sphere.paint_uniform_color([0, 1, 0])

    # Place a small red sphere at the tgt center
    tgt_center = tgt.mean(dim=0).cpu().numpy()
    tgt_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
    tgt_sphere.translate(tgt_center)
    tgt_sphere.paint_uniform_color([1, 0, 0])

    geometries = [target_pcd, current_pcd]
    if obj_vis is not None:
        geometries.append(obj_vis)
    geometries.extend([sphere, src_sphere, tgt_sphere])

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


class ManoHandFitter:
    def __init__(
        self,
        manopth_path: str = "/data/yxz/MASH-HOI/MASH/manopth",
        mano_root: Optional[str] = None,
        device: str = "cuda",
        submesh_num: int = 256,
        points_per_submesh: int = 1024,
        barycentric_coords: Optional[np.ndarray] = None,
        triangle_ids: Optional[np.ndarray] = None,
        permutation: Optional[np.ndarray] = None,
        use_cmap: bool = True,
        use_joint_limits: bool = True,
        use_srf: bool = True,
        batch_size: int = 64,
    ) -> None:
        self.device = torch.device(device)
        _ensure_manopth_importable(manopth_path)

        from manopth.manolayer import ManoLayer

        if mano_root is None:
            mano_root = os.path.join(manopth_path, "mano", "models")

        self.mano = ManoLayer(
            mano_root=mano_root,
            side="right",
            use_pca=False,
            ncomps=45,
            flat_hand_mean=True,
            root_rot_mode="axisang",
            joint_rot_mode="axisang",
        ).to(device=self.device)

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
        self.batch_size = batch_size
        self.faces = torch.from_numpy(np.concatenate([self.mano.th_faces.cpu().numpy(), LAST_FACES], axis=0)).to(device=self.device, dtype=torch.int32)
    
    def initialize_parameters(self) -> Tuple[torch.Tensor]:
        B = self.batch_size
        mano_parameters = torch.zeros(B, 48, device=self.device, requires_grad=True)
        global_transl = torch.zeros(B, 3, device=self.device, requires_grad=True)
        return mano_parameters, global_transl
    
    def estimate_global_rotation(
        self,
        source_centers: torch.Tensor,   # [B, P, 3]
        target_centers: torch.Tensor    # [B, P, 3]
    ) -> torch.Tensor:                  # [B, 3, 3]
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
        lambda_spf: float = 0.0,
        lambda_erf: float = 0.0,
        lambda_reg: float = 0.0,
        contact_threshold: float = 0.005,
        verbose: bool = True,
        vis: bool = False,
        vis_id: int = 0,
        stop_error: float = 0.006,
        reg_start_iter: int = 100,
    ) -> Dict[str, np.ndarray]:
        B, P, N, _ = target_points.shape

        assert vis_id < B

        target_points_tensor = toCorrectTensor(target_points, self.device)          # [B, P, N, 3]
        object_points_tensor = toCorrectTensor(object_points, self.device)          # [B, P, N, 3]
        contact_strengths_tensor = toCorrectTensor(contact_strengths, self.device)  # [B, P, N]
        transform_inv_tensor = toCorrectTensor(transform_inv, self.device)          # [B, 4, 4]
        
        if object_normals is not None:
            object_normals_tensor = toCorrectTensor(object_normals, self.device)    # [B, P, N, 3]
        else:
            object_normals_tensor = None

        rotation = transform_inv_tensor[:, :3, :3].transpose(1, 2)  # [B, 3, 3]
        translation = transform_inv_tensor[:, :3, 3]                # [B, 3]

        target_points_tensor = torch.bmm(target_points_tensor.view(B, -1, 3), rotation) + translation.unsqueeze(1)  # [B, NP, 3]
        object_points_tensor = torch.bmm(object_points_tensor.view(B, -1, 3), rotation) + translation.unsqueeze(1)  # [B, NP, 3]
        
        if object_normals_tensor is not None:
             # Normals only rotate
            object_normals_tensor = torch.bmm(object_normals_tensor.view(B, -1, 3), rotation) # [B, NP, 3]
        
        mano_params, global_transl = self.initialize_parameters()   # [B, 45] [B, 3]

        target_centers = torch.mean(target_points_tensor.view(B, P, N, 3), dim=2)  # [B, P, 3]

        all_vertices, all_joints = self.mano(mano_params)  # [B, V, 3], [F, 3]
        
        # self.faces: [F, 3], triangle_ids: [PN] -> face_indices: [PN, 3]
        face_indices = self.faces[self.triangle_ids]  # [PN, 3]
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
        estimated_global_translation = torch.mean(target_centers - source_centers, dim=1)  # [B, 3]

        ga_list = []
        R_batch = estimated_global_rotation.detach().cpu().numpy()
        for i in range(B):
            axis, angle = transforms3d.axangles.mat2axangle(R_batch[i])
            ga_list.append(axis * angle)
        estimated_global_axis_angle = torch.from_numpy(np.asarray(ga_list, dtype=np.float32)).to(
            device=self.device, dtype=torch.float32
        )

        mano_params = mano_params.detach().clone()
        mano_params[:, :3] = estimated_global_axis_angle
        mano_params.requires_grad_(True)

        global_transl = estimated_global_translation.detach().clone()
        global_transl.requires_grad_(True)
        scale = torch.ones(self.batch_size, device=self.device, requires_grad=True)

        optimizer = torch.optim.Adam([
            {'params': mano_params, 'lr': lr},
            {'params': global_transl, 'lr': lr * 10},
            {'params': scale, 'lr': lr}
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

            all_vertices, all_joints = self.mano(mano_params)
            center = all_vertices.mean(dim=1, keepdim=True)
            all_vertices = (all_vertices - center) * scale.unsqueeze(1).unsqueeze(1) + center + global_transl.unsqueeze(1)
            # self.faces: [F, 3], triangle_ids: [PN] -> face_indices: [PN, 3]
            face_indices = self.faces[self.triangle_ids]  # [PN, 3]
            # all_vertices: [B, V, 3]
            batch_indices = torch.arange(B, device=self.device).view(B, 1, 1).expand(B, self.triangle_ids.shape[0], 3)  # [B, PN, 3]
            triangle_verts = all_vertices[batch_indices, face_indices]  # [B, PN, 3, 3]

            # barycentric_coords: [PN, 3] -> [1, PN, 3, 1]
            weights = self.barycentric_coords.unsqueeze(0).unsqueeze(-1)    # [1, PN, 3, 1]
            sampled_points = torch.sum(triangle_verts * weights, dim=2)     # [B, PN, 3]
            if self.permutation is not None:
                sampled_points = sampled_points.reshape(B, P, N, 3)[:, self.permutation, :, :]
            sampled_points = sampled_points.reshape(B*P, N, 3)

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
            
            if lambda_reg > 0:
                reg_loss = torch.mean(mano_params ** 2)
            else:
                reg_loss = torch.tensor(0.0, device=self.device)

            if iteration < reg_start_iter:
                spf_loss = torch.tensor(0.0, device=self.device)
                spf_weight = 0.0
                erf_loss = torch.tensor(0.0, device=self.device)
            else:
                hand_points_flat = sampled_points.reshape(B, -1, 3)
                d2_ho, idx_ho, _ = knn_points(hand_points_flat, obj, K=1)
                d_ho = torch.sqrt(d2_ho)
                
                mask_spf = d_ho < contact_threshold
                if mask_spf.any():
                    spf_loss = d_ho[mask_spf].mean()
                else:
                    spf_loss = torch.tensor(0.0, device=self.device)
                
                effective_steps = max(num_iterations - reg_start_iter, 1)
                spf_weight = lambda_spf * min((iteration - reg_start_iter) / effective_steps, 1.0)

                # ERF Logic
                if lambda_erf > 0 and obj_normals is not None:
                    idx_ho = idx_ho.squeeze(-1) # [B, P]
                    batch_indices = torch.arange(B, device=self.device).view(B, 1).expand(B, idx_ho.shape[1])
                    nearest_normals = obj_normals[batch_indices, idx_ho] # [B, P, 3]
                    nearest_points = obj[batch_indices, idx_ho] # [B, P, 3]
                    
                    vec = hand_points_flat - nearest_points # [B, P, 3]
                    dot = (vec * nearest_normals).sum(dim=-1) # [B, P]
                    
                    mask_erf = dot < 0
                    if mask_erf.any():
                        # Penalize distance for penetrating points
                        # Penalize distance for penetrating points
                        erf_loss = d_ho.squeeze(-1)[mask_erf].mean()
                    else:
                        erf_loss = torch.tensor(0.0, device=self.device)
                else:
                    erf_loss = torch.tensor(0.0, device=self.device)

            loss = (
                lambda_geom * chamfer_dist
                + contact_weight * contact_loss
                + spf_weight * spf_loss
                + lambda_erf * erf_loss
                + lambda_reg * reg_loss
            )

            loss.backward()
            optimizer.step()
            scheduler.step()

            if verbose:
                pbar.set_description(
                    f"Loss: {loss.item():.4f} "
                    f"Chamfer: {chamfer_dist.item():.4f} "
                    f"Contact: {contact_loss.item():.4f} "
                    f"SPF: {spf_loss.item():.4f} "
                    f"ERF: {erf_loss.item():.4f}"
                    f"Reg: {reg_loss.item():.4f}"
                )

            if vis and iteration % 100 == 0:
                with torch.no_grad():
                    # try:
                    visualize_fitter(src=src.view(B, -1, 3)[vis_id], tgt=tgt.view(B, -1, 3)[vis_id], object_points=obj[vis_id], contact_tensor=s[vis_id])
                    # except Exception as e:
                    #     if verbose:
                    #         print(f"Visualization error: {e}")
            
            if chamfer_dist.item() < stop_error:
                break
    
        if vis:
            with torch.no_grad():
                visualize_fitter(src=src.view(B, -1, 3)[vis_id], tgt=tgt.view(B, -1, 3)[vis_id], object_points=obj[vis_id], contact_tensor=s[vis_id])
        
        all_vertices, all_joints = self.mano(mano_params)
        center = all_vertices.mean(dim=1, keepdim=True)
        all_vertices = (all_vertices - center) * scale.unsqueeze(1).unsqueeze(1) + center + global_transl.unsqueeze(1)
        hand_verts = all_vertices.detach().cpu().numpy()
        hand_faces = self.faces.cpu().numpy()

        return {
            'hand_verts': hand_verts,
            'hand_faces': hand_faces,
            'mano_params': mano_params.detach().cpu().numpy(),
        }
