import math
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Dict, List
import fcl
from pytorch3d.ops import knn_points
from pytorch3d.ops import estimate_pointcloud_normals
import signal
from multiprocessing import get_context

import mash_cpp
from ma_sh.Config.constant import W0
from ma_sh.Method.rotate import compute_rotation_matrix_from_ortho6d


def get_masked_stats(stats: Dict[str, torch.Tensor], mask: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Get masked statistics from a dictionary of tensors.
    
    Args:
        stats: Dictionary of tensors with shape (batch_size, ...)
        mask: Boolean tensor with shape (batch_size,)
        
    Returns:
        Dictionary of masked tensors with shape (sum(mask), ...)
    """
    return {k: v[mask] for k, v in stats.items()}


# ============================================================================
# Simple multiprocessing worker for collision detection
# ============================================================================

def simple_collision_worker(args):
    """
    Simple worker function for collision detection.
    Creates FCL objects from scratch for each sample.
    
    Args:
        args: tuple of (batch_idx, hand_center, hand_rotation, hand_extent,
                       object_center, object_rotation, object_extent,
                       include_self_intersection, anchor_num)
    
    Returns:
        tuple of (batch_idx, hand_object_pairs, hand_self_pairs)
    """
    # Disable signal handlers in worker process to avoid PyTorch Lightning's CUDA signal handler
    # This allows fork() to work in DDP environments without CUDA re-initialization errors
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    
    (batch_idx, hand_center_np, hand_rotation_np, hand_extent_np,
     object_center_np, object_rotation_np, object_extent_np,
     include_self_intersection, anchor_num) = args
    
    # Create FCL collision objects for hand
    # IMPORTANT: Keep track of geometries for ID mapping
    hand_boxes = []
    hand_geoms = []
    for i in range(anchor_num):
        box_size = hand_extent_np[i] * 2.0
        geom = fcl.Box(box_size[0], box_size[1], box_size[2])
        R = hand_rotation_np[i]
        T = hand_center_np[i]
        tf = fcl.Transform(R, T)
        obj = fcl.CollisionObject(geom, tf)
        hand_boxes.append(obj)
        hand_geoms.append(geom)
    
    # Create FCL collision objects for object
    object_boxes = []
    object_geoms = []
    for i in range(anchor_num):
        box_size = object_extent_np[i] * 2.0
        geom = fcl.Box(box_size[0], box_size[1], box_size[2])
        R = object_rotation_np[i]
        T = object_center_np[i]
        tf = fcl.Transform(R, T)
        obj = fcl.CollisionObject(geom, tf)
        object_boxes.append(obj)
        object_geoms.append(geom)
    
    # Create managers
    hand_manager = fcl.DynamicAABBTreeCollisionManager()
    object_manager = fcl.DynamicAABBTreeCollisionManager()
    
    hand_manager.registerObjects(hand_boxes)
    object_manager.registerObjects(object_boxes)
    hand_manager.setup()
    object_manager.setup()
    
    # Detect hand-object collisions
    crequest = fcl.CollisionRequest(num_max_contacts=1000, enable_contact=True)
    cdata = fcl.CollisionData(crequest, fcl.CollisionResult())
    hand_manager.collide(object_manager, cdata, fcl.defaultCollisionCallback)
    
    # Create ID to index mappings for hand and object geometries
    # CRITICAL: contact.o1 and contact.o2 are GEOMETRY objects, not CollisionObjects!
    hand_geom_id_to_idx = {id(geom): i for i, geom in enumerate(hand_geoms)}
    object_geom_id_to_idx = {id(geom): i for i, geom in enumerate(object_geoms)}
    
    # Extract hand-object collision pairs
    hand_object_pairs = []
    seen_pairs = set()
    for contact in cdata.result.contacts:
        # Get geometry IDs (contact.o1 and contact.o2 are GEOMETRY objects)
        geom1_id = id(contact.o1)
        geom2_id = id(contact.o2)
        
        # Try to find indices in both mappings
        hand_idx = hand_geom_id_to_idx.get(geom1_id)
        object_idx = object_geom_id_to_idx.get(geom2_id)
        
        # If not found, try reverse
        if hand_idx is None or object_idx is None:
            hand_idx = hand_geom_id_to_idx.get(geom2_id)
            object_idx = object_geom_id_to_idx.get(geom1_id)
        
        if hand_idx is not None and object_idx is not None:
            pair = (hand_idx, object_idx)
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                hand_object_pairs.append(pair)
    
    # Detect hand self-intersections
    hand_self_pairs = []
    if include_self_intersection:
        cdata2 = fcl.CollisionData(crequest, fcl.CollisionResult())
        hand_manager.collide(cdata2, fcl.defaultCollisionCallback)
        
        seen_pairs2 = set()
        for contact in cdata2.result.contacts:
            geom1_id = id(contact.o1)
            geom2_id = id(contact.o2)
            
            # Find indices using the geometry mapping
            idx1 = hand_geom_id_to_idx.get(geom1_id)
            idx2 = hand_geom_id_to_idx.get(geom2_id)
            
            if idx1 is not None and idx2 is not None and idx1 != idx2:
                pair = (min(idx1, idx2), max(idx1, idx2))
                if pair not in seen_pairs2:
                    seen_pairs2.add(pair)
                    hand_self_pairs.append(pair)
    
    return batch_idx, hand_object_pairs, hand_self_pairs


def to_tensor(x, device=None, dtype=None):
    """
    Convert numpy array or tensor to torch tensor with specified device and dtype.
    
    Args:
        x: numpy array or torch tensor
        device: target device (if None, keeps original device)
        dtype: target dtype (if None, keeps original dtype)
        
    Returns:
        torch.Tensor on specified device with specified dtype
    """
    if isinstance(x, torch.Tensor):
        if device is not None or dtype is not None:
            return x.to(device=device, dtype=dtype)
        return x
    else:
        return torch.tensor(x, device=device, dtype=dtype)


class MultiHandCollisionLossModuleOnlyHand(nn.Module):
    """
    Collision loss module for MASH-based hand-object interaction.
    Designed for Hand Contact Map (Hand CMap) scenarios.
    
    Architecture:
    1. Sample points from MASH parameters (pure function, no shared state)
    2. Compute OBBs using PCA (for broadphase collision detection)
    3. Use python-fcl to efficiently detect potentially colliding pairs
    4. Compute penetration loss using signed distance to surface normals
    
    Supports both hand-object collision and hand self-intersection detection.
    """
    
    def __init__(
        self,
        anchor_num: int,
        mask_dim: int,
        sh_dim: int,
        ortho_dim: int,
        mask_degree_max: int,
        sh_degree_max: int,
        sample_phi_num: int = 8,
        sample_theta_num: int = 8,
        device: str = "cuda",
        include_self_intersection: bool = True,
        use_multiprocessing: bool = False,
        num_workers: int = 4,
        loss_type: str = "fcl",  # "fcl" or "dexgrasp"
        erf_weight: float = 1.0,
        cmap_loss_weight: float = 1.0,
        srf_weight: float = 1.0,
        srf_threshold: float = 0.02,
        srf_k_neighbors: int = 10,
    ):
        super().__init__()
        self.anchor_num = anchor_num
        self.mask_dim = mask_dim
        self.sh_dim = sh_dim
        self.ortho_dim = ortho_dim
        self.mask_degree_max = mask_degree_max
        self.sh_degree_max = sh_degree_max
        self.sample_phi_num = sample_phi_num
        self.sample_theta_num = sample_theta_num
        self.device = device
        self.include_self_intersection = include_self_intersection
        self.use_multiprocessing = use_multiprocessing
        self.num_workers = num_workers
        self._statistics = None
        self.statistics_tensors = None
        
        # DexGrasp-style loss configuration
        self.loss_type = loss_type
        self.erf_weight = erf_weight
        self.cmap_loss_weight = cmap_loss_weight
        self.srf_weight = srf_weight
        self.srf_threshold = srf_threshold
        self.srf_k_neighbors = srf_k_neighbors
        
        # Create fcl Collision Objects (will be updated during forward pass)
        self.boxes = {"hand": [], "object": []}
        self.geoms = {"hand": [], "object": []}  # Store geometries for ID mapping
        self._init_fcl_objects()
        
        # Create geometry ID to index mappings (fixed in __init__)
        self.geom_id_to_idx = {
            "hand": {id(geom): i for i, geom in enumerate(self.geoms["hand"])},
            "object": {id(geom): i for i, geom in enumerate(self.geoms["object"])}
        }
        
        # Create FCL collision managers for efficient broadphase collision detection
        self.managers = {
            "hand": fcl.DynamicAABBTreeCollisionManager(),
            "object": fcl.DynamicAABBTreeCollisionManager()
        }
        self.managers["hand"].registerObjects(self.boxes["hand"])
        self.managers["object"].registerObjects(self.boxes["object"])
        self.managers["hand"].setup()
        self.managers["object"].setup()
        
        # Store current OBB parameters for collision loss computation
        self.obb_params = {
            "hand": {"centers": None, "rotations": None, "extents": None, "points": None, "normals": None},
            "object": {"centers": None, "rotations": None, "extents": None, "points": None, "normals": None}
        }
    
    def _init_fcl_objects(self):
        """Initialize fcl Collision Objects for hand and object"""
        # Create placeholder boxes (will be updated with actual sizes during forward)
        box_size = np.array([0.01, 0.01, 0.01])  # Placeholder size
        transform = fcl.Transform()
        
        for i in range(self.anchor_num):
            # Create unique geometries for each object (needed for ID mapping)
            hand_geom = fcl.Box(box_size[0], box_size[1], box_size[2])
            object_geom = fcl.Box(box_size[0], box_size[1], box_size[2])
            
            hand_obj = fcl.CollisionObject(hand_geom, transform)
            object_obj = fcl.CollisionObject(object_geom, transform)
            
            self.boxes["hand"].append(hand_obj)
            self.boxes["object"].append(object_obj)
            self.geoms["hand"].append(hand_geom)
            self.geoms["object"].append(object_geom)
    
    def split_mash_params(self, mash_params: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Split MASH parameters into mask, sh, and ortho components.
        
        Args:
            mash_params: [B, N, 22] or [N, 22] MASH parameters
            
        Returns:
            mask_params: [B, N, mask_dim] or [N, mask_dim]
            sh_params: [B, N, sh_dim] or [N, sh_dim]
            ortho_poses: [B, N, ortho_dim] or [N, ortho_dim]
        """
        if mash_params.dim() == 3:
            # Batch mode
            mask_params = mash_params[:, :, :self.mask_dim]
            sh_params = mash_params[:, :, self.mask_dim:self.mask_dim+self.sh_dim]
            ortho_poses = mash_params[:, :, self.mask_dim+self.sh_dim:]
        else:
            # Single mode
            mask_params = mash_params[:, :self.mask_dim]
            sh_params = mash_params[:, self.mask_dim:self.mask_dim+self.sh_dim]
            ortho_poses = mash_params[:, self.mask_dim+self.sh_dim:]
        
        return mask_params, sh_params, ortho_poses
    
    def compute_sample_points_from_mash_params(
        self,
        mask_params: torch.Tensor,
        sh_params: torch.Tensor,
        ortho_poses: torch.Tensor,
        positions: torch.Tensor,
        target: str = "hand",
    ) -> torch.Tensor:
        """
        Compute sample points from MASH parameters (pure function, no shared state).
        
        Args:
            mask_params: [B, N, mask_dim] MASH mask parameters
            sh_params: [B, N, sh_dim] MASH sh parameters
            ortho_poses: [B, N, 6] MASH ortho poses
            positions: [B, N, 3] anchor positions
            target: "hand" or "object"
            
        Returns:
            sample_points: [B, N, S, S, 3] sample points
        """
        B, N = positions.shape[:2]
        S = self.sample_phi_num
        device = positions.device
        dtype = positions.dtype
        
        # Step 1: Generate sampling grid (phi, theta)
        sample_phis = torch.linspace(
            2 * math.pi / S, 2 * math.pi, S, dtype=dtype, device=device
        )
        sample_thetas = torch.linspace(
            math.pi / S, math.pi, S, dtype=dtype, device=device
        )
        phi_grid, theta_grid = torch.meshgrid(sample_phis, sample_thetas, indexing="ij")
        sample_phi_theta = torch.stack([phi_grid, theta_grid], dim=-1)  # [S, S, 2]
        
        # Step 2: Compute mask (theta weights) using Fourier series
        theta_weights = mask_params[:, :, 0:1].expand(-1, -1, S)  # [B, N, S]
        if self.mask_degree_max > 0:
            degrees = torch.arange(1, self.mask_degree_max + 1, dtype=dtype, device=device)
            angles = degrees.unsqueeze(1) * sample_phis.unsqueeze(0)  # [mask_degree_max, S]
            cos_terms = torch.cos(angles)  # [mask_degree_max, S]
            sin_terms = torch.sin(angles)  # [mask_degree_max, S]
            
            cos_params = mask_params[:, :, 1::2]  # [B, N, mask_degree_max]
            sin_params = mask_params[:, :, 2::2]  # [B, N, mask_degree_max]
            theta_weights = theta_weights + torch.matmul(cos_params, cos_terms)  # [B, N, S]
            theta_weights = theta_weights + torch.matmul(sin_params, sin_terms)  # [B, N, S]
        
        theta_weights = torch.sigmoid(theta_weights)  # [B, N, S]
        
        # Step 3: Apply mask to thetas
        phis = sample_phi_theta[:, :, 0].unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1)  # [B, N, S, S]
        thetas = sample_phi_theta[:, :, 1].unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1)  # [B, N, S, S]
        weighted_thetas = thetas * theta_weights.unsqueeze(-1)  # [B, N, S, S]
        
        # Step 4: Compute SH distances
        BP = B * N
        phis_flat = phis.reshape(BP, S, S)
        thetas_flat = weighted_thetas.reshape(BP, S, S)
        base_values = mash_cpp.toSHBaseValues(phis_flat, thetas_flat, self.sh_degree_max)  # [(D+1)^2, BP, S, S]
        base_values = base_values.transpose(1, 0)  # [BP, (D+1)^2, S, S]
        
        sh_params_flat = sh_params.reshape(BP, -1)  # [BP, (D+1)^2]
        distances_flat = torch.einsum("bn,bn...->b...", sh_params_flat, base_values)  # [BP, S, S]
        distances = distances_flat.reshape(B, N, S, S)  # [B, N, S, S]
        
        # Step 5: Compute directions from (phi, theta)
        sin_theta, cos_theta = torch.sin(weighted_thetas), torch.cos(weighted_thetas)
        sin_phi, cos_phi = torch.sin(phis), torch.cos(phis)
        directions = torch.stack([
            sin_theta * cos_phi,
            sin_theta * sin_phi,
            cos_theta
        ], dim=-1)  # [B, N, S, S, 3]
        
        # Step 6: Compute move vectors
        move_vectors = directions * distances.unsqueeze(-1)  # [B, N, S, S, 3]
        
        # Step 7: Apply inversion transformation
        inv_radius = W0 * sh_params[:, :, 0]  # [B, N]
        inv_centers = torch.zeros(B, N, 1, 3, dtype=dtype, device=device)
        inv_centers[:, :, 0, 2] = -inv_radius
        
        move_vectors_flat = move_vectors.reshape(B, N, -1, 3)  # [B, N, S*S, 3]
        in_inv_points = move_vectors_flat - inv_centers  # [B, N, S*S, 3]
        in_inv_norms_sq = torch.sum(in_inv_points.square(), dim=3, keepdim=True)  # [B, N, S*S, 1]
        in_inv_weights = torch.rsqrt(in_inv_norms_sq)  # [B, N, S*S, 1]
        in_inv_directions = in_inv_points * in_inv_weights  # [B, N, S*S, 3]
        
        inv_radius_sq_4 = (4.0 * inv_radius.square()).view(B, N, 1, 1)  # [B, N, 1, 1]
        in_inv_lengths = inv_radius_sq_4 * in_inv_weights  # [B, N, S*S, 1]
        inv_points = inv_centers + in_inv_lengths * in_inv_directions  # [B, N, S*S, 3]
        inv_points = inv_points.view(B, N, S, S, 3)  # [B, N, S, S, 3]
        
        # Step 8: Apply rotation and translation
        ortho_flat = ortho_poses.reshape(-1, 6)  # [B*N, 6]
        rotate_mats = compute_rotation_matrix_from_ortho6d(ortho_flat).reshape(B, N, 3, 3)  # [B, N, 3, 3]
        rotated_vectors = torch.einsum("bn...i,bnij->bn...j", inv_points, rotate_mats)  # [B, N, S, S, 3]
        
        positions_expanded = positions.view(B, N, 1, 1, 3)  # [B, N, 1, 1, 3]
        sample_points = positions_expanded + rotated_vectors  # [B, N, S, S, 3]
        
        return sample_points
    
    def compute_obb_from_points(self, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute OBB from sample points using PCA (differentiable).
        Only supports batched input.
        
        Args:
            points: [B, N, M, 3] sample points for each anchor (batch mode)
            
        Returns:
            center: [B, N, 3] OBB center
            rotation: [B, N, 3, 3] rotation matrix (principal axes)
            extent: [B, N, 3] OBB extent (half-lengths)
        """
        B, N, M, _ = points.shape
        
        # Compute centroid
        center = points.mean(dim=-2)  # [B, N, 3]
        
        # Center points
        centered = points - center.unsqueeze(2)  # [B, N, M, 3]
        
        # Compute covariance matrix
        # [B, N, 3, M] @ [B, N, M, 3] -> [B, N, 3, 3]
        # Use torch.matmul for 4D batch matrix multiplication
        cov = torch.matmul(centered.transpose(-1, -2), centered) / (M - 1)
        
        # Eigenvalue decomposition
        try:
            eigenvals, eigenvecs = torch.linalg.eigh(cov)  # [B, N, 3], [B, N, 3, 3]
        except torch.linalg.LinAlgError:
            # Fallback to identity if decomposition fails
            eigenvals = torch.ones(B, N, 3, device=points.device, dtype=points.dtype)
            eigenvecs = torch.eye(3, device=points.device, dtype=points.dtype).unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1)
        
        # Sort by eigenvalues (descending)
        indices = torch.argsort(eigenvals, dim=-1, descending=True)  # [B, N, 3]
        eigenvals_sorted = torch.gather(eigenvals, -1, indices)
        eigenvecs_sorted = torch.gather(eigenvecs, -2, indices.unsqueeze(-1).expand(-1, -1, -1, 3))
        
        # Handle degenerate cases (near-zero eigenvalues)
        eps = 1e-6
        valid_mask = eigenvals_sorted > eps
        eigenvecs_sorted = torch.where(
            valid_mask.unsqueeze(-1).expand(-1, -1, -1, 3),
            eigenvecs_sorted,
            torch.eye(3, device=points.device, dtype=points.dtype).unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1)
        )
        
        rotation = eigenvecs_sorted  # Principal axes as rotation matrix
        
        # Project points onto principal axes and compute extent
        proj = torch.einsum('bnmi,bnij->bnmj', centered, rotation)  # [B, N, M, 3]
        proj_min = proj.min(dim=2)[0]  # [B, N, 3]
        proj_max = proj.max(dim=2)[0]  # [B, N, 3]
        extent = (proj_max - proj_min) / 2.0  # [B, N, 3]
        # Add small epsilon to avoid zero extent
        extent = extent + 1e-6
        
        return center, rotation, extent
    
    def compute_signed_distance_to_object_surface(
        self,
        hand_points: torch.Tensor,
        object_points: torch.Tensor,
        object_normals: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute signed distance from hand points to object surface using nearest neighbor and normals (batch version).
        For each pair, hand points are matched to nearest object points in the same pair's object surfel.
        Positive = outside, negative = inside (penetration).
        
        Args:
            hand_points: [B, M, 3] hand surfel points (batch of pairs)
            object_points: [B, M, 3] object surfel points (batch of pairs, corresponding to hand_points)
            object_normals: [B, M, 3] object point normals (batch of pairs, corresponding to object_points)
            
        Returns:
            signed_distances: [B, M] signed distances
        """
        B, M, _ = hand_points.shape
        
        # IMPORTANT: Use detached versions for KNN search to avoid complex computation graph
        # KNN is only used to find correspondences, not for gradient computation
        # The actual distance calculation will use the original (gradient-enabled) points
        hand_points_detached = hand_points.detach()
        object_points_detached = object_points.detach()
        
        # For each pair, find nearest neighbor in corresponding object anchor's points
        # Use detached versions for KNN search
        knn_result = knn_points(hand_points_detached, object_points_detached, K=1)  # Per-pair KNN
        nearest_indices = knn_result.idx[:, :, 0]  # [B, M] - indices of nearest object points within each pair
        
        # Get nearest points and normals using batch indexing
        # Use .clone() to avoid "backward through graph a second time" error when
        # multiple hand points map to the same object point
        batch_indices = torch.arange(B, device=nearest_indices.device).unsqueeze(1).expand(-1, M)
        nearest_points = object_points[batch_indices, nearest_indices].clone()
        nearest_normals = object_normals[batch_indices, nearest_indices].clone()
        
        # Compute vector from nearest object point to hand point (using original points)
        vec_to_hand = hand_points - nearest_points  # [B, M, 3] - gradient-enabled
        
        # Signed distance: dot product with normal
        # Positive = hand point is outside (in direction of normal)
        # Negative = hand point is inside (opposite to normal)
        signed_dist = torch.sum(vec_to_hand * nearest_normals, dim=-1)  # [B, M]
        
        return signed_dist
    
    def compute_batched_penetration_loss(
        self,
        collision_pairs: List[Tuple[int, int, int]],
        hand_points: torch.Tensor,
        object_points: torch.Tensor,
        object_normals: torch.Tensor,
        is_self_intersection: bool = False,
        penetration_threshold: float = 0.0,
    ) -> torch.Tensor:
        """
        Compute penetration loss for collision pairs in a batched manner (NO FOR LOOP).
        
        Args:
            collision_pairs: List of (batch_idx, hand_anchor_idx, object_anchor_idx)
            hand_points: [B, N, M, 3] hand surfel points (with gradient)
            object_points: [B, N, M, 3] object surfel points (with gradient)
            object_normals: [B, N, M, 3] object normals (detached)
            is_self_intersection: Whether this is hand self-intersection
            penetration_threshold: Distance threshold for penetration
            
        Returns:
            loss: scalar tensor
        """
        if len(collision_pairs) == 0:
            return torch.tensor(0.0, device=hand_points.device, dtype=hand_points.dtype)
        
        # Extract indices for collision pairs
        batch_indices = torch.tensor([p[0] for p in collision_pairs], device=hand_points.device, dtype=torch.long)
        hand_indices = torch.tensor([p[1] for p in collision_pairs], device=hand_points.device, dtype=torch.long)
        object_indices = torch.tensor([p[2] for p in collision_pairs], device=hand_points.device, dtype=torch.long)
        
        # Gather points for collision pairs (use .clone() to avoid double backward)
        hand_points_pairs = hand_points[batch_indices, hand_indices, :, :].clone()
        object_points_pairs = object_points[batch_indices, object_indices, :, :].clone()
        object_normals_pairs = object_normals[batch_indices, object_indices, :, :]
        
        # Compute signed distance in batch (calls existing batched function)
        signed_distances = self.compute_signed_distance_to_object_surface(
            hand_points=hand_points_pairs,  # [num_pairs, M, 3]
            object_points=object_points_pairs,  # [num_pairs, M, 3]
            object_normals=object_normals_pairs,  # [num_pairs, M, 3]
        )  # [num_pairs, M]
        
        # Penetration loss: penalize negative distances (inside object)
        penetrations = torch.clamp(-signed_distances - penetration_threshold, min=0.0)  # [num_pairs, M]
        loss = penetrations.mean()  # Average over all pairs and points
        
        return loss
    
    def _update_fcl_objects_from_obb(
        self,
        centers: torch.Tensor,
        rotations: torch.Tensor,
        extents: torch.Tensor,
        fcl_objects: List,
        fcl_geoms: List = None,
        batch_idx: int = 0,
    ):
        """
        Update fcl Collision Objects from OBB parameters.
        
        Args:
            centers: [N, 3] OBB centers
            rotations: [N, 3, 3] rotation matrices
            extents: [N, 3] OBB extents (half-lengths)
            fcl_objects: List of fcl.CollisionObject to update
            fcl_geoms: Optional list of fcl.CollisionGeometry to update (preserves ID)
            batch_idx: batch index (for batch mode)
        """
        N = centers.shape[0]
        centers_np = centers.detach().cpu().numpy()
        rotations_np = rotations.detach().cpu().numpy()
        extents_np = extents.detach().cpu().numpy()
        
        for i in range(N):
            # Create box geometry (fcl.Box uses full size, not half-size)
            box_size = extents_np[i] * 2.0
            
            # Update existing geometry if it exists (preserves ID)
            if fcl_geoms is not None and i < len(fcl_geoms):
                geom = fcl_geoms[i]
                # Try to update Box size if it's a Box geometry
                if isinstance(geom, fcl.Box):
                    # FCL Box has a 'side' attribute that can be updated
                    geom.side = box_size
                else:
                    # If not a Box, create new Box and update the list
                    box_geom = fcl.Box(box_size[0], box_size[1], box_size[2])
                    fcl_geoms[i] = box_geom
                    fcl_objects[i].setGeometry(box_geom)
            else:
                # Create new geometry if not in list
                box_geom = fcl.Box(box_size[0], box_size[1], box_size[2])
                fcl_objects[i].setGeometry(box_geom)
            
            # Create transform
            R = rotations_np[i]
            T = centers_np[i]
            transform = fcl.Transform(R, T)
            
            # Update collision object transform
            fcl_objects[i].setTransform(transform)
    
    def _detect_collisions_fcl(
        self,
        key1: str,
        key2: str,
        exclude_same_anchor: bool = False,
    ) -> List[Tuple[int, int]]:
        """
        Detect collisions between two sets of fcl objects using FCL's broadphase collision detection.
        
        Args:
            key1: Key for first group ("hand" or "object")
            key2: Key for second group ("hand" or "object")
            exclude_same_anchor: If True, exclude (i, i) pairs
            
        Returns:
            List of (idx1, idx2) collision pairs
        """
        manager1 = self.managers[key1]
        manager2 = self.managers[key2]
        geom_id_to_idx1 = self.geom_id_to_idx[key1]
        geom_id_to_idx2 = self.geom_id_to_idx[key2]
        
        # Create collision request
        crequest = fcl.CollisionRequest(num_max_contacts=1000, enable_contact=True)
        cdata = fcl.CollisionData(crequest, fcl.CollisionResult())
        
        # Managers should already be updated by update_obbs, but update again to be safe
        manager1.update()
        manager2.update()
        
        # Check if same manager (for self-intersection)
        if manager1 is manager2:
            # Internal collision detection
            manager1.collide(cdata, fcl.defaultCollisionCallback)
        else:
            # Group collision detection
            manager1.collide(manager2, cdata, fcl.defaultCollisionCallback)
        
        # Extract collision pairs from contacts
        collision_pairs = []
        seen_pairs = set()  # Avoid duplicates
        
        for contact in cdata.result.contacts:
            # Get geometry IDs
            geom1_id = id(contact.o1)
            geom2_id = id(contact.o2)
            
            # Map to indices
            idx1 = geom_id_to_idx1.get(geom1_id)
            idx2 = geom_id_to_idx2.get(geom2_id)
            
            # Handle reverse mapping (contact might be in reverse order)
            if idx1 is None:
                idx1 = geom_id_to_idx1.get(geom2_id)
                idx2 = geom_id_to_idx2.get(geom1_id)
            
            if idx1 is not None and idx2 is not None:
                # Check exclude_same_anchor
                if exclude_same_anchor and idx1 == idx2:
                    continue
                
                # Create sorted pair to avoid duplicates
                pair = (min(idx1, idx2), max(idx1, idx2)) if key1 == key2 else (idx1, idx2)
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    collision_pairs.append(pair)
        
        return collision_pairs
    
    def _denormalize_params(
        self,
        hand_params: torch.Tensor,
        object_params: torch.Tensor,
        positions: torch.Tensor,
        cmap_params: Optional[torch.Tensor] = None,
        stats: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Denormalize all parameters using statistics (all at once).
        
        Args:
            hand_params: [B, N, 22] normalized hand MASH parameters
            object_params: [B, N, 22] normalized object MASH parameters
            positions: [B, N, 3] normalized anchor positions
            cmap_params: [B, N, sh_dim] normalized contact map parameters (optional) (hand_cmap)
            stats: Dict[str, torch.Tensor]
            
        Returns:
            hand_params_denorm: [B, N, 22]
            object_params_denorm: [B, N, 22]
            positions_denorm: [B, N, 3]
            cmap_params_denorm: [B, N, sh_dim] or None
        """
        if stats is None:
            return hand_params, object_params, positions, cmap_params
        
        device = hand_params.device
        dtype = hand_params.dtype
        
        # ============================================================================
        # 1. Denormalize hand_params
        # ============================================================================
        hand_mask, hand_sh, hand_ortho = self.split_mash_params(hand_params)

        # print("=================================================")
        # print(hand_mask.shape)
        # print(stats["std_hand_mask"].shape)
        # print(stats["mean_hand_mask"].shape)
        # print("=================================================")
        
        hand_mask_denorm = hand_mask * stats["std_hand_mask"].unsqueeze(1).to(device=device, dtype=dtype) + stats["mean_hand_mask"].unsqueeze(1).to(device=device, dtype=dtype)
        hand_sh_denorm = hand_sh * stats["std_hand_sh"].unsqueeze(1).to(device=device, dtype=dtype) + stats["mean_hand_sh"].unsqueeze(1).to(device=device, dtype=dtype)
        hand_ortho_denorm = hand_ortho * stats["std_hand_ortho"].unsqueeze(1).to(device=device, dtype=dtype) + stats["mean_hand_ortho"].unsqueeze(1).to(device=device, dtype=dtype)
        
        hand_params_denorm = torch.cat([hand_mask_denorm, hand_sh_denorm, hand_ortho_denorm], dim=-1)
        
        # ============================================================================
        # 2. Denormalize object_params
        # ============================================================================
        object_mask, object_sh, object_ortho = self.split_mash_params(object_params)
        
        object_mask_denorm = object_mask * stats["std_obj_mask"].unsqueeze(1).to(device=device, dtype=dtype) + stats["mean_obj_mask"].unsqueeze(1).to(device=device, dtype=dtype)
        object_sh_denorm = object_sh * stats["std_obj_sh"].unsqueeze(1).to(device=device, dtype=dtype) + stats["mean_obj_sh"].unsqueeze(1).to(device=device, dtype=dtype)
        object_ortho_denorm = object_ortho * stats["std_obj_ortho"].unsqueeze(1).to(device=device, dtype=dtype) + stats["mean_obj_ortho"].unsqueeze(1).to(device=device, dtype=dtype)
        
        object_params_denorm = torch.cat([object_mask_denorm, object_sh_denorm, object_ortho_denorm], dim=-1)
        
        # ============================================================================
        # 3. Denormalize positions
        # ============================================================================
        positions_denorm = positions * stats["std_positions"].unsqueeze(1).to(device=device, dtype=dtype) + stats["mean_positions"].unsqueeze(1).to(device=device, dtype=dtype)
        
        # ============================================================================
        # 4. Denormalize cmap_params (if provided)
        # ============================================================================
        cmap_params_denorm = None
        if cmap_params is not None:
            cmap_std = stats["std_hand_cmap"].to(device=device, dtype=dtype)
            cmap_mean = stats["mean_hand_cmap"].to(device=device, dtype=dtype)
            
            # Check statistics for issues
            if torch.isnan(cmap_std).any() or torch.isinf(cmap_std).any():
                print(f"[ERROR] cmap_params std has NaN/Inf! std range: [{cmap_std.min()}, {cmap_std.max()}]")
                cmap_std = torch.nan_to_num(cmap_std, nan=1.0, posinf=1.0, neginf=1.0)
            
            if torch.isnan(cmap_mean).any() or torch.isinf(cmap_mean).any():
                print(f"[ERROR] cmap_params mean has NaN/Inf! mean range: [{cmap_mean.min()}, {cmap_mean.max()}]")
                cmap_mean = torch.nan_to_num(cmap_mean, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Check if std has zeros (would cause issues)
            if (cmap_std == 0).any():
                print("[WARNING] cmap_params std has zeros! Replacing with 1.0")
                cmap_std = torch.where(cmap_std == 0, torch.ones_like(cmap_std), cmap_std)
            
            # Check input cmap_params
            if torch.isnan(cmap_params).any() or torch.isinf(cmap_params).any():
                print("[WARNING] Input cmap_params has NaN/Inf before denormalization!")
                cmap_params = torch.nan_to_num(cmap_params, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Denormalize
            cmap_params_denorm = cmap_params * cmap_std.unsqueeze(1) + cmap_mean.unsqueeze(1)
            
            # Check for NaN/Inf after denormalization
            if torch.isnan(cmap_params_denorm).any() or torch.isinf(cmap_params_denorm).any():
                print("[WARNING] NaN/Inf in cmap_params_denorm after denormalization")
                print(f"  cmap_params range: [{cmap_params.min()}, {cmap_params.max()}]")
                print(f"  cmap_std range: [{cmap_std.min()}, {cmap_std.max()}]")
                print(f"  cmap_mean range: [{cmap_mean.min()}, {cmap_mean.max()}]")
                cmap_params_denorm = torch.nan_to_num(cmap_params_denorm, nan=0.0, posinf=10.0, neginf=-10.0)
        elif cmap_params is not None:
            # No statistics available, keep as is
            cmap_params_denorm = cmap_params
        
        return hand_params_denorm, object_params_denorm, positions_denorm, cmap_params_denorm
    
    def update_obbs(
        self,
        key: str,
        centers: torch.Tensor,
        rotations: torch.Tensor,
        extents: torch.Tensor,
    ):
        """
        Update OBB parameters and FCL objects for a given key.
        
        This method updates:
        - self.boxes[key]: FCL CollisionObjects (geometry and transform)
        - self.geoms[key]: FCL CollisionGeometries (if supported)
        - self.managers[key]: Updates AABB tree
        
        For "object" key, also computes point cloud normals.
        
        Args:
            key: Key for the group ("hand" or "object")
            centers: [N, 3] OBB centers
            rotations: [N, 3, 3] OBB rotations
            extents: [N, 3] OBB extents
        """
        # Update FCL objects (boxes and geoms)
        self._update_fcl_objects_from_obb(centers, rotations, extents, self.boxes[key], self.geoms[key])
        
        # Update manager's AABB tree
        self.managers[key].update()
        
        # Store OBB parameters for later use
        self.obb_params[key]["centers"] = centers
        self.obb_params[key]["rotations"] = rotations
        self.obb_params[key]["extents"] = extents
    
    # ============================================================================
    # DexGrasp-style Loss Functions
    # ============================================================================
    
    def compute_ERF_loss(
        self,
        hand_pcd: torch.Tensor,
        obj_pcd_nor: torch.Tensor,
        return_per_sample: bool = False,
    ) -> torch.Tensor:
        """
        ERF Loss (Exterior Region Force Loss) - Penalize hand penetration into object.
        
        Args:
            hand_pcd: [B, N_hand, 3] hand point cloud
            obj_pcd_nor: [B, N_obj, 6] object point cloud with normals (xyz + normal_xyz)
            return_per_sample: If True, return [B] per-sample losses instead of scalar
            
        Returns:
            ERF_loss: scalar tensor or [B] per-sample losses
        """
        B = hand_pcd.shape[0]
        
        # Separate object point cloud and normals
        obj_pcd = obj_pcd_nor[:, :, :3]  # [B, N_obj, 3]
        obj_nor = obj_pcd_nor[:, :, 3:6]  # [B, N_obj, 3]
        
        # Compute K-nearest neighbors (find nearest object point for each hand point)
        knn_result = knn_points(hand_pcd, obj_pcd, K=1, return_nn=True)
        distances = knn_result.dists.sqrt()  # [B, N_hand, 1]
        indices = knn_result.idx  # [B, N_hand, 1]
        
        # Extract the closest object points and normals
        batch_idx = torch.arange(B, device=indices.device).unsqueeze(1).unsqueeze(2)  # [B, 1, 1]
        indices_expanded = indices.expand(-1, -1, 3)  # [B, N_hand, 3]
        batch_idx_expanded = batch_idx.expand(-1, hand_pcd.shape[1], 3)  # [B, N_hand, 3]
        
        hand_obj_points = torch.gather(obj_pcd, 1, indices_expanded)  # [B, N_hand, 3]
        hand_obj_normals = torch.gather(obj_nor, 1, indices_expanded)  # [B, N_hand, 3]
        
        # Compute the signs (determine if hand point is inside object)
        # Vector from hand point to object point
        hand_obj_signs = ((hand_obj_points - hand_pcd) * hand_obj_normals).sum(dim=2)  # [B, N_hand]
        hand_obj_signs = (hand_obj_signs > 0.).float()  # 1 = penetration, 0 = outside
        
        # Compute collision value (only penalize penetration)
        penetration_distances = hand_obj_signs * distances.squeeze(2)  # [B, N_hand]
        
        # Per-sample loss: mean of penetrating points only
        # For samples with no penetration, loss = 0
        collision_value = torch.zeros(B, device=hand_pcd.device, dtype=hand_pcd.dtype)
        for i in range(B):
            penetrating_mask = penetration_distances[i] > 0  # [N_hand]
            if penetrating_mask.any():
                # Only compute loss for penetrating points
                collision_value[i] = penetration_distances[i][penetrating_mask].mean()
            # else: collision_value[i] remains 0 (no penetration)
        
        # Debug: Print penetration statistics (only occasionally to avoid spam)
        if torch.rand(1).item() < 0.01:  # 1% chance to print
            num_penetrating = (hand_obj_signs > 0).sum().item()
            total_points = hand_obj_signs.numel()
            if num_penetrating > 0:
                penetrating_dists = penetration_distances[penetration_distances > 0]
                print(f"[DEBUG ERF] Penetrating points: {num_penetrating}/{total_points} "
                      f"({100*num_penetrating/total_points:.1f}%), "
                      f"Distances: min={penetrating_dists.min().item():.4f}, "
                      f"max={penetrating_dists.max().item():.4f}, "
                      f"mean={penetrating_dists.mean().item():.4f}, "
                      f"ERF_loss={collision_value.mean().item():.4f}")
        
        if return_per_sample:
            return collision_value  # [B]
        else:
            return collision_value.mean()  # scalar
    
    def compute_contact_map_alignment_loss(
        self,
        contact_points: torch.Tensor,
        obj_pcd: torch.Tensor,
        cmap_scalar: torch.Tensor,
        return_per_sample: bool = False,
    ) -> torch.Tensor:
        """
        Contact Map Alignment Loss - Align predicted contact with target contact map.
        Based on shadow_fitter.py implementation.
        
        Args:
            contact_points: [B, N_contact, 3] predicted contact points from cmap
            obj_pcd: [B, N_obj, 3] object point cloud
            cmap_scalar: [B, N_contact] target contact map values (higher = stronger contact)
            return_per_sample: If True, return [B] per-sample losses instead of scalar
            
        Returns:
            contact_map_loss: scalar tensor or [B] per-sample losses
        """
        B = contact_points.shape[0]
        
        # Compute distance from contact points to object surface
        # d2_oh: [B, N_contact, 1] - squared distance to nearest object point
        d2_oh, _, _ = knn_points(obj_pcd, contact_points, K=1, return_nn=False)
        d2 = d2_oh.squeeze(-1)  # [B, N_contact]
        d = torch.sqrt(torch.clamp(d2, min=1e-12))  # [B, N_contact]
        
        # Convert distance to contact probability using sigmoid
        # contact_curr = 1 means "on surface", 0 means "far from surface"
        # Same formula as shadow_fitter.py
        contact_curr = 1.0 - 2.0 * (torch.sigmoid(1 * d) - 0.5)  # [B, N_contact]
        
        # Normalize cmap_scalar to [0, 1] range for comparison
        # Use min-max normalization per batch
        cmap_flat = cmap_scalar.reshape(B, -1)  # [B, N_contact]
        cmap_min = cmap_flat.min(dim=-1, keepdim=True)[0]  # [B, 1]
        cmap_max = cmap_flat.max(dim=-1, keepdim=True)[0]  # [B, 1]
        
        # Normalize to [0, 1]
        cmap_normalized = torch.where(
            (cmap_max - cmap_min) > 1e-6,
            (cmap_flat - cmap_min) / (cmap_max - cmap_min + 1e-6),
            torch.zeros_like(cmap_flat)
        )  # [B, N_contact]
        
        # MSE loss per sample
        per_sample_loss = torch.mean((contact_curr - cmap_normalized) ** 2, dim=-1)  # [B]
        
        if return_per_sample:
            return per_sample_loss  # [B]
        else:
            return per_sample_loss.mean()  # scalar
    
    def compute_SRF_loss(
        self,
        hand_points: torch.Tensor,
        thres_dis: float = 0.02,
        k_neighbors: int = 10,
        return_per_sample: bool = False,
    ) -> torch.Tensor:
        """
        SRF Loss (Self-penetration Repulsion Force Loss) - Penalize hand self-collision.
        Uses K-nearest neighbors for memory efficiency instead of all pairwise distances.
        
        Args:
            hand_points: [B, N_hand, 3] hand point cloud
            thres_dis: distance threshold for self-collision
            k_neighbors: number of nearest neighbors to consider (much more efficient than O(N^2))
            return_per_sample: If True, return [B] per-sample losses instead of scalar
            
        Returns:
            SRF_loss: scalar tensor or [B] per-sample losses
        """
        B, N, _ = hand_points.shape
        
        # Use KNN to find K nearest neighbors for each point (O(N*K) instead of O(N^2))
        # This avoids OOM by only computing distances to K nearest neighbors
        knn_result = knn_points(hand_points, hand_points, K=k_neighbors + 1, return_nn=False)
        # knn_result.dists: [B, N, K+1] - squared distances to K+1 nearest neighbors (includes self)
        
        # Remove the first neighbor (self) and take sqrt to get actual distances
        distances = knn_result.dists[:, :, 1:].sqrt()  # [B, N, K]
        
        # Compute penetration depth (only for points closer than threshold)
        penetrations = thres_dis - distances  # [B, N, K]
        penetrations = torch.clamp(penetrations, min=0.0)  # Only keep positive values (penetrations)
        
        # Debug: Print self-penetration statistics (only occasionally)
        if torch.rand(1).item() < 0.01:  # 1% chance to print
            num_penetrating = (penetrations > 0).sum().item()
            total_pairs = penetrations.numel()
            if num_penetrating > 0:
                penetrating_vals = penetrations[penetrations > 0]
                srf_val = penetrations.sum() / (B * N * k_neighbors)
                print(f"[DEBUG SRF] Self-penetrating pairs: {num_penetrating}/{total_pairs} "
                      f"({100*num_penetrating/total_pairs:.1f}%), "
                      f"Penetrations: min={penetrating_vals.min().item():.4f}, "
                      f"max={penetrating_vals.max().item():.4f}, "
                      f"mean={penetrating_vals.mean().item():.4f}, "
                      f"SRF_loss={srf_val.item():.4f}")
        
        if return_per_sample:
            # Average penetration per sample
            per_sample_loss = penetrations.sum(dim=[1, 2]) / (N * k_neighbors)  # [B]
            return per_sample_loss
        else:
            # Average penetration depth across all points and neighbors
            num_pairs = B * N * k_neighbors
            if num_pairs > 0:
                SRF_loss = penetrations.sum() / num_pairs
            else:
                SRF_loss = torch.tensor(0.0, device=hand_points.device, dtype=hand_points.dtype)
            return SRF_loss
    
    def compute_cmap_scalar_field(
        self,
        cmap_params: torch.Tensor,
        hand_params: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute contact map scalar field (similar to toCMap() in hoi_mash.py).
        
        Returns contact strength/probability at each sampled direction (phi, theta).
        
        Args:
            cmap_params: [B, N, sh_dim] contact map SH parameters
            hand_params: [B, N, 22] hand MASH parameters (for mask)
            
        Returns:
            cmap_scalar: [B, N, S, S] contact strength at each direction
        """
        B, N, _ = cmap_params.shape
        S = self.sample_phi_num
        device = cmap_params.device
        dtype = cmap_params.dtype
        
        # Extract hand mask parameters
        hand_mask, _, _ = self.split_mash_params(hand_params)
        
        # Step 1: Compute weighted (phi, theta) using hand's mask
        sample_phis = torch.linspace(
            2 * math.pi / S, 2 * math.pi, S, dtype=dtype, device=device
        )
        sample_thetas = torch.linspace(
            math.pi / S, math.pi, S, dtype=dtype, device=device
        )
        phi_grid, theta_grid = torch.meshgrid(sample_phis, sample_thetas, indexing="ij")
        
        # Compute theta weights using hand's mask (same as in compute_sample_points_from_mash_params)
        theta_weights = hand_mask[:, :, 0:1].expand(-1, -1, S)  # [B, N, S]
        if self.mask_degree_max > 0:
            degrees = torch.arange(1, self.mask_degree_max + 1, dtype=dtype, device=device)
            angles = degrees.unsqueeze(1) * sample_phis.unsqueeze(0)
            cos_terms = torch.cos(angles)
            sin_terms = torch.sin(angles)
            cos_params = hand_mask[:, :, 1::2]
            sin_params = hand_mask[:, :, 2::2]
            theta_weights = theta_weights + torch.matmul(cos_params, cos_terms)
            theta_weights = theta_weights + torch.matmul(sin_params, sin_terms)
        theta_weights = torch.sigmoid(theta_weights)  # [B, N, S]
        
        # Apply mask to thetas
        phis = phi_grid.unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1)  # [B, N, S, S]
        thetas = theta_grid.unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1)  # [B, N, S, S]
        weighted_thetas = thetas * theta_weights.unsqueeze(-1)  # [B, N, S, S]
        
        # Step 2: Compute cmap scalar field using SH basis
        BP = B * N
        phis_flat = phis.reshape(BP, S, S)
        weighted_thetas_flat = weighted_thetas.reshape(BP, S, S)
        
        # Compute SH base values
        # Infer cmap SH degree from cmap_params dimensions
        cmap_sh_dim = cmap_params.shape[-1]
        cmap_sh_degree = int(math.sqrt(cmap_sh_dim)) - 1  # (degree+1)^2 = sh_dim
        
        base_values = mash_cpp.toSHBaseValues(phis_flat, weighted_thetas_flat, cmap_sh_degree)
        base_values = base_values.transpose(1, 0)  # [BP, cmap_sh_dim, S, S]
        base_values = base_values.detach()
        
        # Compute cmap scalar field: cmap = sum(cmap_params * base_values)
        cmap_params_flat = cmap_params.reshape(BP, -1)  # [BP, cmap_sh_dim]
        
        # Check for NaN/Inf in inputs
        if torch.isnan(cmap_params_flat).any() or torch.isinf(cmap_params_flat).any():
            print("[WARNING] NaN/Inf detected in cmap_params_flat")
            cmap_params_flat = torch.nan_to_num(cmap_params_flat, nan=0.0, posinf=1e6, neginf=-1e6)
        
        if torch.isnan(base_values).any() or torch.isinf(base_values).any():
            print("[WARNING] NaN/Inf detected in base_values")
            base_values = torch.nan_to_num(base_values, nan=0.0, posinf=1e6, neginf=-1e6)
        
        cmap_scalar = torch.einsum("bn,bn...->b...", cmap_params_flat, base_values)  # [BP, S, S]
        cmap_scalar = cmap_scalar.reshape(B, N, S, S)  # [B, N, S, S]
        
        # Keep original cmap values (no clamping or sigmoid)
        # Will normalize by sum when computing weights
        return cmap_scalar
    
    def compute_contact_points_from_cmap(
        self,
        cmap_params: torch.Tensor,
        hand_params: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Sample contact points from cmap parameters using the original cmap scalar field method.
        
        Process:
        1. Compute cmap scalar field (contact strength at each direction)
        2. Sample hand surface points
        3. Weight surface points by cmap values to get contact candidates
        
        Args:
            cmap_params: [B, N, sh_dim] contact map SH parameters
            hand_params: [B, N, 22] hand MASH parameters
            positions: [B, N, 3] anchor positions
            
        Returns:
            contact_points: [B, N, S, S, 3] weighted contact points
        """
        B, N, _ = cmap_params.shape
        device = cmap_params.device
        dtype = cmap_params.dtype
        
        # Step 1: Compute cmap scalar field (contact strength)
        cmap_scalar = self.compute_cmap_scalar_field(cmap_params, hand_params)  # [B, N, S, S]
        
        # Step 2: Sample hand surface points
        hand_mask, hand_sh, hand_ortho = self.split_mash_params(hand_params)
        hand_surface_points = self.compute_sample_points_from_mash_params(
            hand_mask, hand_sh, hand_ortho, positions, target="hand"
        )  # [B, N, S, S, 3]
        
        
        return hand_surface_points

    def forward(
        self,
        valid_timesteps_mask: torch.Tensor,
        hand_params: torch.Tensor,
        object_params: torch.Tensor,
        positions: torch.Tensor,
        denormalize: bool = True,
        hand_cmap_params: Optional[torch.Tensor] = None,
        sample_weights: Optional[torch.Tensor] = None,
        stats: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Compute collision loss between hand and object, and optionally hand self-intersection.
        
        Args:
            hand_params: [B, N, 22] normalized hand MASH parameters
            object_params: [B, N, 22] normalized object MASH parameters
            positions: [B, N, 3] normalized anchor positions
            denormalize: Whether to denormalize parameters
            hand_cmap_params: [B, N, sh_dim] contact map parameters (optional, for DexGrasp-style SPF loss)
            sample_weights: [B] per-sample weights (optional, for timestep-aware weighting)
            stats: dict of statistics
            
        Returns:
            total_loss: scalar tensor
        """        
        hand_params = hand_params[valid_timesteps_mask]
        object_params = object_params[valid_timesteps_mask]
        positions = positions[valid_timesteps_mask]
        hand_cmap_params = hand_cmap_params[valid_timesteps_mask]

        stats = get_masked_stats(stats, valid_timesteps_mask)

        # ============================================================================
        # Denormalize all parameters at once (shared by both loss types)
        # ============================================================================
        if denormalize:
            hand_params_denorm, object_params_denorm, positions_denorm, hand_cmap_params_denorm = \
                self._denormalize_params(hand_params, object_params, positions, hand_cmap_params, stats)
        else:
            hand_params_denorm = hand_params
            object_params_denorm = object_params
            positions_denorm = positions
            hand_cmap_params_denorm = hand_cmap_params
        
        # ============================================================================
        # Dispatch to specific loss implementation
        # ============================================================================
        if self.loss_type == "dexgrasp":
            return self._forward_dexgrasp_style(
                hand_params_denorm, object_params_denorm, positions_denorm, hand_cmap_params_denorm, sample_weights
            )
        else:  # "fcl"
            return self._forward_fcl_style(
                hand_params_denorm, object_params_denorm, positions_denorm
            )
    
    def _forward_fcl_style(
        self,
        hand_params_denorm: torch.Tensor,
        object_params_denorm: torch.Tensor,
        positions_denorm: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute FCL-based collision loss (OBB + broadphase + signed distance).
        
        Args:
            hand_params_denorm: [B, N, 22] denormalized hand MASH parameters
            object_params_denorm: [B, N, 22] denormalized object MASH parameters
            positions_denorm: [B, N, 3] denormalized anchor positions
            
        Returns:
            total_loss: scalar tensor
        """
        B, N, _ = hand_params_denorm.shape
        device = hand_params_denorm.device
        
        # Split parameters for batch
        hand_mask, hand_sh, hand_ortho = self.split_mash_params(hand_params_denorm)
        object_mask, object_sh, object_ortho = self.split_mash_params(object_params_denorm)
        
        # Compute sample points (batch operation)
        hand_points = self.compute_sample_points_from_mash_params(hand_mask, hand_sh, hand_ortho, positions_denorm, target="hand")
        object_points = self.compute_sample_points_from_mash_params(object_mask, object_sh, object_ortho, positions_denorm, target="object")
        
        # Reshape to [B, N, M, 3] format for OBB computation
        hand_points_flat = hand_points.reshape(B, N, -1, 3)
        object_points_flat = object_points.reshape(B, N, -1, 3)
        
        # Detach for OBB computation (FCL doesn't need gradients)
        hand_points_flat_detached = hand_points_flat.detach()
        object_points_flat_detached = object_points_flat.detach()
        
        # Compute OBBs (batch operation, using detached points)
        hand_center, hand_rotation, hand_extent = self.compute_obb_from_points(hand_points_flat_detached)
        object_center, object_rotation, object_extent = self.compute_obb_from_points(object_points_flat_detached)
        
        # Compute normals for object (always needed) and hand (if self-intersection enabled)
        # Normals are computed on detached points (used for direction, not gradient)
        M = hand_points_flat.shape[2]
        object_points_for_normals = object_points_flat.detach().reshape(B * N, M, 3)
        object_normals_batch = estimate_pointcloud_normals(
            object_points_for_normals,
            neighborhood_size=10
        )
        object_normals_flat = object_normals_batch.reshape(B, N, M, 3)
        
        hand_normals_flat = None
        if self.include_self_intersection:
            hand_points_for_normals = hand_points_flat_detached.reshape(B * N, M, 3)
            hand_normals_batch = estimate_pointcloud_normals(
                hand_points_for_normals,
                neighborhood_size=10
            )
            hand_normals_flat = hand_normals_batch.reshape(B, N, M, 3)
        
        # NEW ARCHITECTURE: Separate collision detection (for loop) from loss computation (batched)
        # ==========================================================================================
        # Phase 1: Collision Detection in for loop (NO GRADIENT operations)
        # - Uses detached OBB parameters for FCL (Python-FCL is CPU-only and non-differentiable)
        # - Purpose: Efficiently identify WHICH pairs of anchors are potentially colliding
        # - Output: List of collision pair indices
        # ==========================================================================================
        
        # Phase 1: Detect collisions for all batch samples (detached, no gradient)
        all_hand_object_pairs = []
        all_hand_self_pairs = []
        
        # Use multiprocessing if enabled and batch size is large enough
        if self.use_multiprocessing and B >= self.num_workers:
            # Prepare arguments for workers
            worker_args = []
            for b in range(B):
                hand_center_np = hand_center[b].cpu().numpy()
                hand_rotation_np = hand_rotation[b].cpu().numpy()
                hand_extent_np = hand_extent[b].cpu().numpy()
                object_center_np = object_center[b].cpu().numpy()
                object_rotation_np = object_rotation[b].cpu().numpy()
                object_extent_np = object_extent[b].cpu().numpy()
                
                worker_args.append((
                    b,
                    hand_center_np,
                    hand_rotation_np,
                    hand_extent_np,
                    object_center_np,
                    object_rotation_np,
                    object_extent_np,
                    self.include_self_intersection,
                    self.anchor_num
                ))
            
            # Run in parallel
            # Use fork() for efficiency (collision detection is pure CPU/numpy, no CUDA)
            # Worker disables signal handlers to avoid PyTorch Lightning's CUDA signal handler issues
            with get_context('fork').Pool(processes=min(self.num_workers, B)) as pool:
                results = pool.map(simple_collision_worker, worker_args)
            
            # Collect results
            for batch_idx, hand_object_pairs, hand_self_pairs in results:
                for (i, j) in hand_object_pairs:
                    all_hand_object_pairs.append((batch_idx, i, j))
                for (i, j) in hand_self_pairs:
                    all_hand_self_pairs.append((batch_idx, i, j))
        else:
            # Sequential processing (original)
            for b in range(B):
                # Extract single sample for FCL detection (all detached)
                hand_center_b = hand_center[b]
                hand_rotation_b = hand_rotation[b]
                hand_extent_b = hand_extent[b]
                
                object_center_b = object_center[b]
                object_rotation_b = object_rotation[b]
                object_extent_b = object_extent[b]
                
                # Update FCL objects and detect collisions
                self.update_obbs("hand", hand_center_b, hand_rotation_b, hand_extent_b)
                self.update_obbs("object", object_center_b, object_rotation_b, object_extent_b)
                
                # Detect hand-object collisions
                hand_object_pairs = self._detect_collisions_fcl("hand", "object", exclude_same_anchor=False)
                for (i, j) in hand_object_pairs:
                    all_hand_object_pairs.append((b, i, j))
                
                # Detect hand self-intersections
                if self.include_self_intersection:
                    hand_self_pairs = self._detect_collisions_fcl("hand", "hand", exclude_same_anchor=True)
                    for (i, j) in hand_self_pairs:
                        all_hand_self_pairs.append((b, i, j))
        
        # ==========================================================================================
        # Phase 2: Loss Computation in batch (WITH GRADIENT)
        # - Uses ORIGINAL tensors (hand_points_flat) that have gradient history
        # - Computes signed distance and penetration loss for all detected pairs
        # - Purpose: Generate gradients that flow back to hand_params to resolve collisions
        # ==========================================================================================
        device = hand_points_flat.device
        dtype = hand_points_flat.dtype
        total_loss = torch.tensor(0.0, device=device, dtype=dtype)
        
        # Compute hand-object collision loss
        if len(all_hand_object_pairs) > 0:
            hand_object_loss = self.compute_batched_penetration_loss(
                collision_pairs=all_hand_object_pairs,
                hand_points=hand_points_flat,
                object_points=object_points_flat,
                object_normals=object_normals_flat,
                is_self_intersection=False,
            )
            total_loss = total_loss + hand_object_loss
        
        # Compute hand self-intersection loss
        if self.include_self_intersection and len(all_hand_self_pairs) > 0:
            hand_self_loss = self.compute_batched_penetration_loss(
                collision_pairs=all_hand_self_pairs,
                hand_points=hand_points_flat,
                object_points=hand_points_flat,
                object_normals=hand_normals_flat,
                is_self_intersection=True,
            )
            total_loss = total_loss + hand_self_loss
        
        return total_loss
    
    def _forward_dexgrasp_style(
        self,
        hand_params_denorm: torch.Tensor,
        object_params_denorm: torch.Tensor,
        positions_denorm: torch.Tensor,
        cmap_params_denorm: Optional[torch.Tensor] = None,
        sample_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute DexGrasp-style collision loss (ERF + SPF + SRF).
        
        Args:
            hand_params_denorm: [B, N, 22] denormalized hand MASH parameters
            object_params_denorm: [B, N, 22] denormalized object MASH parameters
            positions_denorm: [B, N, 3] denormalized anchor positions
            cmap_params_denorm: [B, N, sh_dim] denormalized contact map parameters (for SPF loss)
            sample_weights: [B] per-sample weights (optional, for timestep-aware weighting)
            
        Returns:
            total_loss: scalar tensor (weighted average if sample_weights provided)
        """
        B, N, _ = hand_params_denorm.shape
        device = hand_params_denorm.device
        dtype = hand_params_denorm.dtype
        
        # Default to uniform weights if not provided
        if sample_weights is None:
            sample_weights = torch.ones(B, device=device, dtype=dtype) / B
        else:
            # Ensure weights sum to 1
            sample_weights = sample_weights / sample_weights.sum()
        
        # Split parameters
        hand_mask, hand_sh, hand_ortho = self.split_mash_params(hand_params_denorm)
        object_mask, object_sh, object_ortho = self.split_mash_params(object_params_denorm)
        
        # Sample hand and object points
        hand_points = self.compute_sample_points_from_mash_params(
            hand_mask, hand_sh, hand_ortho, positions_denorm, target="hand"
        )
        object_points = self.compute_sample_points_from_mash_params(
            object_mask, object_sh, object_ortho, positions_denorm, target="object"
        )
        
        # Reshape to [B, N*M, 3] for loss computation
        hand_points_flat = hand_points.reshape(B, -1, 3)  # [B, N*S*S, 3]
        object_points_flat = object_points.reshape(B, -1, 3)  # [B, N*S*S, 3]
        
        # Compute object normals
        M = object_points_flat.shape[1]
        object_normals = estimate_pointcloud_normals(
            object_points_flat,
            neighborhood_size=min(10, M // 2)
        )  # [B, N*S*S, 3]
        
        # Concatenate object points and normals for ERF loss
        obj_pcd_nor = torch.cat([object_points_flat, object_normals], dim=-1)  # [B, N*S*S, 6]
        
        total_loss = torch.tensor(0.0, device=device, dtype=dtype)
        
        # 1. ERF Loss (hand-object penetration)
        erf_loss = torch.tensor(0.0, device=device, dtype=dtype)
        if self.erf_weight > 0:
            erf_loss_per_sample = self.compute_ERF_loss(hand_points_flat, obj_pcd_nor, return_per_sample=True)  # [B]
            if torch.isnan(erf_loss_per_sample).any() or torch.isinf(erf_loss_per_sample).any():
                print("[WARNING] NaN/Inf in ERF loss, replacing with 0")
                erf_loss_per_sample = torch.nan_to_num(erf_loss_per_sample, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Debug: Print raw ERF loss before clipping
            if torch.rand(1).item() < 0.01:  # 1% chance
                raw_max = erf_loss_per_sample.max().item()
                raw_mean = erf_loss_per_sample.mean().item()
                print(f"[DEBUG ERF LOSS] Before clip: max={raw_max:.4f}, mean={raw_mean:.4f}")
            
            # Clip per-sample to prevent extreme values
            erf_loss_per_sample = torch.clamp(erf_loss_per_sample, min=0.0, max=100.0)
            # Weighted average
            erf_loss = (erf_loss_per_sample * sample_weights).sum()
            total_loss = total_loss + self.erf_weight * erf_loss
        
        # 2. Contact Map Alignment Loss
        cmap_loss = torch.tensor(0.0, device=device, dtype=dtype)
        if self.cmap_loss_weight > 0 and cmap_params_denorm is not None:
            # Compute cmap scalar field (contact strength)
            cmap_scalar = self.compute_cmap_scalar_field(
                cmap_params_denorm, hand_params_denorm
            )  # [B, N, S, S]
            cmap_weights_flat = cmap_scalar.reshape(B, -1)  # [B, N*S*S]
            
            # Sample contact points from cmap (weighted by contact strength)
            contact_points = self.compute_contact_points_from_cmap(
                cmap_params_denorm, hand_params_denorm, positions_denorm
            )  # [B, N, S, S, 3]
            contact_points_flat = contact_points.reshape(B, -1, 3)  # [B, N*S*S, 3]
            
            # Compute Contact Map Alignment loss (replaces SPF loss)
            cmap_loss_per_sample = self.compute_contact_map_alignment_loss(
                contact_points_flat, 
                object_points_flat, 
                cmap_weights_flat,  # cmap_scalar as target
                return_per_sample=True
            )  # [B]
            if torch.isnan(cmap_loss_per_sample).any() or torch.isinf(cmap_loss_per_sample).any():
                print("[WARNING] NaN/Inf in Contact Map Alignment loss, replacing with 0")
                cmap_loss_per_sample = torch.nan_to_num(cmap_loss_per_sample, nan=0.0, posinf=0.0, neginf=0.0)
            # Clip per-sample to prevent extreme values
            # CMAP loss raw values ~0.47-0.49 are too large, clip to 0.1 for stability
            cmap_loss_per_sample = torch.clamp(cmap_loss_per_sample, min=0.0, max=0.1)
            # Weighted average
            cmap_loss = (cmap_loss_per_sample * sample_weights).sum()
            total_loss = total_loss + self.cmap_loss_weight * cmap_loss
        
        # 3. SRF Loss (hand self-collision)
        srf_loss = torch.tensor(0.0, device=device, dtype=dtype)
        if self.srf_weight > 0 and self.include_self_intersection:
            srf_loss_per_sample = self.compute_SRF_loss(
                hand_points_flat, thres_dis=self.srf_threshold, k_neighbors=self.srf_k_neighbors, return_per_sample=True
            )  # [B]
            if torch.isnan(srf_loss_per_sample).any() or torch.isinf(srf_loss_per_sample).any():
                print("[WARNING] NaN/Inf in SRF loss, replacing with 0")
                srf_loss_per_sample = torch.nan_to_num(srf_loss_per_sample, nan=0.0, posinf=0.0, neginf=0.0)
            # Clip per-sample to prevent extreme values
            srf_loss_per_sample = torch.clamp(srf_loss_per_sample, min=0.0, max=100.0)
            # Weighted average
            srf_loss = (srf_loss_per_sample * sample_weights).sum()
            total_loss = total_loss + self.srf_weight * srf_loss
        
        # Debug: Print final loss composition (occasionally)
        if torch.rand(1).item() < 0.01:  # 1% chance
            print(f"[DEBUG LOSS COMPOSITION] "
                  f"ERF={erf_loss.item():.4f} (weight={self.erf_weight}), "
                  f"CMAP={cmap_loss.item():.4f} (weight={self.cmap_loss_weight}), "
                  f"SRF={srf_loss.item():.4f} (weight={self.srf_weight}), "
                  f"Total={total_loss.item():.4f}")
        
        # Final check
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print("[ERROR] Final total_loss is NaN/Inf, replacing with 0")
            total_loss = torch.tensor(0.0, device=device, dtype=dtype)
        
        return total_loss
