import os
import math
import argparse
import numpy as np
import torch
import pytorch_lightning as pl
# import open3d as o3d
import trimesh
import pickle
from tqdm import tqdm

from mash_hoi.source.PostProcess.hand_fitter import ShadowHandFitter
from ma_sh.Model import HOIMash


pl.seed_everything(42)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mash_root", type=str, default="/home/yxz/MyProjects/MASH-HOI/mash_hoi/Results/MASH/exp_ddpm_phys_dgn")
    parser.add_argument("--urdf_path", type=str, default="/home/yxz/MyProjects/assets_clean/shadow/urdf/sr_grasp_description/urdf/shadowhand.urdf")
    parser.add_argument("--part_path", type=str, default="/home/yxz/MyProjects/assets_clean/shadow/urdf/sr_grasp_description/meshes")
    parser.add_argument("--template_info_path", type=str, default="/mnt/d/yxz/DATASET/template_info/sphere_shadow_template.npy")
    parser.add_argument("--original_root", type=str, default="/home/yxz/MyProjects/DATASET/Original-Dataset/Original-DexGraspNet/DexGraspNet")
    parser.add_argument("--target_root", type=str, default="/home/yxz/MyProjects/MASH-HOI/mash_hoi/Results/Mesh/exp_ddpm_phys_dgn")
    parser.add_argument("--sample_num", type=int, default=64)
    parser.add_argument("--part_num", type=int, default=1)
    parser.add_argument("--current_num", type=int, default=0)
    args = parser.parse_args()

    mash_root = args.mash_root
    urdf_path = args.urdf_path
    part_path = args.part_path
    template_info_path = args.template_info_path
    original_root = args.original_root
    target_root = args.target_root
    sample_num = args.sample_num

    hand_template = np.load(template_info_path, allow_pickle=True)[()]
    barycentric_coords = hand_template["barycentric_coords"]
    submesh_num = hand_template["submesh_num"]
    points_per_submesh = hand_template["points_per_submesh"]
    triangle_ids = hand_template["triangle_ids"]
    permutation = hand_template["permutation"]

    os.makedirs(os.path.join(target_root, "mesh"), exist_ok=True)
    os.makedirs(os.path.join(target_root, "qpos"), exist_ok=True)

    mash_files = [name for name in os.listdir(mash_root) if name.endswith(".npy")]
    names = list(set([mash_name.split('_', 2)[2].rsplit('_', 2)[0] for mash_name in mash_files]))
    
    # Split names into parts
    names.sort()
    total_names = len(names)
    names_per_part = math.ceil(total_names / args.part_num)
    start_idx = args.current_num * names_per_part
    end_idx = min((args.current_num + 1) * names_per_part, total_names)
    names = names[start_idx:end_idx]
    
    print(f"Processing part {args.current_num}/{args.part_num}, names: {len(names)}/{total_names}")

    data_dict = {name: [] for name in names}

    for mash_file in mash_files:
        name = mash_file.split('_', 2)[2].rsplit('_', 2)[0]
        if name in data_dict:
            idx = int(mash_file.split('_', 2)[2].rsplit('_', 2)[2].split(".")[0])
            data_dict[name].append(idx)

    with open(os.path.join(original_root, "scales.pkl"), "rb") as f:
        scales = pickle.load(f)
    
    shadow_fitter = ShadowHandFitter(
        urdf_path=urdf_path,
        part_path=part_path,
        device="cuda",
        submesh_num=submesh_num,
        points_per_submesh=points_per_submesh,
        barycentric_coords=barycentric_coords,
        triangle_ids=triangle_ids,
        permutation=permutation,
        use_cmap=True,
        use_joint_limits=True,
        use_srf=True,
    )

    generated_results = {}
    generated_results["method"] = "MASH-HOI"
    generated_results["desc"] = "DDPM+Sphere+DGN"
    generated_results["sample_qpos"] = {}
    
    for name in names:

        object_path = os.path.join(original_root, "obj_scale_urdf", f"{name}.obj")
        object_mesh = trimesh.load(object_path)
        object_mesh.vertices = object_mesh.vertices * scales[name]

        total_num = len(data_dict[name])
        assert total_num == sample_num

        tp_list = []
        op_list = []
        cs_list = []
        tf_list = []
        # on_list = []
        for idx in tqdm(range(total_num), desc=f"Processing {name}"):
            mash_name = f"gen_shadow_{name}_object_{idx}.npy"
            mash_path = os.path.join(mash_root, mash_name)
            hoi_mash = HOIMash.fromParamsFile(mash_path, 32, 32, device="cuda")
            target_points = hoi_mash.toSamplePoints("hand")
            object_points = hoi_mash.toSamplePoints("object")
            contact_strengths = hoi_mash.toCMap().reshape(submesh_num, -1)

            transform_inv = np.eye(4)

            # pts_flat = object_points.reshape(-1, 3).cpu().numpy()
            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(pts_flat)
            # pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

            # center = np.mean(pts_flat, axis=0)
            # pcd.orient_normals_towards_camera_location(camera_location=center)
            # # Flip normals to point OUT (assuming centroid is inside)
            # normals = -np.asarray(pcd.normals)
            # normals = torch.from_numpy(normals).to(object_points.device, dtype=object_points.dtype).reshape(object_points.shape)

            tp_list.append(target_points)
            op_list.append(object_points)
            cs_list.append(contact_strengths)
            tf_list.append(transform_inv)
            # on_list.append(normals)
        
        target_points_batch = torch.stack(tp_list, dim=0)
        object_points_batch = torch.stack(op_list, dim=0)
        contact_strengths_batch = torch.stack(cs_list, dim=0)
        transform_inv_batch = torch.from_numpy(np.stack(tf_list, axis=0)).float()
        # normals_batch = torch.stack(on_list, dim=0)

        result = shadow_fitter.pure_geom_fit(
            target_points=target_points_batch,
            object_points=object_points_batch,
            transform_inv=transform_inv_batch,
            num_iterations=500,
            lr=0.005 * math.sqrt(total_num),    # 0.005
            verbose=True,
            vis=False,
            vis_id=5,
            stop_error=0.001,
        )

        total_qpos = []

        for i in range(total_num):
            hand_mesh = trimesh.Trimesh(vertices=result["hand_verts"][i], faces=result["hand_faces"])
            hand_mesh.export(os.path.join(target_root, "mesh", f"shadow_{name}_{i}_hand.ply"))

            qpos = np.concatenate([result["rot_mat"][i] @ result["global_transl"][i], 
                                result["global_rot6d"][i], 
                                result["joint_angles"][i]], axis=0)
            
            total_qpos.append(qpos)

        all_qpos = np.stack(total_qpos)
        generated_results["sample_qpos"][name] = all_qpos

    with open(os.path.join(target_root, f"res_DGN-shadow-phys-loss-{args.current_num}.pkl"), 'wb') as f:
        pickle.dump(generated_results, f)
