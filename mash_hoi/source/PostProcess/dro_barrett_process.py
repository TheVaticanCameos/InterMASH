import os
import math
import numpy as np
import torch
import pytorch_lightning as pl
import trimesh
import pickle
import open3d as o3d
from tqdm import tqdm

from mash_hoi.source.PostProcess.hand_fitter import BarrettHandFitter
from ma_sh.Model import HOIMash


pl.seed_everything(42)


if __name__ == "__main__":
    mash_root = "/data/yxz/MASH-HOI/MASH/mash_hoi/Results/MASH/exp_ddpm_graph_dro_pe_decoder"

    urdf_path = "/data/yxz/MyProjects/DATASET/assets_clean/barrett/model_extended.urdf"
    part_path = "/data/yxz/MyProjects/DATASET/assets_clean/barrett/meshes"

    template_info_path = "/data/yxz/MyProjects/DATASET/template_info/sphere_barrett_template.npy"
    hand_template = np.load(template_info_path, allow_pickle=True)[()]
    barycentric_coords = hand_template["barycentric_coords"]
    submesh_num = hand_template["submesh_num"]
    points_per_submesh = hand_template["points_per_submesh"]
    triangle_ids = hand_template["triangle_ids"]
    permutation = hand_template["permutation"]

    # original_root = "/data2/yxz/DATASET/DRO-data/DRO-Original/barrett/val"

    target_root = "/data/yxz/MASH-HOI/MASH/mash_hoi/Results/Mesh/exp_ddpm_graph_dro_pe_decoder/barrett"

    os.makedirs(os.path.join(target_root, "mesh"), exist_ok=True)
    os.makedirs(os.path.join(target_root, "qpos"), exist_ok=True)

    sample_num = 100

    mash_files = [name for name in os.listdir(mash_root) if name.startswith("barrett_gen")]

    names = list(set([mash_name.split('_', 2)[2].rsplit('_', 2)[0] for mash_name in mash_files]))

    data_dict = {name: [] for name in names}

    for mash_file in mash_files:
        name = mash_file.split('_', 2)[2].rsplit('_', 2)[0]
        idx = int(mash_file.split('_', 2)[2].rsplit('_', 2)[2].split(".")[0])
        data_dict[name].append(idx)

    barrett_fitter = BarrettHandFitter(
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
    generated_results["desc"] = "DDPM+Sphere"
    generated_results["sample_qpos"] = {}

    for name in names:
        # if name != "contactdb+apple":
        #     continue

        total_num = len(data_dict[name])
        assert total_num == sample_num
        
        tp_list = []
        op_list = []
        cs_list = []
        tf_list = []
        on_list = []
        for idx in tqdm(range(total_num), desc=f"Processing {name}"):
            mash_name = f"barrett_gen_{name}_object_{idx}.npy"
            mash_path = os.path.join(mash_root, mash_name)
            hoi_mash = HOIMash.fromParamsFile(mash_path, 32, 32, device="cuda")
            target_points = hoi_mash.toSamplePoints("hand")
            object_points = hoi_mash.toSamplePoints("object")
            contact_strengths = hoi_mash.toCMap("object").reshape(submesh_num, -1)

            # transform_path = os.path.join(original_root, f"barrett_{name}_{idx}_transform.npy")
            # transform = np.load(transform_path)
            # transform_inv = np.linalg.inv(transform)
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

        # result = barrett_fitter.fit(
        #     target_points=target_points_batch,
        #     object_points=object_points_batch,
        #     contact_strengths=contact_strengths_batch,
        #     transform_inv=transform_inv_batch,
        #     object_normals=normals_batch,
        #     num_iterations=500,
        #     lr=0.005 * math.sqrt(total_num),
        #     lambda_geom=1.0,
        #     lambda_contact=0.,
        #     lambda_reg=1.0,
        #     lambda_limits=0.001,
        #     lambda_natural=0.0001,
        #     lambda_spf=0.1,
        #     lambda_srf=0.1,
        #     lambda_erf=0.8,
        #     d_threshold=0.005,
        #     contact_threshold=0.015,
        #     verbose=True,
        #     vis=False,
        #     vis_id=5,
        #     stop_error=0.005,
        #     reg_start_iter=200
        # )
        result = barrett_fitter.pure_geom_fit(
            target_points=target_points_batch,
            object_points=object_points_batch,
            transform_inv=transform_inv_batch,
            num_iterations=500,
            lr=0.005 * math.sqrt(total_num),
            verbose=True,
            vis=False,
            vis_id=5,
            stop_error=0.005,
        )

        total_qpos = []

        for i in range(total_num):
            hand_mesh = trimesh.Trimesh(vertices=result["hand_verts"][i], faces=result["hand_faces"])
            hand_mesh.export(os.path.join(target_root, "mesh", f"barrett_{name}_{i}_hand.ply"))

            qpos = np.concatenate([result["rot_mat"][i] @ result["global_transl"][i], 
                                result["global_rot6d"][i], 
                                result["joint_angles"][i]], axis=0)

            total_qpos.append(qpos)

        all_qpos = np.stack(total_qpos)
        generated_results["sample_qpos"][name] = all_qpos

    with open(os.path.join(target_root, "res_DRO-barrett-graphdiff-pe-decoder.pkl"), 'wb') as f:
        pickle.dump(generated_results, f)
