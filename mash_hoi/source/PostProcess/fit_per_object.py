import os
import math
import numpy as np
import torch
import pytorch_lightning as pl
import trimesh
import pickle
import argparse

from mash_hoi.source.PostProcess.hand_fitter import ShadowHandFitter

from ma_sh.Model.hoi_mash import HOIMash

pl.seed_everything(42)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mash_root", type=str, default="/home/yxz/MyProjects/MASH-HOI/mash-hoi/Results/MASH/exp_edm_fixed_udg")
    parser.add_argument("--object_root", type=str, default="/mnt/d/yxz/DATASET/Original-Meshes/UniDexGrasp/test")
    parser.add_argument("--urdf_path", type=str, default="/home/yxz/MyProjects/DATASET/assets_clean/shadow//urdf/sr_grasp_description/urdf/shadowhand.urdf")
    parser.add_argument("--part_path", type=str, default="/home/yxz/MyProjects/DATASET/assets_clean/shadow/urdf/sr_grasp_description/meshes")
    parser.add_argument("--target_root", type=str, default="/home/yxz/MyProjects/MASH-HOI/mash-hoi/Results/Mesh/exp_edm_fixed_udg")
    parser.add_argument("--template_info_path", type=str, default="//mnt/d/yxz/DATASET/template_info/full_shadow_template.npy")
    parser.add_argument("--part_num", type=int, default=1)
    parser.add_argument("--current_num", type=int, default=0)

    args = parser.parse_args()

    mash_root = args.mash_root
    object_root = args.object_root
    urdf_path = args.urdf_path
    part_path = args.part_path
    target_root = args.target_root
    template_info_path = args.template_info_path
    part_num = args.part_num
    current_num = args.current_num

    os.makedirs(os.path.join(target_root, "mesh"), exist_ok=True)
    os.makedirs(os.path.join(target_root, "qpos"), exist_ok=True)

    sample_num = 64

    mash_files = os.listdir(mash_root)

    names = list(set([mash_name.split('_', 2)[2].rsplit('_', 2)[0] for mash_name in mash_files]))

    data_dict = {name: [] for name in names}

    for mash_file in mash_files:
        name = mash_file.split('_', 2)[2].rsplit('_', 2)[0]
        idx = int(mash_file.split('_', 2)[2].rsplit('_', 2)[1])
        data_dict[name].append(idx)

    hand_template = np.load(template_info_path, allow_pickle=True)[()]
    barycentric_coords = hand_template["barycentric_coords"]
    submesh_num = hand_template["submesh_num"]
    points_per_submesh = hand_template["points_per_submesh"]
    triangle_ids = hand_template["triangle_ids"]
    permutation = hand_template["permutation"]

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
    generated_results["desc"] = "EDM+DGN"
    generated_results["sample_qpos"] = {}

    # Split names into part_num chunks and process only chunk current_num
    total_names = len(names)
    part_size = (total_names + part_num - 1) // part_num
    start_idx = current_num * part_size
    end_idx = min(start_idx + part_size, total_names)
    current_names = names[start_idx:end_idx]

    for name in current_names:
        try:
            total_num = len(data_dict[name])
            # Randomly select sample_num distinct elements from 0, 1, ..., total_num - 1
            if total_num <= sample_num:
                selected_indices = data_dict[name]
            else:
                selected_indices = np.random.choice(data_dict[name], size=sample_num, replace=False).tolist()
            
            # Truncate selected_indices if needed so its length is a multiple of 8
            remainder = len(selected_indices) % 8
            if remainder != 0:
                selected_indices = selected_indices[:-remainder]

            tp_list = []
            op_list = []
            cs_list = []
            tf_list = []
            for idx in selected_indices:
                mash_name = f"gen_shadow_{name}_{idx}_object.npy"
                mash_path = os.path.join(mash_root, mash_name)
                hoi_mash = HOIMash.fromParamsFile(mash_path, 32, 32, device="cuda")
                target_points = hoi_mash.toSamplePoints("hand")
                object_points = hoi_mash.toSamplePoints("object")
                contact_strengths = hoi_mash.toCMap().reshape(submesh_num, -1)

                transform_path = os.path.join(object_root, f"shadow_{name}_{idx}_transform.npy")
                transform = np.load(transform_path)
                transform_inv = np.linalg.inv(transform)

                tp_list.append(target_points)
                op_list.append(object_points)
                cs_list.append(contact_strengths)
                tf_list.append(transform_inv)

            target_points_batch = torch.stack(tp_list, dim=0)
            object_points_batch = torch.stack(op_list, dim=0)
            contact_strengths_batch = torch.stack(cs_list, dim=0)
            transform_inv_batch = torch.from_numpy(np.stack(tf_list, axis=0)).float()

            result = shadow_fitter.fit(
                target_points_batch,
                object_points_batch,
                contact_strengths_batch,
                transform_inv_batch,
                num_iterations=300,
                lr=0.01 * math.sqrt(len(selected_indices)),
                lambda_geom=1.0,
                lambda_contact=0.5,
                lambda_reg=1.0,
                lambda_limits=0.001,
                lambda_natural=0.0001,
                lambda_spf=1.0,
                contact_threshold=0.01,
                verbose=True,
                vis=False,
                vis_id=0,
                stop_error=0.005
            )

            total_qpos = []

            for i in range(len(selected_indices)):
                selected_idx = selected_indices[i]
                hand_mesh = trimesh.Trimesh(vertices=result["hand_verts"][i], faces=result["hand_faces"])
                hand_mesh.export(os.path.join(target_root, "mesh", f"shadow_{name}_{selected_idx}_hand.ply"))

                qpos = np.concatenate([result["rot_mat"][i] @ result["global_transl"][i], 
                                    result["global_rot6d"][i], 
                                    result["joint_angles"][i]], axis=0)

                total_qpos.append(qpos)

            all_qpos = np.stack(total_qpos)
            generated_results["sample_qpos"][name] = all_qpos
        
        except Exception as e:
            print(f"Error processing {name}: {e}")
            continue

    with open(os.path.join(target_root, f"res_mash-hoi-{current_num}.pkl"), 'wb') as f:
        pickle.dump(generated_results, f)


if __name__ == "__main__":
    main()
