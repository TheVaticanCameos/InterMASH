import argparse
import os
from pathlib import Path
import numpy as np
import trimesh
from tqdm import trange
import torch
import math

from mash_hoi.source.PostProcess.hand_fitter import ShadowHandFitter
from ma_sh.Model.hoi_mash import HOIMash


def main():
    
    def str2bool(v):
        if isinstance(v, bool):
            return v
        v = v.strip().lower()
        if v in ("true", "1", "t", "yes", "y"):
            return True
        if v in ("false", "0", "f", "no", "n"):
            return False
        raise argparse.ArgumentTypeError("boolean value expected")
    
    parser = argparse.ArgumentParser(description="Process shadow hand fitting.")
    parser.add_argument("--mash_root", type=str, default="/data/yxz/MASH-HOI/MASH/mash-hoi/runs/outputs_edm_fix", help="Root directory of mash files")
    parser.add_argument("--object_root", type=str, default="/data2/yxz/DATASET/Original-Meshes/DexGraspNet/test", help="Root directory of object files")
    parser.add_argument("--urdf_path", type=str, default="/data2/yxz/DATASET/assets_clean/shadow//urdf/sr_grasp_description/urdf/shadowhand.urdf", help="Path to URDF file")
    parser.add_argument("--part_path", type=str, default="/data2/yxz/DATASET/assets_clean/shadow/urdf/sr_grasp_description/meshes", help="Path to part meshes")
    parser.add_argument("--target_root", type=str, default="/data/yxz/MASH-HOI/MASH/mash-hoi/generated/edm-fix-dgn", help="Target root directory for output")
    parser.add_argument("--template_info_path", type=str, default="/data2/yxz/DATASET/template_info/full_shadow_template.npy", help="Path to template info file")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use for computation")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for fitting")
    parser.add_argument("--num_iterations", type=int, default=300, help="Number of iterations for fitting")
    parser.add_argument("--single_lr", type=float, default=0.015, help="Learning rate for fitting one instance")
    parser.add_argument("--lambda_geom", type=float, default=1.0, help="Lambda for geometry loss")
    parser.add_argument("--lambda_contact", type=float, default=0.1, help="Lambda for contact loss")
    parser.add_argument("--lambda_reg", type=float, default=1.0, help="Lambda for regularization loss")
    parser.add_argument("--lambda_limits", type=float, default=0.001, help="Lambda for joint limits loss")
    parser.add_argument("--lambda_natural", type=float, default=0.0001, help="Lambda for natural pose loss")
    parser.add_argument("--verbose", type=str2bool, default=True, help="Enable verbose output")
    parser.add_argument("--use_cmap", type=str2bool, default=True, help="Use contact map")
    parser.add_argument("--use_joint_limits", type=str2bool, default=True, help="Use joint limits")
    parser.add_argument("--use_srf", type=str2bool, default=True, help="Use surface regularization")
    parser.add_argument("--vis", type=str2bool, default=False, help="Enable visualization")

    args = parser.parse_args()

    mash_files = os.listdir(args.mash_root)

    os.makedirs(os.path.join(args.target_root, "mesh"), exist_ok=True)
    os.makedirs(os.path.join(args.target_root, "qpos"), exist_ok=True)

    hand_template = np.load(args.template_info_path, allow_pickle=True)[()]
    barycentric_coords = hand_template["barycentric_coords"]
    submesh_num = hand_template["submesh_num"]
    points_per_submesh = hand_template["points_per_submesh"]
    triangle_ids = hand_template["triangle_ids"]
    permutation = hand_template["permutation"]

    shadow_fitter = ShadowHandFitter(
        urdf_path=args.urdf_path,
        part_path=args.part_path,
        device=args.device,
        submesh_num=submesh_num,
        points_per_submesh=points_per_submesh,
        barycentric_coords=barycentric_coords,
        triangle_ids=triangle_ids,
        permutation=permutation,
        use_cmap=args.use_cmap,
        use_joint_limits=args.use_joint_limits,
        use_srf=args.use_srf,
    )

    names_batch = []
    tp_list = []
    op_list = []
    cs_list = []
    tf_list = []

    for i in trange(len(mash_files)):
        mash_file = mash_files[i]
        name = Path(mash_file).stem.rsplit('_', 1)[0].split('_', 2)[2]

        hand_ply_path = os.path.join(args.target_root, "mesh", f"shadow_{name}_hand.ply")
        qpos_npy_path = os.path.join(args.target_root, "qpos", f"shadow_{name}_qpos.npy")

        if os.path.exists(hand_ply_path) and os.path.exists(qpos_npy_path):
            continue

        transform_path = os.path.join(args.object_root, f"shadow_{name}_transform.npy")
        transform = np.load(transform_path)
        transform_inv = np.linalg.inv(transform)

        hoi_mash = HOIMash.fromParamsFile(os.path.join(args.mash_root, mash_file), 32, 32, device=args.device)
        target_points = hoi_mash.toSamplePoints("hand")
        object_points = hoi_mash.toSamplePoints("object")
        contact_strengths = hoi_mash.toCMap().reshape(submesh_num, -1)

        names_batch.append(name)
        tp_list.append(target_points)
        op_list.append(object_points)
        cs_list.append(contact_strengths)
        tf_list.append(transform_inv)

        full = len(names_batch) == args.batch_size
        last = i == len(mash_files) - 1
        if full or last:

            import pdb
            pdb.set_trace()
            
            if len(names_batch) == 0:
                continue
            target_points_batch = torch.stack(tp_list, dim=0)
            object_points_batch = torch.stack(op_list, dim=0)
            contact_strengths_batch = torch.stack(cs_list, dim=0)
            contact_strengths_batch = contact_strengths_batch.clamp(0.0, 1.0)
            transform_inv_batch = torch.from_numpy(np.stack(tf_list, axis=0)).float()

            result = shadow_fitter.fit(
                target_points=target_points_batch,
                object_points=object_points_batch,
                contact_strengths=contact_strengths_batch,
                transform_inv=transform_inv_batch,
                num_iterations=args.num_iterations,
                lr=args.single_lr * math.sqrt(len(names_batch)),
                lambda_geom=args.lambda_geom,
                lambda_contact=args.lambda_contact,
                lambda_reg=args.lambda_reg,
                lambda_limits=args.lambda_limits,
                lambda_natural=args.lambda_natural,
                verbose=args.verbose,
                vis=args.vis
            )

            for j, nm in enumerate(names_batch):
                hand_mesh = trimesh.Trimesh(vertices=result['hand_verts'][j], faces=result['hand_faces'])
                hand_mesh.export(os.path.join(args.target_root, "mesh", f"shadow_{nm}_hand.ply"))
                qpos = np.concatenate([result["global_transl"][j], result["global_orient"][j], result["joint_angles"][j]], axis=0)
                
                np.save(os.path.join(args.target_root, "qpos", f"shadow_{nm}_qpos.npy"), qpos)

            names_batch = []
            tp_list = []
            op_list = []
            cs_list = []
            tf_list = []


if __name__ == "__main__":
    main()
