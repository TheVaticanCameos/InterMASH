import os
import sys
import trimesh
from tqdm import tqdm
import argparse


sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from pre_divide_object import get_object_template


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--object_source_root", type=str, default="/home/yxz/MyProjects/DATASET/Original-Dataset/Original-DexGraspNet/DexGraspNet/meshdata")
    parser.add_argument("--object_template_root", type=str, default="/home/yxz/MyProjects/DATASET/Original-Meshes/DexGraspNet/object_template")
    parser.add_argument("--submesh_num", type=int, default=256)
    parser.add_argument("--points_per_submesh", type=int, default=1024)
    
    args = parser.parse_args()

    os.makedirs(args.object_template_root, exist_ok=True)
    object_names = [name for name in os.listdir(args.object_source_root) if not name.startswith(".")]
    pbar = tqdm(object_names)
    for object_name in pbar:
        pbar.set_description(object_name)
        object_path = os.path.join(args.object_source_root, object_name, "coacd/decomposed.obj")
        
        object_mesh = trimesh.load(object_path)
        object_mesh.export("test.ply")
        obejct_path = "test.ply"

        save_path = os.path.join(args.object_template_root, object_name + ".npy")
        if not os.path.exists(save_path):
            get_object_template(
                object_path,
                save_path,
                submesh_num=args.submesh_num,
                points_per_submesh=args.points_per_submesh,
            )
