import os
import sys
import json
import trimesh
from tqdm import tqdm
import argparse


sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from pre_divide_object import get_object_template


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--object_source_root", type=str, default="/data2/yxz/DATASET/DRO-data/data_urdf/object")
    parser.add_argument("--object_template_root", type=str, default="/data2/yxz/DATASET/DRO-data/DRO-Original/shadow/object_template")
    parser.add_argument("--split_path", type=str, default="/data2/yxz/DATASET/DRO-data/CMapDataset_filtered/split_train_validate_objects.json")
    parser.add_argument("--submesh_num", type=int, default=256)
    parser.add_argument("--points_per_submesh", type=int, default=1024)
    
    args = parser.parse_args()

    object_template_root = args.object_template_root

    os.makedirs(object_template_root, exist_ok=True)

    object_source_root = args.object_source_root

    split_path = args.split_path

    with open(split_path, "r") as f:
        split_info = json.load(f)

    train_objects = split_info["train"]
    val_objects = split_info["validate"]

    pbar = tqdm(train_objects + val_objects)
    for object_full_name in pbar:
        pbar.set_description(object_full_name)
        object_type = object_full_name.split("+", 1)[0]
        object_name = object_full_name.split("+", 1)[1]    
        object_path = os.path.join(object_source_root, object_type, object_name, object_name + ".stl")

        object_mesh = trimesh.load(object_path)
        object_mesh.export("test.ply")
        object_path = "test.ply"

        save_path = os.path.join(object_template_root, f"{object_type}+{object_name}.npy")
        if not os.path.exists(save_path):
            get_object_template(
                object_path,
                save_path,
                submesh_num=args.submesh_num,
                points_per_submesh=args.points_per_submesh,
            )
