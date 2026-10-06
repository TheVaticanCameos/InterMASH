import os
import argparse
import h5py
import numpy as np
from tqdm import tqdm
from typing import Dict



# def _write_to_group_train(grp: h5py._hl.group.Group, data_dict: Dict, name: str) -> bool:
#     positions = data_dict["positions"]
#     hand_params = np.concatenate([data_dict["hand_params"]["mask_params"], data_dict["hand_params"]["sh_params"], data_dict["hand_params"]["ortho_poses"]], axis=-1)
#     object_params = np.concatenate([data_dict["object_params"]["mask_params"], data_dict["object_params"]["sh_params"], data_dict["object_params"]["ortho_poses"]], axis=-1)
#     cmap_params = data_dict["cmap_params"]["sh_params"]

#     grp.create_dataset("positions", data=np.asarray(positions))
#     grp.create_dataset("hand_params", data=np.asarray(hand_params))
#     grp.create_dataset("object_params", data=np.asarray(object_params))
#     grp.create_dataset("cmap_params", data=np.asarray(cmap_params))
#     grp.attrs["name"] = name

#     return True


# def _write_to_group_test(grp: h5py._hl.group.Group, data_dict: Dict, name: str) -> bool:
#     positions = data_dict["positions"]
#     object_params = np.concatenate([data_dict["object_params"]["mask_params"], data_dict["object_params"]["sh_params"], data_dict["object_params"]["ortho_poses"]], axis=-1)
    
#     grp.create_dataset("positions", data=np.asarray(positions))
#     grp.create_dataset("object_params", data=np.asarray(object_params))
#     grp.attrs["name"] = name

#     return True


def _write_to_group_train(grp: h5py._hl.group.Group, data_dict: Dict, name: str) -> bool:
    positions = data_dict["positions"]
    hand_params = np.concatenate([
        data_dict["hand_params"]["mask_params"], 
        data_dict["hand_params"]["sh_params"], 
        data_dict["hand_params"]["ortho_poses"], 
        data_dict["hand_params"]["cmap_params"]
    ], axis=-1)
    object_params = np.concatenate([
        data_dict["object_params"]["mask_params"], 
        data_dict["object_params"]["sh_params"], 
        data_dict["object_params"]["ortho_poses"], 
        data_dict["object_params"]["cmap_params"]
    ], axis=-1)

    grp.create_dataset("positions", data=np.asarray(positions))
    grp.create_dataset("hand_params", data=np.asarray(hand_params))
    grp.create_dataset("object_params", data=np.asarray(object_params))
    grp.attrs["name"] = name

    return True


def _write_to_group_test(grp: h5py._hl.group.Group, data_dict: Dict, name: str) -> bool:
    positions = data_dict["positions"]
    object_params = np.concatenate([
        data_dict["object_params"]["mask_params"], 
        data_dict["object_params"]["sh_params"], 
        data_dict["object_params"]["ortho_poses"]
    ], axis=-1)
    
    grp.create_dataset("positions", data=np.asarray(positions))
    grp.create_dataset("object_params", data=np.asarray(object_params))
    grp.attrs["name"] = name

    return True


def merge_npy_to_h5(data_root: str, output_path: str, mode: str = "train") -> None:
    files = [n for n in os.listdir(data_root) if n.endswith(".npy")]
    with h5py.File(output_path, "w") as hf:
        for idx, name in enumerate(tqdm(files)):
            file_path = os.path.join(data_root, name)
            data = np.load(file_path, allow_pickle=True)[()]
            grp = hf.create_group(f"{idx}")
            if mode == "train":
                _write_to_group_train(grp, data, name)
            elif mode == "test":
                _write_to_group_test(grp, data, name)
            else:
                raise ValueError(f"Unknown mode: {mode}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="/data2/yxz/DATASET/DexGraspNet/DexGraspNet-HOI-Dataset/NEW-Dataset/train")
    parser.add_argument("--output_path", type=str, default="merged_data.h5")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"])
    args = parser.parse_args()
    merge_npy_to_h5(args.data_root, args.output_path, args.mode)

