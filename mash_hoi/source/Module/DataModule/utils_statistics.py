import h5py
import numpy as np
import torch
from tqdm import tqdm

from typing import Dict, Any

from mash_hoi.source.Utils.constant import MASK_DIM, SH_DIM, ORTHO_DIM


# def get_statistics(data_path: str) -> Dict[str, Any]:
#     with h5py.File(data_path, 'r') as hf:
#         data_num = len(hf.keys())
#         all_hand_mask_params = [None] * data_num
#         all_hand_sh_params = [None] * data_num
#         all_hand_ortho_poses = [None] * data_num
#         all_object_mask_params = [None] * data_num
#         all_object_sh_params = [None] * data_num
#         all_object_ortho_poses = [None] * data_num
#         all_cmap_params = [None] * data_num
#         all_positions = [None] * data_num

#         for count in tqdm(range(data_num)):
#             hand_params = hf[str(count)]["hand_params"][:]
#             object_params = hf[str(count)]["object_params"][:]
#             cmap_params = hf[str(count)]["cmap_params"][:]
#             positions = hf[str(count)]["positions"][:]
            
#             all_hand_mask_params[count] = hand_params[:, :MASK_DIM]
#             all_hand_sh_params[count] = hand_params[:, MASK_DIM:MASK_DIM+SH_DIM]
#             all_hand_ortho_poses[count] = hand_params[:, MASK_DIM+SH_DIM:]
#             all_object_mask_params[count] = object_params[:, :MASK_DIM]
#             all_object_sh_params[count] = object_params[:, MASK_DIM:MASK_DIM+SH_DIM]
#             all_object_ortho_poses[count] = object_params[:, MASK_DIM+SH_DIM:]
#             all_cmap_params[count] = cmap_params
#             all_positions[count] = positions
        
#     all_hand_mask_params = np.concatenate(all_hand_mask_params, axis=0)
#     all_hand_sh_params = np.concatenate(all_hand_sh_params, axis=0)
#     all_hand_ortho_poses = np.concatenate(all_hand_ortho_poses, axis=0)
#     all_object_mask_params = np.concatenate(all_object_mask_params, axis=0)
#     all_object_sh_params = np.concatenate(all_object_sh_params, axis=0)
#     all_object_ortho_poses = np.concatenate(all_object_ortho_poses, axis=0)
#     all_cmap_params = np.concatenate(all_cmap_params, axis=0)
#     all_positions = np.concatenate(all_positions, axis=0)
    
#     eps = 1e-8
#     statistics = {
#         "positions": {
#             "mean": np.mean(all_positions, axis=0),
#             "std": np.std(all_positions, axis=0) + eps
#         },
#         "hand": {
#             "mask_params": {
#                 "mean": np.mean(all_hand_mask_params, axis=0),
#                 "std": np.std(all_hand_mask_params, axis=0) + eps
#             },
#             "sh_params": {
#                 "mean": np.mean(all_hand_sh_params, axis=0),
#                 "std": np.std(all_hand_sh_params, axis=0) + eps
#             },
#             "ortho_poses": {
#                 "mean": np.mean(all_hand_ortho_poses, axis=0),
#                 "std": np.std(all_hand_ortho_poses, axis=0) + eps
#             }
#         },
#         "object": {
#             "mask_params": {
#                 "mean": np.mean(all_object_mask_params, axis=0),
#                 "std": np.std(all_object_mask_params, axis=0) + eps
#             },
#             "sh_params": {
#                 "mean": np.mean(all_object_sh_params, axis=0),
#                 "std": np.std(all_object_sh_params, axis=0) + eps
#             },
#             "ortho_poses": {
#                 "mean": np.mean(all_object_ortho_poses, axis=0),
#                 "std": np.std(all_object_ortho_poses, axis=0) + eps
#             }
#         },
#         "cmap_params": {
#             "sh_params": {
#                 "mean": np.mean(all_cmap_params, axis=0),
#                 "std": np.std(all_cmap_params, axis=0) + eps
#             }
#         }
#     }

#     return statistics


# def to_multi_hand_format(statistics: Dict) -> Dict[str, torch.Tensor]:
#     multi_hand_statistics = {
#         "mean_positions": torch.from_numpy(statistics["positions"]["mean"]),
#         "std_positions": torch.from_numpy(statistics["positions"]["std"]),

#         "mean_hand_mask": torch.from_numpy(statistics["hand"]["mask_params"]["mean"]),
#         "std_hand_mask": torch.from_numpy(statistics["hand"]["mask_params"]["std"]),
#         "mean_hand_sh": torch.from_numpy(statistics["hand"]["sh_params"]["mean"]),
#         "std_hand_sh": torch.from_numpy(statistics["hand"]["sh_params"]["std"]),
#         "mean_hand_ortho": torch.from_numpy(statistics["hand"]["ortho_poses"]["mean"]),
#         "std_hand_ortho": torch.from_numpy(statistics["hand"]["ortho_poses"]["std"]),
        
#         "mean_obj_mask": torch.from_numpy(statistics["object"]["mask_params"]["mean"]),
#         "std_obj_mask": torch.from_numpy(statistics["object"]["mask_params"]["std"]),
#         "mean_obj_sh": torch.from_numpy(statistics["object"]["sh_params"]["mean"]),
#         "std_obj_sh": torch.from_numpy(statistics["object"]["sh_params"]["std"]),
#         "mean_obj_ortho": torch.from_numpy(statistics["object"]["ortho_poses"]["mean"]),
#         "std_obj_ortho": torch.from_numpy(statistics["object"]["ortho_poses"]["std"]),

#         "mean_cmap": torch.from_numpy(statistics["cmap_params"]["sh_params"]["mean"]),
#         "std_cmap": torch.from_numpy(statistics["cmap_params"]["sh_params"]["std"]),
#     }
#     return multi_hand_statistics


def get_statistics(data_path: str) -> Dict[str, Any]:
    with h5py.File(data_path, 'r') as hf:
        data_num = len(hf.keys())
        all_hand_mask_params = [None] * data_num
        all_hand_sh_params = [None] * data_num
        all_hand_ortho_poses = [None] * data_num
        all_hand_cmap_params = [None] * data_num
        all_object_mask_params = [None] * data_num
        all_object_sh_params = [None] * data_num
        all_object_ortho_poses = [None] * data_num
        all_object_cmap_params = [None] * data_num
        all_positions = [None] * data_num

        for count in tqdm(range(data_num)):
            hand_params = hf[str(count)]["hand_params"][:]
            object_params = hf[str(count)]["object_params"][:]
            positions = hf[str(count)]["positions"][:]
            
            all_hand_mask_params[count] = hand_params[:, :MASK_DIM]
            all_hand_sh_params[count] = hand_params[:, MASK_DIM:MASK_DIM+SH_DIM]
            all_hand_ortho_poses[count] = hand_params[:, MASK_DIM+SH_DIM:MASK_DIM+SH_DIM+ORTHO_DIM]
            all_hand_cmap_params[count] = hand_params[:, MASK_DIM+SH_DIM+ORTHO_DIM:]
            
            all_object_mask_params[count] = object_params[:, :MASK_DIM]
            all_object_sh_params[count] = object_params[:, MASK_DIM:MASK_DIM+SH_DIM]
            all_object_ortho_poses[count] = object_params[:, MASK_DIM+SH_DIM:MASK_DIM+SH_DIM+ORTHO_DIM]
            all_object_cmap_params[count] = object_params[:, MASK_DIM+SH_DIM+ORTHO_DIM:]
            
            all_positions[count] = positions
        
    all_hand_mask_params = np.concatenate(all_hand_mask_params, axis=0)
    all_hand_sh_params = np.concatenate(all_hand_sh_params, axis=0)
    all_hand_ortho_poses = np.concatenate(all_hand_ortho_poses, axis=0)
    all_hand_cmap_params = np.concatenate(all_hand_cmap_params, axis=0)
    all_object_mask_params = np.concatenate(all_object_mask_params, axis=0)
    all_object_sh_params = np.concatenate(all_object_sh_params, axis=0)
    all_object_ortho_poses = np.concatenate(all_object_ortho_poses, axis=0)
    all_object_cmap_params = np.concatenate(all_object_cmap_params, axis=0)
    all_positions = np.concatenate(all_positions, axis=0)
    
    eps = 1e-8
    statistics = {
        "positions": {
            "mean": np.mean(all_positions, axis=0),
            "std": np.std(all_positions, axis=0) + eps
        },
        "hand": {
            "mask_params": {
                "mean": np.mean(all_hand_mask_params, axis=0),
                "std": np.std(all_hand_mask_params, axis=0) + eps
            },
            "sh_params": {
                "mean": np.mean(all_hand_sh_params, axis=0),
                "std": np.std(all_hand_sh_params, axis=0) + eps
            },
            "ortho_poses": {
                "mean": np.mean(all_hand_ortho_poses, axis=0),
                "std": np.std(all_hand_ortho_poses, axis=0) + eps
            },
            "cmap_params": {
                "mean": np.mean(all_hand_cmap_params, axis=0),
                "std": np.std(all_hand_cmap_params, axis=0) + eps
            }
        },
        "object": {
            "mask_params": {
                "mean": np.mean(all_object_mask_params, axis=0),
                "std": np.std(all_object_mask_params, axis=0) + eps
            },
            "sh_params": {
                "mean": np.mean(all_object_sh_params, axis=0),
                "std": np.std(all_object_sh_params, axis=0) + eps
            },
            "ortho_poses": {
                "mean": np.mean(all_object_ortho_poses, axis=0),
                "std": np.std(all_object_ortho_poses, axis=0) + eps
            },
            "cmap_params": {
                "mean": np.mean(all_object_cmap_params, axis=0),
                "std": np.std(all_object_cmap_params, axis=0) + eps
            }
        }
    }

    return statistics


def to_multi_hand_format(statistics: Dict) -> Dict[str, torch.Tensor]:
    """
    Convert statistics to multi-hand format.
    
    Handles missing cmap_params gracefully by using dummy values if not present.
    """
    multi_hand_statistics = {
        "mean_positions": torch.from_numpy(statistics["positions"]["mean"]),
        "std_positions": torch.from_numpy(statistics["positions"]["std"]),

        "mean_hand_mask": torch.from_numpy(statistics["hand"]["mask_params"]["mean"]),
        "std_hand_mask": torch.from_numpy(statistics["hand"]["mask_params"]["std"]),
        "mean_hand_sh": torch.from_numpy(statistics["hand"]["sh_params"]["mean"]),
        "std_hand_sh": torch.from_numpy(statistics["hand"]["sh_params"]["std"]),
        "mean_hand_ortho": torch.from_numpy(statistics["hand"]["ortho_poses"]["mean"]),
        "std_hand_ortho": torch.from_numpy(statistics["hand"]["ortho_poses"]["std"]),
        "mean_hand_cmap": torch.from_numpy(statistics["hand"]["cmap_params"]["mean"]),
        "std_hand_cmap": torch.from_numpy(statistics["hand"]["cmap_params"]["std"]),
        
        "mean_obj_mask": torch.from_numpy(statistics["object"]["mask_params"]["mean"]),
        "std_obj_mask": torch.from_numpy(statistics["object"]["mask_params"]["std"]),
        "mean_obj_sh": torch.from_numpy(statistics["object"]["sh_params"]["mean"]),
        "std_obj_sh": torch.from_numpy(statistics["object"]["sh_params"]["std"]),
        "mean_obj_ortho": torch.from_numpy(statistics["object"]["ortho_poses"]["mean"]),
        "std_obj_ortho": torch.from_numpy(statistics["object"]["ortho_poses"]["std"]),
    }
    
    # Handle cmap_params - add if they exist, otherwise use dummy values
    if "cmap_params" in statistics.get("hand", {}):
        multi_hand_statistics["mean_hand_cmap"] = torch.from_numpy(statistics["hand"]["cmap_params"]["mean"])
        multi_hand_statistics["std_hand_cmap"] = torch.from_numpy(statistics["hand"]["cmap_params"]["std"])
        # Also add mean_cmap and std_cmap for compatibility with SphereDROTrainDataset
        multi_hand_statistics["mean_cmap"] = torch.from_numpy(statistics["hand"]["cmap_params"]["mean"])
        multi_hand_statistics["std_cmap"] = torch.from_numpy(statistics["hand"]["cmap_params"]["std"])
    else:
        # Create dummy cmap stats (1D tensor, value doesn't matter as we don't use cmap)
        multi_hand_statistics["mean_hand_cmap"] = torch.zeros(1)
        multi_hand_statistics["std_hand_cmap"] = torch.ones(1)
        multi_hand_statistics["mean_cmap"] = torch.zeros(1)
        multi_hand_statistics["std_cmap"] = torch.ones(1)
    
    if "cmap_params" in statistics.get("object", {}):
        multi_hand_statistics["mean_obj_cmap"] = torch.from_numpy(statistics["object"]["cmap_params"]["mean"])
        multi_hand_statistics["std_obj_cmap"] = torch.from_numpy(statistics["object"]["cmap_params"]["std"])
    else:
        # Create dummy cmap stats for object
        multi_hand_statistics["mean_obj_cmap"] = torch.zeros(1)
        multi_hand_statistics["std_obj_cmap"] = torch.ones(1)
    
    return multi_hand_statistics
