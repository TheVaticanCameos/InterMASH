import sys
import math
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.append("../ma-sh")
from ma_sh.Module.data_prepare import DGNTrainDataset
from ma_sh.Module.batch_hoi_trainer import BatchHOITrainer

import argparse
from tqdm import tqdm


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="/mnt/d/yxz/DATASET/Original-Meshes/Sphere-DexGraspNet/train")
    parser.add_argument("--object_template_root", type=str, default="/mnt/d/yxz/DATASET/Original-Meshes/Sphere-DexGraspNet/object_template")
    parser.add_argument("--hand_template_path", type=str, default="/mnt/d/yxz/DATASET/template_info/sphere_shadow_template.npy")

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=12)

    parser.add_argument("--output_root", type=str, default="/mnt/d/yxz/DATASET/HOI-Dataset/Sphere-DexGraspNet/shadow/train")

    parser.add_argument("--part_num", type=int, default=1)
    parser.add_argument("--current_num", type=int, default=0)

    args = parser.parse_args()
    
    dataset = DGNTrainDataset(
        data_root=args.data_root,
        object_template_root=args.object_template_root,
        hand_template_path=args.hand_template_path,
        use_permutation=True,
    )
    part_len = len(dataset) // args.part_num
    subset = Subset(dataset, range(args.current_num * part_len, min((args.current_num + 1) * part_len, len(dataset))))
    dataloader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    bad_data = []
    for hand_points, object_points, cmap, data_name in tqdm(dataloader, desc="Processing batches"):
        start_time = time.time()
        trainer = BatchHOITrainer(
            anchor_num=256,
            mask_degree_max=3,
            sh_degree_max=0,
            sample_phi_num=32,
            sample_theta_num=32,
            points_per_submesh=1024,
            batch_size=len(data_name),
            dtype=torch.float32,
            device="cuda",
            lr=0.01 * math.sqrt(len(data_name)),
            warmup_step_num=50,
            patience=2,
            save_log_folder_path=None,
            initial_method="sphere_fixed"
        )
        trainer.loadMeshFile(hand_points, object_points, cmap)
        loss = trainer.escalateTrainMash()
        if loss > 0.0035:
            print(f"{data_name} loss: {loss:.6f}")
            bad_data.append(data_name)
        else:
            trainer.saveHOIMash(args.output_root, data_name)
        
        end_time = time.time()
        print(f"Total time: {end_time - start_time:.6f}")
        print(f"Average time: {(end_time - start_time) / len(data_name):.6f}")

    np.save(f"{args.output_root}/../bad_data_part_{args.current_num}.npy", bad_data)
