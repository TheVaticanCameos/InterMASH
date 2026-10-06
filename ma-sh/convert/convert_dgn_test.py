import sys
import math
import time
import torch
from torch.utils.data import DataLoader, Subset

sys.path.append("../ma-sh")
from ma_sh.Module.data_prepare import DGNTestDataset
from ma_sh.Module.batch_object_trainer import BatchObjectTrainer

import argparse
from tqdm import tqdm

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="/mnt/d/yxz/DATASET/Original-Meshes/DexGraspNet/test")
    parser.add_argument("--object_template_root", type=str, default="/mnt/d/yxz/DATASET/Original-Meshes/DexGraspNet/object_template")
    
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=12)

    parser.add_argument("--output_root", type=str, default="/mnt/d/yxz/DATASET/HOI-Dataset/DexGraspNet/shadow/test")

    parser.add_argument("--part_num", type=int, default=1)
    parser.add_argument("--current_num", type=int, default=0)

    args = parser.parse_args()

    dataset = DGNTestDataset(
        data_root=args.data_root,
        object_template_root=args.object_template_root,
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

    for object_points, data_name in tqdm(dataloader, desc="Processing batches"):
        start_time = time.time()
        trainer = BatchObjectTrainer(
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
            initial_method="middle_plane_fixed"
        )
        trainer.loadMeshFile(object_points)
        loss = trainer.escalateTrainMash()
        trainer.saveObjectMash(args.output_root, data_name)
        
        end_time = time.time()
        print(f"Loss: {loss:.6f}")
        print(f"Total time: {end_time - start_time:.6f}")
        print(f"Average time: {(end_time - start_time) / len(data_name):.6f}")
