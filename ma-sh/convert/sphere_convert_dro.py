import os
import sys
import math
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.append("../ma-sh")
from ma_sh.Module.data_prepare import DRODataset
from ma_sh.Module.batch_hoi_trainer import BatchHOITrainer

import argparse
from tqdm import tqdm
from pathlib import Path

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="/data2/yxz/DATASET/DRO-data/DRO-Original/barrett/val")
    parser.add_argument("--object_template_root", type=str, default="/data2/yxz/DATASET/DRO-data/DRO-Original/object_template")
    parser.add_argument("--hand_template_path", type=str, default="/data2/yxz/DATASET/template_info/sphere_barrett_template.npy")

    parser.add_argument("--hand_type", type=str, choices=["shadow", "allegro", "barrett"], default="barrett")

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=12)

    parser.add_argument("--output_root", type=str, default="/data2/yxz/DATASET/HOI-Dataset/CMapDataset/barrett/val")

    parser.add_argument("--part_num", type=int, default=1)
    parser.add_argument("--current_num", type=int, default=0)

    args = parser.parse_args()

    data_root = args.data_root
    object_template_root = args.object_template_root
    hand_template_path = args.hand_template_path
    
    hand_type = args.hand_type

    batch_size = args.batch_size
    num_workers = args.num_workers
    
    output_root = args.output_root

    part_num = args.part_num
    current_num = args.current_num

    assert Path(data_root).parent.stem == Path(hand_template_path).stem.split('_', 2)[1] == Path(output_root).parent.stem == hand_type

    os.makedirs(output_root, exist_ok=True)
    
    dataset = DRODataset(
        data_root=data_root,
        object_template_root=object_template_root,
        hand_template_path=hand_template_path,
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
        
        trainer.saveHOIMash(output_root, data_name)
        
        end_time = time.time()
        print(f"Total time: {end_time - start_time:.6f}")
        print(f"Average time: {(end_time - start_time) / len(data_name):.6f}")
    
    np.save(f"{output_root}/../bad_data_part_{current_num}.npy", bad_data)

