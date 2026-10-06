import sys
import math
import time
import torch
from torch.utils.data import DataLoader, Subset

sys.path.append("../ma-sh")
from ma_sh.Module.data_prepare import SphereDRODataset
from ma_sh.Module.batch_object_trainer import BatchObjectTrainer

import argparse
from tqdm import tqdm


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--object_root", type=str, default="/data2/yxz/DATASET/DRO-data/data_urdf/object")
    parser.add_argument("--object_template_root", type=str, default="/data2/yxz/DATASET/DRO-data/DRO-Original/object_template")
    parser.add_argument("--split_path", type=str, default="/data2/yxz/DATASET/DRO-data/CMapDataset_filtered/split_train_validate_objects.json")

    parser.add_argument("--output_root", type=str, default="/data2/yxz/DATASET/HOI-Dataset/Sphere-CMapDataset/test")
    
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=4)

    args = parser.parse_args()

    sphere_dro_dataset = SphereDRODataset(
        object_root=args.object_root,
        object_template_root=args.object_template_root,
        split_path=args.split_path,
    )

    dataloader = DataLoader(
        sphere_dro_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    for object_points, object_name in tqdm(dataloader, desc="Processing Batches"):
        start_time = time.time()
        trainer = BatchObjectTrainer(
                anchor_num=256,
                mask_degree_max=3,
                sh_degree_max=0,
                sample_phi_num=32,
                sample_theta_num=32,
                points_per_submesh=1024,
                batch_size=args.batch_size,
                dtype=torch.float32,
                device="cuda",
                lr=0.01 * math.sqrt(args.batch_size),
                warmup_step_num=50,
                patience=2,
                save_log_folder_path=None,
                initial_method="sphere_fixed"
        )
        trainer.loadMeshFile(object_points)
        loss = trainer.escalateTrainMash()
        trainer.saveObjectMash(args.output_root, object_name)
        end_time = time.time()
        print(f"Total time: {end_time - start_time}")
        print(f"Average time: {(end_time - start_time) / args.batch_size}")
