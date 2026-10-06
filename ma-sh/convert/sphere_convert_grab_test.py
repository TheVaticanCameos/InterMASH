import math
import time
import torch
from torch.utils.data import DataLoader

from ma_sh.Module.data_prepare import SphereGRABTestDataset
from ma_sh.Module.batch_object_trainer import BatchObjectTrainer

import argparse
from tqdm import tqdm

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="/data2/yxz/DATASET/grab_data/obj_meshes")
    parser.add_argument("--test_list", type=str, default="/data2/yxz/DATASET/grab_data/test_list.txt")
    parser.add_argument("--object_template_root", type=str, default="/data2/yxz/DATASET/GRAB-Original/object_templates")
    
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=12)

    parser.add_argument("--output_root", type=str, default="/data2/yxz/DATASET/HOI-Dataset/HandCMap-Sphere-GRAB/test")

    args = parser.parse_args()

    dataset = SphereGRABTestDataset(
        data_root=args.data_root,
        test_object_names=args.test_list,
        object_template_root=args.object_template_root,   
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    for object_points, data_name in tqdm(dataloader, desc="Converting GRAB Test Data"):
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
            initial_method="sphere_fixed"
        )
        trainer.loadMeshFile(object_points)
        loss = trainer.escalateTrainMash()
        trainer.saveObjectMash(args.output_root, data_name)

        end_time = time.time()
        print(f"Loss: {loss:.6f}")
        print(f"Total time: {end_time - start_time:.6f}")
        print(f"Average time: {(end_time - start_time) / len(data_name):.6f}")
