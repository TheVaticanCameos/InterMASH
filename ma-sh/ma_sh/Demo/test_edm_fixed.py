import pytorch_lightning as pl
import os
import sys
import hydra
from omegaconf import DictConfig
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Model.edm_fixed import EDMFixedAnchor
from Module.module_data import DGNDataModule

torch.set_float32_matmul_precision("high")


@hydra.main(
    version_base="1.3", 
    config_path="../Config", 
    config_name="edm_fix",
)
def main(cfg : DictConfig) -> None:
    pl.seed_everything(42)
    
    if cfg.dataset.data_name == "DexGraspNet":
        datamodule = DGNDataModule(
            data_root=cfg.dataset.root_dir,
            template_mash_path=cfg.dataset.template_mash_path,
            batch_size=cfg.dataset.batch_size,
            num_workers=cfg.dataset.num_workers,
        )
    else:
        raise ValueError(f"Unknown dataset name: {cfg.dataset.data_name}")

    model = EDMFixedAnchor(
        diffusion_cfg=cfg.diffusion_cfg,
        model_cfg=cfg.model_cfg,
        mash_cfg=cfg.mash_cfg
    )
    model.update_statistics(datamodule.statistics)

    # Save the trained model weights
    ckpt_path = os.path.join("/home/yxz/MyProjects/MASH-HOI/mash-hoi/runs/exp_edm_fixed_dgn/version_3/best_model.ckpt")

    tester = pl.Trainer(
        devices=cfg.tester.devices,
        precision=cfg.trainer.precision,
        num_sanity_val_steps=0,
        fast_dev_run=cfg.trainer.fast_dev_run,
        limit_test_batches=cfg.trainer.limit_test_batches
    )
    
    tester.test(model, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
