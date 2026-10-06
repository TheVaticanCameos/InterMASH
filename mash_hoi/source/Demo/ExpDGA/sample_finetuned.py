import torch
import pytorch_lightning as pl
from omegaconf import DictConfig
import hydra

from mash_hoi.source.Model.diffusion_collision import DiffusionCollision
from mash_hoi.source.Module.module_data import SphereGRABDataModule, SphereDGNDataModule

torch.set_float32_matmul_precision('high')


@hydra.main(
    version_base="1.3", 
    config_path="../../Config/DGA", 
    config_name="fine_tune_grab",
)
def main(cfg: DictConfig) -> None:
    pl.seed_everything(42)

    if cfg.tester.dataset.data_name == "GRAB":
        datamodule = SphereGRABDataModule(
            data_root=cfg.tester.dataset.data_root,
            template_mash_path=cfg.tester.dataset.template_mash_path,
            batch_size=cfg.tester.dataset.batch_size,
            num_workers=cfg.tester.dataset.num_workers,
        )
    elif cfg.tester.dataset.data_name == "DexGraspNet":
        datamodule = SphereDGNDataModule(
            data_root=cfg.tester.dataset.data_root,
            template_mash_path=cfg.tester.dataset.template_mash_path,
            batch_size=cfg.tester.dataset.batch_size,
            num_workers=cfg.tester.dataset.num_workers,
        )
    else:
        raise ValueError(f"Unknown dataset name: {cfg.tester.dataset.data_name}")
    
    model = DiffusionCollision(
        diffusion_cfg=cfg.diffusion_cfg,
        model_cfg=cfg.model_cfg,
        mash_cfg=cfg.mash_cfg,
        loss_cfg=cfg.loss_cfg,
    )

    tester = pl.Trainer(
        accelerator="gpu",
        devices=cfg.tester.devices,
        precision=cfg.trainer.precision,
        num_sanity_val_steps=0,
        fast_dev_run=cfg.trainer.fast_dev_run,
        limit_test_batches=cfg.trainer.limit_test_batches
    )

    ckpt_path = cfg.tester.ckpt_path
    model.update_statistics(datamodule.statistics)
    tester.test(model, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
