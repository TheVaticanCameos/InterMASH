import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from omegaconf import DictConfig
import hydra

from mash_hoi.source.Model.diffusion_collision_multihand import MultiHandDiffusionCollision
from mash_hoi.source.Model.diffusion_collision_mul_cmap import MultiHandDiffusionCollision as MultiHandDiffusionCollisionHandCMap
from mash_hoi.source.Module.module_data import SphereDROTrainDataModule, SphereDROTrainDataModuleHandCMap

torch.set_float32_matmul_precision('high')


@hydra.main(
    version_base="1.3", 
    config_path="../../Config/DRO", 
    # config_name="ddpm_dga_loss_dro_handcmap",
    config_name="ddpm_dga_loss_dro",
)
def main(cfg: DictConfig) -> None:
    pl.seed_everything(42)
    
    if cfg.dataset.data_name == "CMapDataset":
        datamodule = SphereDROTrainDataModule(
            data_root=cfg.dataset.data_root,
            template_mash_root=cfg.dataset.template_mash_root,
            batch_size=cfg.dataset.batch_size,
            num_workers=cfg.dataset.num_workers,
        )
        model = MultiHandDiffusionCollision(
            diffusion_cfg=cfg.diffusion_cfg,
            model_cfg=cfg.model_cfg,
            mash_cfg=cfg.mash_cfg,
            loss_cfg=cfg.loss_cfg,
        )
    elif cfg.dataset.data_name == "HandCMap-CMapDataset":
        datamodule = SphereDROTrainDataModuleHandCMap(
            data_root=cfg.dataset.data_root,
            template_mash_root=cfg.dataset.template_mash_root,
            batch_size=cfg.dataset.batch_size,
            num_workers=cfg.dataset.num_workers,
        )
        model = MultiHandDiffusionCollisionHandCMap(
            diffusion_cfg=cfg.diffusion_cfg,
            model_cfg=cfg.model_cfg,
            mash_cfg=cfg.mash_cfg,
            loss_cfg=cfg.loss_cfg,
        )
    else:
        raise ValueError(f"Unknown dataset name: {cfg.dataset.data_name}")

    logger = TensorBoardLogger(
        cfg.trainer.log_dir,
        name=cfg.exp_name
    )

    trainer = pl.Trainer(
        logger=logger,
        fast_dev_run=cfg.trainer.fast_dev_run,
        max_epochs=cfg.trainer.max_epochs,
        check_val_every_n_epoch=cfg.trainer.check_val_every_n_epoch,
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision,
        num_sanity_val_steps=0,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        limit_test_batches=cfg.trainer.limit_test_batches,
    )

    trainer.fit(model, datamodule=datamodule)

    ckpt_path = os.path.join(logger.log_dir, "best_model.ckpt")
    trainer.save_checkpoint(ckpt_path)


if __name__ == "__main__":
    main()
