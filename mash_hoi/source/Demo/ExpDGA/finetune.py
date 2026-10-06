import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from omegaconf import DictConfig
import hydra

from mash_hoi.source.Model.diffusion_collision import DiffusionCollision
from mash_hoi.source.Module.module_data import SphereGRABDataModule

torch.set_float32_matmul_precision('high')


@hydra.main(
    version_base="1.3", 
    config_path="../../Config/DGA", 
    config_name="fine_tune_grab",
)
def main(cfg: DictConfig) -> None:
    pl.seed_everything(42)

    if cfg.dataset.data_name == "GRAB":
        datamodule = SphereGRABDataModule(
            data_root=cfg.dataset.data_root,
            template_mash_path=cfg.dataset.template_mash_path,
            batch_size=cfg.dataset.batch_size,
            num_workers=cfg.dataset.num_workers,
        )
    else:
        raise ValueError(f"Unknown dataset name: {cfg.dataset.data_name}")

    ckpt_path = cfg.finetune.ckpt_path
    model = DiffusionCollision.load_from_checkpoint(
        ckpt_path,
        diffusion_cfg=cfg.diffusion_cfg,
        model_cfg=cfg.model_cfg,
        mash_cfg=cfg.mash_cfg,
        loss_cfg=cfg.loss_cfg,
    )
    model.update_statistics(datamodule.statistics)

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

    save_name = cfg.finetune.get("save_name", "finetuned.ckpt")
    save_dir = cfg.finetune.get("save_dir", logger.log_dir)
    os.makedirs(save_dir, exist_ok=True)
    trainer.save_checkpoint(os.path.join(save_dir, save_name))


if __name__ == "__main__":
    main()
