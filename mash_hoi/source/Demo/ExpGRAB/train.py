import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from omegaconf import DictConfig
import hydra

import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from mash_hoi.source.Model.graph_diffusion import GraphDiffusion
from mash_hoi.source.Module.module_data import GraphGRABDataModule

torch.set_float32_matmul_precision('high')


@hydra.main(
    version_base="1.3", 
    config_path="../../Config/MANO", 
    config_name="ddpm_grab",
)
def main(cfg: DictConfig) -> None:
    pl.seed_everything(42)

    if cfg.dataset.data_name == "GRAB":
        datamodule = GraphGRABDataModule(
            data_root=cfg.dataset.data_root,
            template_mash_path=cfg.dataset.template_mash_path,
            batch_size=cfg.dataset.batch_size,
            num_workers=cfg.dataset.num_workers,
        )
    else:
        raise ValueError(f"Unknown dataset name: {cfg.dataset.data_name}")

    # Build model with optional Edge Weight VAE config
    edge_vae_cfg = cfg.get("edge_vae_cfg", None)
    if edge_vae_cfg is not None:
        edge_vae_cfg = dict(edge_vae_cfg)  # Convert OmegaConf to dict
    
    model = GraphDiffusion(
        diffusion_cfg=cfg.diffusion_cfg,
        model_cfg=cfg.model_cfg,
        mash_cfg=cfg.mash_cfg,
        loss_cfg=cfg.loss_cfg,
        edge_vae_cfg=edge_vae_cfg,
    )

    logger = TensorBoardLogger(
        cfg.trainer.log_dir,
        name=cfg.exp_name
    )

    # Use DDP strategy with find_unused_parameters=True to handle conditional parameter usage
    strategy = DDPStrategy(find_unused_parameters=True) if len(cfg.trainer.devices) > 1 else "auto"

    checkpoint_callback = ModelCheckpoint(
        dirpath=logger.log_dir,
        filename="epoch{epoch:04d}",
        save_top_k=-1,
        every_n_epochs=100,
    )
    
    trainer = pl.Trainer(
        logger=logger,
        callbacks=[checkpoint_callback],
        fast_dev_run=cfg.trainer.fast_dev_run,
        max_epochs=cfg.trainer.max_epochs,
        check_val_every_n_epoch=cfg.trainer.check_val_every_n_epoch,
        devices=cfg.trainer.devices,
        strategy=strategy,
        precision=cfg.trainer.precision,
        num_sanity_val_steps=0,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        limit_test_batches=cfg.trainer.limit_test_batches,
    )
    model.update_statistics(datamodule.statistics)

    trainer.fit(model, datamodule=datamodule)

    ckpt_path = os.path.join(logger.log_dir, "best_model.ckpt")
    trainer.save_checkpoint(ckpt_path)


if __name__ == "__main__":
    main()
