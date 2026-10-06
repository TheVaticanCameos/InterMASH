"""
Sampling script for Graph-Aware Diffusion Model with IoU Token.

This script trains a diffusion model that uses:
1. K-neighbor graph attention for spatial relationships
2. IoU token for generating global IoU latent codes
"""

import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from omegaconf import DictConfig
import hydra

from mash_hoi.source.Model.graph_diffusion import GraphDiffusion
from mash_hoi.source.Module.module_data import GraphDGNDataModule

torch.set_float32_matmul_precision('high')


@hydra.main(
    version_base="1.3", 
    config_path="../../Config/DGA", 
    config_name="ddpm_graph_dgn_no_edgevae",
)
def main(cfg: DictConfig) -> None:
    pl.seed_everything(42)

    if cfg.dataset.data_name == "GraphDexGraspNetDataset":
        datamodule = GraphDGNDataModule(
            data_root=cfg.dataset.data_root,
            template_mash_path=cfg.dataset.template_mash_path,
            batch_size=cfg.dataset.batch_size,
            num_workers=cfg.dataset.num_workers,
            knn_cache_dir=cfg.dataset.get("knn_cache_dir", None),
            k_neighbors=cfg.dataset.get("k_neighbors", 8),
            precompute_device=cfg.dataset.get("precompute_device", "cuda"),
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

    # Use DDP strategy with find_unused_parameters=True to handle conditional parameter usage
    strategy = DDPStrategy(find_unused_parameters=True) if len(cfg.trainer.devices) > 1 else "auto"
    
    tester = pl.Trainer(
        logger=logger,
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

    ckpt_path = cfg.tester.ckpt_path
    model.update_statistics(datamodule.statistics)
    tester.test(model, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
