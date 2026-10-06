import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from omegaconf import DictConfig
import hydra
import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from mash_hoi.source.Model.graph_diffusion_multihand_pe import MultiHandGraphDiffusionPE
from mash_hoi.source.Module.DataModule.data_dro_pe import PEGraphDROTrainDataModule

torch.set_float32_matmul_precision('high')


@hydra.main(
    version_base="1.3",
    config_path="../../Config/DRO",
    config_name="ddpm_graph_dro_pe",
)
def main(cfg: DictConfig) -> None:
    pl.seed_everything(42)

    max_freq = int(cfg.dataset.get("max_freq", 8))
    pe_dim = 3 * max_freq * 2
    expected_anchor_dim = 22 + 4 + pe_dim
    expected_model_in_dim = expected_anchor_dim + 22 + 3 + 22
    cfg.model_cfg.pe_dim = pe_dim
    cfg.model_cfg.d_in = expected_model_in_dim
    cfg.model_cfg.d_out = expected_anchor_dim

    if cfg.dataset.data_name == "GraphPEDataset":
        datamodule = PEGraphDROTrainDataModule(
            data_root=cfg.dataset.data_root,
            template_mash_root=cfg.dataset.template_mash_root,
            batch_size=cfg.dataset.batch_size,
            num_workers=cfg.dataset.num_workers,
            knn_cache_dir=cfg.dataset.get("knn_cache_dir", None),
            k_neighbors=cfg.dataset.get("k_neighbors", 8),
            max_freq=max_freq,
            precompute_device=cfg.dataset.get("precompute_device", "cuda"),
        )
    else:
        raise ValueError(f"Unknown dataset name: {cfg.dataset.data_name}")

    edge_vae_cfg = cfg.get("edge_vae_cfg", None)
    if edge_vae_cfg is not None:
        edge_vae_cfg = dict(edge_vae_cfg)

    model = MultiHandGraphDiffusionPE(
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

    strategy = DDPStrategy(find_unused_parameters=True) if len(cfg.trainer.devices) > 1 else "auto"

    trainer = pl.Trainer(
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

    trainer.fit(model, datamodule=datamodule)

    ckpt_path = os.path.join(logger.log_dir, "best_model.ckpt")
    trainer.save_checkpoint(ckpt_path)


if __name__ == "__main__":
    main()
