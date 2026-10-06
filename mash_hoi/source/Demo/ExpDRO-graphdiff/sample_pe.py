"""
Sample script for Graph-Aware Diffusion Model with Anchor P.E. generation.
"""

import torch
import pytorch_lightning as pl
from omegaconf import DictConfig
import hydra
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from mash_hoi.source.Model.graph_diffusion_multihand_pe import MultiHandGraphDiffusionPE
from mash_hoi.source.Module.module_data import GraphDROTestDataModule

torch.set_float32_matmul_precision("high")


@hydra.main(
    version_base="1.3",
    config_path="../../Config/DRO",
    config_name="ddpm_graph_dro_pe",
)
def main(cfg: DictConfig) -> None:
    pl.seed_everything(42)

    if cfg.dataset.data_name != "GraphPEDataset":
        raise ValueError(f"Unknown dataset name: {cfg.dataset.data_name}")

    max_freq = int(cfg.dataset.get("max_freq", 8))
    pe_dim = 3 * max_freq * 2
    expected_anchor_dim = 22 + 4 + pe_dim
    expected_model_in_dim = expected_anchor_dim + 22 + 3 + 22
    cfg.model_cfg.pe_dim = pe_dim
    cfg.model_cfg.d_in = expected_model_in_dim
    cfg.model_cfg.d_out = expected_anchor_dim

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

    tester = pl.Trainer(
        accelerator="gpu",
        devices=cfg.tester.devices,
        precision=cfg.trainer.precision,
        num_sanity_val_steps=0,
        fast_dev_run=cfg.trainer.fast_dev_run,
        limit_test_batches=cfg.trainer.limit_test_batches,
    )

    ckpt_path = cfg.tester.ckpt_path

    shadow_datamodule = GraphDROTestDataModule(
        data_root=cfg.dataset.data_root,
        template_mash_root=cfg.dataset.template_mash_root,
        batch_size=cfg.dataset.batch_size,
        num_workers=cfg.dataset.num_workers,
        knn_cache_dir=cfg.dataset.get("knn_cache_dir", None),
        k_neighbors=cfg.dataset.get("k_neighbors", 8),
        precompute_device=cfg.dataset.get("precompute_device", "cuda"),
        hand_type="shadow",
    )
    model.update_statistics(shadow_datamodule.statistics)
    model.update_hand_type("shadow")
    tester.test(model, datamodule=shadow_datamodule, ckpt_path=ckpt_path)

    allegro_datamodule = GraphDROTestDataModule(
        data_root=cfg.dataset.data_root,
        template_mash_root=cfg.dataset.template_mash_root,
        batch_size=cfg.dataset.batch_size,
        num_workers=cfg.dataset.num_workers,
        knn_cache_dir=cfg.dataset.get("knn_cache_dir", None),
        k_neighbors=cfg.dataset.get("k_neighbors", 8),
        precompute_device=cfg.dataset.get("precompute_device", "cuda"),
        hand_type="allegro",
    )
    model.update_statistics(allegro_datamodule.statistics)
    model.update_hand_type("allegro")
    tester.test(model, datamodule=allegro_datamodule, ckpt_path=ckpt_path)

    barrett_datamodule = GraphDROTestDataModule(
        data_root=cfg.dataset.data_root,
        template_mash_root=cfg.dataset.template_mash_root,
        batch_size=cfg.dataset.batch_size,
        num_workers=cfg.dataset.num_workers,
        knn_cache_dir=cfg.dataset.get("knn_cache_dir", None),
        k_neighbors=cfg.dataset.get("k_neighbors", 8),
        precompute_device=cfg.dataset.get("precompute_device", "cuda"),
        hand_type="barrett",
    )
    model.update_statistics(barrett_datamodule.statistics)
    model.update_hand_type("barrett")
    tester.test(model, datamodule=barrett_datamodule, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
