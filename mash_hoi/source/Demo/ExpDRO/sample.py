import torch
import pytorch_lightning as pl
from omegaconf import DictConfig
import hydra

from mash_hoi.source.Model.diffusion_collision_multihand import MultiHandDiffusionCollision
from mash_hoi.source.Model.diffusion_collision_mul_cmap import MultiHandDiffusionCollision as MultiHandDiffusionCollisionHandCMap
from mash_hoi.source.Module.module_data import SphereDROTestDataModule, SphereDROTestDataModuleHandCMap

torch.set_float32_matmul_precision('high')


@hydra.main(
    version_base="1.3", 
    config_path="../../Config/DRO", 
    config_name="ddpm_dga_loss_dro_handcmap",
)
def main(cfg : DictConfig) -> None:
    pl.seed_everything(42)

    if cfg.dataset.data_name == "CMapDataset":
        model = MultiHandDiffusionCollision(
            diffusion_cfg=cfg.diffusion_cfg,
            model_cfg=cfg.model_cfg,
            mash_cfg=cfg.mash_cfg,
            loss_cfg=cfg.get('loss_cfg', None),
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
        
        shadow_datamodule = SphereDROTestDataModule(
            data_root=cfg.dataset.data_root,
            template_mash_root=cfg.dataset.template_mash_root,
            batch_size=cfg.dataset.batch_size,
            num_workers=cfg.dataset.num_workers,
            hand_type="shadow"
        )
        model.update_statistics(shadow_datamodule.statistics)
        model.update_hand_type("shadow")
        tester.test(model, datamodule=shadow_datamodule, ckpt_path=ckpt_path)

        allegro_datamodule = SphereDROTestDataModule(
            data_root=cfg.dataset.data_root,
            template_mash_root=cfg.dataset.template_mash_root,
            batch_size=cfg.dataset.batch_size,
            num_workers=cfg.dataset.num_workers,
            hand_type="allegro"
        )
        model.update_statistics(allegro_datamodule.statistics)
        model.update_hand_type("allegro")
        tester.test(model, datamodule=allegro_datamodule, ckpt_path=ckpt_path)

        barrett_datamodule = SphereDROTestDataModule(
            data_root=cfg.dataset.data_root,
            template_mash_root=cfg.dataset.template_mash_root,
            batch_size=cfg.dataset.batch_size,
            num_workers=cfg.dataset.num_workers,
            hand_type="barrett"
        )
        model.update_statistics(barrett_datamodule.statistics)
        model.update_hand_type("barrett")
        tester.test(model, datamodule=barrett_datamodule, ckpt_path=ckpt_path)

    elif cfg.dataset.data_name == "HandCMap-CMapDataset":
        model = MultiHandDiffusionCollisionHandCMap(
            diffusion_cfg=cfg.diffusion_cfg,
            model_cfg=cfg.model_cfg,
            mash_cfg=cfg.mash_cfg,
            loss_cfg=cfg.get('loss_cfg', None),
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

        shadow_datamodule = SphereDROTestDataModuleHandCMap(
            data_root=cfg.dataset.data_root,
            template_mash_root=cfg.dataset.template_mash_root,
            batch_size=cfg.dataset.batch_size,
            num_workers=cfg.dataset.num_workers,
            hand_type="shadow"
        )
        model.update_statistics(shadow_datamodule.statistics)
        model.update_hand_type("shadow")
        tester.test(model, datamodule=shadow_datamodule, ckpt_path=ckpt_path)

        allegro_datamodule = SphereDROTestDataModuleHandCMap(
            data_root=cfg.dataset.data_root,
            template_mash_root=cfg.dataset.template_mash_root,
            batch_size=cfg.dataset.batch_size,
            num_workers=cfg.dataset.num_workers,
            hand_type="allegro"
        )
        model.update_statistics(allegro_datamodule.statistics)
        model.update_hand_type("allegro")
        tester.test(model, datamodule=allegro_datamodule, ckpt_path=ckpt_path)

        barrett_datamodule = SphereDROTestDataModuleHandCMap(
            data_root=cfg.dataset.data_root,
            template_mash_root=cfg.dataset.template_mash_root,
            batch_size=cfg.dataset.batch_size,
            num_workers=cfg.dataset.num_workers,
            hand_type="barrett"
        )
        model.update_statistics(barrett_datamodule.statistics)
        model.update_hand_type("barrett")
        tester.test(model, datamodule=barrett_datamodule, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
