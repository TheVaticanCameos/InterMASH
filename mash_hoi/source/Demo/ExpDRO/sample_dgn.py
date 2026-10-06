import torch
import pytorch_lightning as pl
from omegaconf import DictConfig
import hydra

from mash_hoi.source.Model.diffusion_collision import DiffusionCollision
# from mash_hoi.source.Model.diffusion_fixed import DiffusionFixedAnchor
from mash_hoi.source.Module.module_data import SphereDGNDataModule

torch.set_float32_matmul_precision('high')

@hydra.main(
    version_base="1.3", 
    config_path="../../Config/DRO", 
    config_name="ddpm_obb_loss_dro_dgn",
)
def main(cfg: DictConfig) -> None:
    pl.seed_everything(42)

    if cfg.dataset.data_name == "DexGraspNet":
        datamodule = SphereDGNDataModule(
            data_root=cfg.dataset.data_root,
            template_mash_path=cfg.dataset.template_mash_path,
            batch_size=cfg.dataset.batch_size,
            num_workers=cfg.dataset.num_workers,
        )
    else:
        raise ValueError(f"Unknown dataset name: {cfg.dataset.data_name}")

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



