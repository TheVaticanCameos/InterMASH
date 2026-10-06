import os
import torch
import pytorch_lightning as pl
from typing import Tuple, Optional
import numpy as np
from addict import Dict
from tqdm import tqdm
from pathlib import Path
from lightning.pytorch.utilities import grad_norm
from Module.module_network import DiffusionTransformerV3Fixed
from Utils.muon import build_muon_param_groups, Muon


class DiffusionFixedAnchor(pl.LightningModule):
    def __init__(
        self,
        diffusion_cfg: Dict,
        model_cfg: Dict,
        mash_cfg: Dict
    ) -> None:
        super().__init__()
        assert model_cfg.d_in == model_cfg.d_out, "d_in must be equal to d_out"

        if model_cfg.name == "dit_v3":
            self.model = DiffusionTransformerV3Fixed(
                d_in=model_cfg.d_in,
                d_feat=model_cfg.d_feat,
                d_out=model_cfg.d_out,
                n_blocks=model_cfg.n_blocks
            )
        else:
            raise ValueError(f"Unknown model name: {model_cfg.name}")

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"{model_cfg.name} total parameters: {total_params}, trainable parameters: {trainable_params}")

        self.lr = diffusion_cfg.lr
        self.diffuse_steps = diffusion_cfg.diffuse_steps
        beta_start = diffusion_cfg.beta_start
        beta_end = diffusion_cfg.beta_end
        self.betas = torch.linspace(beta_start, beta_end, self.diffuse_steps, dtype=torch.float32)
        self.register_buffer("betas_buf", self.betas)
        self.alphas = 1.0 - self.betas
        self.register_buffer("alphas_buf", self.alphas)
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.register_buffer("alpha_bars_buf", self.alpha_bars)
        self.alpha_bars_prev = torch.cat([torch.ones(1, dtype=torch.float32), self.alpha_bars[:-1]])
        self.register_buffer("alpha_bars_prev_buf", self.alpha_bars_prev)

        self.output_dir = diffusion_cfg.output_dir

        self.mask_dim = mash_cfg.mask_dim
        self.sh_dim = mash_cfg.sh_dim
        self.ortho_dim = mash_cfg.ortho_dim

        self.hand_dim = self.mask_dim + self.sh_dim + self.ortho_dim
        self.object_dim = self.hand_dim

        self.statistics: Optional[Dict] = None

        if model_cfg.use_complie:
            self.model = torch.compile(self.model)

    def update_statistics(self, statistics: Dict):
        self.statistics = statistics

    def forward(
        self,
        noisy_feat: torch.Tensor,
        object_feat: torch.Tensor,
        pos: torch.Tensor,
        tem_hand: torch.Tensor,
        timesteps: torch.Tensor
    ) -> torch.Tensor:
        all_features = torch.cat([noisy_feat, object_feat, pos, tem_hand], dim=-1)
        pred_noise = self.model(all_features, timesteps)
        return pred_noise

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str]
    ) -> torch.Tensor:
        positions, hand_params, object_params, cmap_params, tem_hand, data_name = batch
        B = positions.shape[0]
        device = hand_params.device

        feat = torch.cat([hand_params, cmap_params], dim=-1)
        timesteps = torch.randint(0, self.diffuse_steps, (B,), device=device, dtype=torch.long)
        noise = torch.randn_like(feat)
        x_noisy = self.q_sample(feat, timesteps, noise)

        pred_noise = self.forward(
            noisy_feat=x_noisy,
            object_feat=object_params,
            pos=positions,
            tem_hand=tem_hand,
            timesteps=timesteps
        )
        loss = torch.nn.functional.mse_loss(pred_noise, noise)

        tr = getattr(self, "trainer", None)
        if tr is not None and hasattr(tr, "_results") and hasattr(tr, "optimizers"):
            self.log("train_loss", loss, prog_bar=True, on_step=True, sync_dist=False)
            self.log("lr", float(self.trainer.optimizers[0].param_groups[0]['lr']), prog_bar=False, on_step=True, sync_dist=False)
        return loss

    def on_before_optimizer_step(self, optimizer) -> None:
        if self.global_step % 1000 == 0:
            norms = grad_norm(self.model, norm_type=2)
            self.log_dict(norms)

    def validation_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str]
    ) -> torch.Tensor:
        positions, hand_params, object_params, cmap_params, tem_hand, data_name = batch
        B = positions.shape[0]
        device = hand_params.device
        feat = torch.cat([hand_params, cmap_params], dim=-1)
        x_t = torch.randn_like(feat)

        pbar = tqdm(range(self.diffuse_steps), desc="Validation")
        mse_step = None
        for t in reversed(range(self.diffuse_steps)):
            t_tensor = torch.full((B,), t, device=device, dtype=torch.long)
            pred_noise = self.forward(
                noisy_feat=x_t,
                object_feat=object_params,
                pos=positions,
                tem_hand=tem_hand,
                timesteps=t_tensor
            )
            x_t = self.p_sample_step(x_t, t_tensor, pred_noise)
            mse_step = torch.nn.functional.mse_loss(x_t, feat).item()
            pbar.set_description(f"Validation t={t}, mse_step={mse_step:.4f}")
            pbar.update(1)
        return mse_step

    def test_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]
    ) -> None:
        with torch.no_grad():
            positions, object_params, tem_hand, data_name_list = batch
            B, N, _ = positions.shape
            device = positions.device

            x_t = torch.randn(B, N, 22 + 4, device=device)

            pbar = tqdm(range(self.diffuse_steps), desc="Test")
            for t in reversed(range(self.diffuse_steps)):
                t_tensor = torch.full((B,), t, device=device, dtype=torch.long)
                pred_noise = self.forward(
                    noisy_feat=x_t,
                    object_feat=object_params,
                    pos=positions,
                    tem_hand=tem_hand,
                    timesteps=t_tensor
                )
                x_t = self.p_sample_step(x_t, t_tensor, pred_noise)
                pbar.update(1)

            os.makedirs(self.output_dir, exist_ok=True)
            for i in torch.randperm(B):
                data_name = Path(data_name_list[i]).stem
                hand_cmap_params_single = x_t[i]
                object_params_single = object_params[i]
                positions_single = positions[i]

                hand_params_single = hand_cmap_params_single[:, :self.hand_dim]
                cmap_params_single = hand_cmap_params_single[:, self.hand_dim:]

                hand_mask = hand_params_single[:, :self.mask_dim].detach().cpu().numpy()
                hand_sh = hand_params_single[:, self.mask_dim:self.mask_dim + self.sh_dim].detach().cpu().numpy()
                hand_ortho = hand_params_single[:, self.mask_dim + self.sh_dim:].detach().cpu().numpy()

                object_mask = object_params_single[:, :self.mask_dim].detach().cpu().numpy()
                object_sh = object_params_single[:, self.mask_dim:self.mask_dim + self.sh_dim].detach().cpu().numpy()
                object_ortho = object_params_single[:, self.mask_dim + self.sh_dim:].detach().cpu().numpy()

                cmap = cmap_params_single.detach().cpu().numpy()

                if self.statistics is None:
                    pos_out = positions_single.detach().cpu().numpy()
                    hand_mask_out = hand_mask
                    hand_sh_out = hand_sh
                    hand_ortho_out = hand_ortho
                    obj_mask_out = object_mask
                    obj_sh_out = object_sh
                    obj_ortho_out = object_ortho
                    cmap_out = cmap
                else:
                    pos_out = positions_single.detach().cpu().numpy() * self.statistics.positions.std + self.statistics.positions.mean
                    hand_mask_out = hand_mask * self.statistics.hand.mask_params.std + self.statistics.hand.mask_params.mean
                    hand_sh_out = hand_sh * self.statistics.hand.sh_params.std + self.statistics.hand.sh_params.mean
                    hand_ortho_out = hand_ortho * self.statistics.hand.ortho_poses.std + self.statistics.hand.ortho_poses.mean
                    obj_mask_out = object_mask * self.statistics.object.mask_params.std + self.statistics.object.mask_params.mean
                    obj_sh_out = object_sh * self.statistics.object.sh_params.std + self.statistics.object.sh_params.mean
                    obj_ortho_out = object_ortho * self.statistics.object.ortho_poses.std + self.statistics.object.ortho_poses.mean
                    cmap_out = cmap * self.statistics.cmap_params.sh_params.std + self.statistics.cmap_params.sh_params.mean

                generate_dict = {
                    "positions": pos_out,
                    "hand_params": {
                        "mask_params": hand_mask_out,
                        "sh_params": hand_sh_out,
                        "ortho_poses": hand_ortho_out
                    },
                    "object_params": {
                        "mask_params": obj_mask_out,
                        "sh_params": obj_sh_out,
                        "ortho_poses": obj_ortho_out
                    },
                    "cmap_params": {
                        "sh_params": cmap_out
                    }
                }
                np.save(os.path.join(self.output_dir, f"gen_{data_name}.npy"), generate_dict)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        param_groups = build_muon_param_groups(self.model)
        optimizer = Muon(
            param_groups,
            lr=self.lr,
            weight_decay=0.01,
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.lr,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.1,
            cycle_momentum=False
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        }

    def save_model(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load_model(self, path: str, map_location: Optional[str] = None):
        state = torch.load(path, map_location=map_location)
        self.model.load_state_dict(state)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        sqrt_ab = torch.sqrt(self.alpha_bars_buf[t]).view(-1, 1, 1)
        sqrt_one_minus_ab = torch.sqrt(1.0 - self.alpha_bars_buf[t]).view(-1, 1, 1)
        return sqrt_ab * x0 + sqrt_one_minus_ab * noise

    def p_sample_step(self, x_t: torch.Tensor, t: torch.Tensor, pred_noise: torch.Tensor) -> torch.Tensor:
        alpha_t = self.alphas_buf[t].view(-1, 1, 1)
        beta_t = self.betas_buf[t].view(-1, 1, 1)
        alpha_bar_t = self.alpha_bars_buf[t].view(-1, 1, 1)
        prev_alpha_bar = self.alpha_bars_prev_buf[t].view(-1, 1, 1)
        coef = beta_t / torch.sqrt(1.0 - alpha_bar_t)
        mean = (1.0 / torch.sqrt(alpha_t)) * (x_t - coef * pred_noise)
        posterior_var = beta_t * (1.0 - prev_alpha_bar) / (1.0 - alpha_bar_t)
        posterior_std = torch.sqrt(posterior_var)
        noise = torch.randn_like(x_t)
        nonzero_mask = (t > 0).float().view(-1, 1, 1).expand_as(x_t)
        posterior_std = posterior_std.expand_as(x_t)
        x_prev = mean + nonzero_mask * posterior_std * noise
        return x_prev
