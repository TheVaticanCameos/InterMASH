import os
import torch
import pytorch_lightning as pl
import numpy as np
from tqdm import tqdm
from pathlib import Path
from lightning.pytorch.utilities import grad_norm
from mash_hoi.source.Module.module_network import DiffusionTransformerV4Graph
from mash_hoi.source.Utils.muon import build_muon_param_groups, Muon
from mash_hoi.source.Module.module_loss import MultiHandCollisionLossModule

from typing import Tuple, Optional, Dict


class MultiHandGraphDiffusionPE(pl.LightningModule):
    def __init__(
        self,
        diffusion_cfg: Dict,
        model_cfg: Dict,
        mash_cfg: Dict,
        loss_cfg: Dict,
        edge_vae_cfg: Optional[Dict] = None,
    ) -> None:
        super().__init__()

        self.latent_dim = model_cfg.get("latent_dim", 64)
        self.k_neighbors = model_cfg.get("k_neighbors", 8)
        self.pe_dim = model_cfg.get("pe_dim", 48)
        self.anchor_dim = model_cfg.d_out

        if model_cfg.name == "dit_v4_graph_pe":
            self.model = DiffusionTransformerV4Graph(
                d_in=model_cfg.d_in,
                d_feat=model_cfg.d_feat,
                d_out=model_cfg.d_out,
                n_blocks=model_cfg.n_blocks,
                pe_dim=self.pe_dim,
                latent_dim=self.latent_dim,
                knn_bias_scale=model_cfg.get("knn_bias_scale", 2.0),
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

        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)
        self.register_buffer("sqrt_alpha_bars_buf", self.sqrt_alpha_bars)
        self.register_buffer("sqrt_one_minus_alpha_bars_buf", self.sqrt_one_minus_alpha_bars)

        self.output_dir = diffusion_cfg.output_dir

        self.mask_dim = mash_cfg.mask_dim
        self.sh_dim = mash_cfg.sh_dim
        self.ortho_dim = mash_cfg.ortho_dim
        self.hand_dim = self.mask_dim + self.sh_dim + self.ortho_dim
        self.object_dim = self.hand_dim

        expected_anchor_dim = self.hand_dim + 4 + self.pe_dim
        if int(self.anchor_dim) != int(expected_anchor_dim):
            raise ValueError(f"model_cfg.d_out must be {expected_anchor_dim} (hand+cmap+pe), got {self.anchor_dim}")

        self.statistics: Optional[Dict] = None
        self.hand_type: Optional[str] = None

        self.lambda_collision = loss_cfg.lambda_collision
        self.lambda_latent = loss_cfg.get("lambda_latent", 1.0)
        self.guidance_scale = loss_cfg.guidance_scale
        self.guidance_threshold = loss_cfg.guidance_threshold
        self.use_spherical_gaussian = loss_cfg.use_spherical_gaussian
        self.collision_timestep_threshold = loss_cfg.get("collision_timestep_threshold", self.diffuse_steps // 2)

        self.collision_loss_module: Optional[MultiHandCollisionLossModule] = None
        if self.lambda_collision > 0.0:
            self.collision_loss_module = MultiHandCollisionLossModule(
                anchor_num=mash_cfg.get("anchor_num", 256),
                mask_dim=self.mask_dim,
                sh_dim=self.sh_dim,
                ortho_dim=self.ortho_dim,
                mask_degree_max=mash_cfg.get("mask_degree_max", 3),
                sh_degree_max=mash_cfg.get("sh_degree_max", 2),
                sample_phi_num=mash_cfg.get("sample_phi_num", 8),
                sample_theta_num=mash_cfg.get("sample_theta_num", 8),
                device="cuda" if torch.cuda.is_available() else "cpu",
                include_self_intersection=loss_cfg.get("include_self_intersection", True),
                use_multiprocessing=loss_cfg.get("use_multiprocessing", False),
                num_workers=loss_cfg.get("num_workers", 4),
                loss_type=loss_cfg.get("loss_type", "fcl"),
                erf_weight=loss_cfg.get("erf_weight", 1.0),
                cmap_loss_weight=loss_cfg.get("cmap_loss_weight", 1.0),
                srf_weight=loss_cfg.get("srf_weight", 1.0),
                srf_threshold=loss_cfg.get("srf_threshold", 0.02),
                srf_k_neighbors=loss_cfg.get("srf_k_neighbors", 10),
            )

        self.edge_vae = None
        self.use_edge_latent = False
        if edge_vae_cfg is not None and edge_vae_cfg.get("checkpoint_path") is not None:
            self._load_edge_vae(edge_vae_cfg)
            self.use_edge_latent = True
            print(f"Loaded Edge VAE from: {edge_vae_cfg['checkpoint_path']}")
        else:
            print("Edge VAE not configured, latent diffusion disabled")

        if model_cfg.get("use_compile", False):
            self.model = torch.compile(self.model)

    def _load_edge_vae(self, edge_vae_cfg: Dict):
        from mash_hoi.source.Model.vae_edge import EdgeVAE

        self.edge_vae = EdgeVAE(
            input_dim=edge_vae_cfg.get("input_dim", 22),
            n_anchors=edge_vae_cfg.get("n_anchors", 256),
            d_model=edge_vae_cfg.get("d_model", 128),
            nhead=edge_vae_cfg.get("nhead", 4),
            num_encoder_layers=edge_vae_cfg.get("num_encoder_layers", 4),
            latent_dim=edge_vae_cfg.get("latent_dim", 64),
            pos_dim=edge_vae_cfg.get("pos_dim", 64),
            phys_dim=edge_vae_cfg.get("phys_dim", 3),
            dropout=edge_vae_cfg.get("dropout", 0.1),
            row_hidden_dims=edge_vae_cfg.get("row_hidden_dims", [128, 64]),
            col_hidden_dims=edge_vae_cfg.get("col_hidden_dims", [256, 128, 64]),
        )

        checkpoint_path = edge_vae_cfg["checkpoint_path"]
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
        self.edge_vae.load_state_dict(state_dict, strict=False)

        self.edge_vae.eval()
        for param in self.edge_vae.parameters():
            param.requires_grad = False

    def update_statistics(self, statistics: Dict):
        self.statistics = statistics

    def update_hand_type(self, hand_type: str):
        self.hand_type = hand_type

    def forward(
        self,
        noisy_feat: torch.Tensor,
        object_feat: torch.Tensor,
        pos: torch.Tensor,
        tem_hand: torch.Tensor,
        timesteps: torch.Tensor,
        noisy_latent: Optional[torch.Tensor] = None,
        hand_knn_indices: Optional[torch.Tensor] = None,
        obj_knn_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        all_features = torch.cat([noisy_feat, object_feat, pos, tem_hand], dim=-1)
        pred_anchor_noise, pred_latent_noise = self.model(
            all_features,
            timesteps,
            noisy_latent=noisy_latent,
            hand_knn_indices=hand_knn_indices,
            obj_knn_indices=obj_knn_indices,
        )
        return pred_anchor_noise, pred_latent_noise

    def training_step(self, batch: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        positions, hand_params, object_params, cmap_params, anchor_pe, tem_hand, hand_knn, obj_knn, data_name, stats = batch
        B = positions.shape[0]
        device = hand_params.device

        if torch.isnan(hand_params).any() or torch.isinf(hand_params).any():
            print("[ERROR] Input hand_params has NaN/Inf! Replacing with 0")
            hand_params = torch.nan_to_num(hand_params, nan=0.0, posinf=1.0, neginf=-1.0)
        if torch.isnan(cmap_params).any() or torch.isinf(cmap_params).any():
            print("[ERROR] Input cmap_params has NaN/Inf! Replacing with 0")
            cmap_params = torch.nan_to_num(cmap_params, nan=0.0, posinf=1.0, neginf=-1.0)
        if torch.isnan(anchor_pe).any() or torch.isinf(anchor_pe).any():
            print("[ERROR] Input anchor_pe has NaN/Inf! Replacing with 0")
            anchor_pe = torch.nan_to_num(anchor_pe, nan=0.0, posinf=1.0, neginf=-1.0)
        if torch.isnan(object_params).any() or torch.isinf(object_params).any():
            print("[ERROR] Input object_params has NaN/Inf! Replacing with 0")
            object_params = torch.nan_to_num(object_params, nan=0.0, posinf=1.0, neginf=-1.0)

        anchor_feat = torch.cat([hand_params, cmap_params, anchor_pe], dim=-1)

        timesteps = torch.randint(0, self.diffuse_steps, (B,), device=device, dtype=torch.long)

        anchor_noise = torch.randn_like(anchor_feat)
        noisy_anchor = self.q_sample(anchor_feat, timesteps, anchor_noise)

        noisy_latent = None
        latent_noise = None
        gt_latent = None
        if self.use_edge_latent and self.edge_vae is not None:
            with torch.no_grad():
                gt_latent, _ = self.edge_vae.encode(hand_params, object_params, positions)

            latent_noise = torch.randn_like(gt_latent)
            noisy_latent = self.q_sample_1d(gt_latent, timesteps, latent_noise)

        pred_anchor_noise, pred_latent_noise = self.forward(
            noisy_feat=noisy_anchor,
            object_feat=object_params,
            pos=positions,
            tem_hand=tem_hand,
            timesteps=timesteps,
            noisy_latent=noisy_latent,
            hand_knn_indices=hand_knn,
            obj_knn_indices=obj_knn,
        )

        if torch.isnan(pred_anchor_noise).any() or torch.isinf(pred_anchor_noise).any():
            print("[ERROR] Model output pred_anchor_noise has NaN/Inf!")
            pred_anchor_noise = torch.nan_to_num(pred_anchor_noise, nan=0.0, posinf=10.0, neginf=-10.0)

        recon_loss = torch.nn.functional.mse_loss(pred_anchor_noise, anchor_noise)

        latent_loss = torch.tensor(0.0, device=device, dtype=hand_params.dtype)
        if self.use_edge_latent and latent_noise is not None:
            latent_loss = torch.nn.functional.mse_loss(pred_latent_noise, latent_noise)

        collision_loss = torch.tensor(0.0, device=device, dtype=hand_params.dtype)
        if self.lambda_collision > 0.0 and self.collision_loss_module is not None:
            valid_timesteps_mask = timesteps < self.collision_timestep_threshold

            if valid_timesteps_mask.any():
                sqrt_alpha_bar_t = self.sqrt_alpha_bars_buf[timesteps].view(B, 1, 1)
                sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bars_buf[timesteps].view(B, 1, 1)
                pred_x0 = (noisy_anchor - sqrt_one_minus_alpha_bar_t * pred_anchor_noise) / sqrt_alpha_bar_t

                if torch.isnan(pred_x0).any() or torch.isinf(pred_x0).any():
                    pred_x0 = torch.nan_to_num(pred_x0, nan=0.0, posinf=10.0, neginf=-10.0)

                pred_hand_params = pred_x0[:, :, :self.hand_dim]
                pred_cmap_params = pred_x0[:, :, self.hand_dim:self.hand_dim + 4]

                valid_timesteps = timesteps[valid_timesteps_mask].float()
                timestep_weights = torch.exp(-valid_timesteps / self.collision_timestep_threshold)
                timestep_weights = timestep_weights / timestep_weights.sum()

                collision_loss = self.collision_loss_module(
                    valid_timesteps_mask=valid_timesteps_mask,
                    hand_params=pred_hand_params,
                    object_params=object_params,
                    positions=positions,
                    denormalize=True,
                    object_cmap_params=pred_cmap_params,
                    sample_weights=timestep_weights,
                    stats=stats,
                )

        loss = recon_loss + self.lambda_latent * latent_loss + self.lambda_collision * collision_loss

        if torch.isnan(loss) or torch.isinf(loss):
            print(
                f"[ERROR] Final loss is NaN/Inf! recon={recon_loss.item()}, latent={latent_loss.item()}, collision={collision_loss.item()}"
            )
            loss = torch.tensor(0.0, device=device, dtype=hand_params.dtype, requires_grad=True)

        tr = getattr(self, "trainer", None)
        if tr is not None and hasattr(tr, "_results") and hasattr(tr, "optimizers"):
            self.log("recon_loss", recon_loss, prog_bar=True, on_step=True, sync_dist=False)
            self.log("latent_loss", latent_loss, prog_bar=True, on_step=True, sync_dist=False)
            self.log("collision_loss", collision_loss, prog_bar=True, on_step=True, sync_dist=False)
            self.log("lr", float(self.trainer.optimizers[0].param_groups[0]["lr"]), prog_bar=False, on_step=True, sync_dist=False)

        return loss

    def on_before_optimizer_step(self, optimizer) -> None:
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    param.grad.zero_()

        if self.global_step % 1000 == 0:
            norms = grad_norm(self.model, norm_type=2)
            self.log_dict(norms)

    def validation_step(self, batch: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        positions, hand_params, object_params, cmap_params, anchor_pe, tem_hand, hand_knn, obj_knn, data_name, stats = batch
        B = positions.shape[0]
        device = hand_params.device

        anchor_feat = torch.cat([hand_params, cmap_params, anchor_pe], dim=-1)
        x_t = torch.randn_like(anchor_feat)
        latent_t = torch.randn(B, self.latent_dim, device=device) if self.use_edge_latent else None

        pbar = tqdm(range(self.diffuse_steps), desc="Validation")
        mse_step = None
        for t in reversed(range(self.diffuse_steps)):
            t_tensor = torch.full((B,), t, device=device, dtype=torch.long)
            pred_anchor_noise, pred_latent_noise = self.forward(
                noisy_feat=x_t,
                object_feat=object_params,
                pos=positions,
                tem_hand=tem_hand,
                timesteps=t_tensor,
                noisy_latent=latent_t,
                hand_knn_indices=hand_knn,
                obj_knn_indices=obj_knn,
            )
            x_t = self.p_sample_step(x_t, t_tensor, pred_anchor_noise)
            if latent_t is not None:
                latent_t = self.p_sample_step_1d(latent_t, t_tensor, pred_latent_noise)
            mse_step = torch.nn.functional.mse_loss(x_t, anchor_feat).item()
            pbar.set_description(f"Validation t={t}, mse={mse_step:.4f}")
            pbar.update(1)
        return mse_step

    def test_step(self, batch: Tuple[torch.Tensor, ...]) -> None:
        with torch.no_grad():
            positions, object_params, tem_hand, hand_knn, obj_knn, data_name_list, stats = batch
            B, N, _ = positions.shape
            device = positions.device

            x_t = torch.randn(B, N, self.anchor_dim, device=device)
            latent_t = torch.randn(B, self.latent_dim, device=device) if self.use_edge_latent else None

            pbar = tqdm(range(self.diffuse_steps), desc="Test")
            for t in reversed(range(self.diffuse_steps)):
                t_tensor = torch.full((B,), t, device=device, dtype=torch.long)
                pred_anchor_noise, pred_latent_noise = self.forward(
                    noisy_feat=x_t,
                    object_feat=object_params,
                    pos=positions,
                    tem_hand=tem_hand,
                    timesteps=t_tensor,
                    noisy_latent=latent_t,
                    hand_knn_indices=hand_knn,
                    obj_knn_indices=obj_knn,
                )

                x_t = self.p_sample_step(
                    x_t,
                    t_tensor,
                    pred_anchor_noise,
                    guidance_grad=None,
                    guidance_scale=self.guidance_scale,
                    use_spherical_gaussian=self.use_spherical_gaussian,
                )
                if latent_t is not None:
                    latent_t = self.p_sample_step_1d(latent_t, t_tensor, pred_latent_noise)
                pbar.update(1)

            os.makedirs(self.output_dir, exist_ok=True)
            for i in torch.randperm(B):
                data_name = Path(data_name_list[i]).stem
                hand_cmap_pe_single = x_t[i]
                object_params_single = object_params[i]
                positions_single = positions[i]

                hand_params_single = hand_cmap_pe_single[:, :self.hand_dim]
                cmap_params_single = hand_cmap_pe_single[:, self.hand_dim:self.hand_dim + 4]
                anchor_pe_single = hand_cmap_pe_single[:, self.hand_dim + 4:]

                hand_mask = hand_params_single[:, :self.mask_dim].detach().cpu().numpy()
                hand_sh = hand_params_single[:, self.mask_dim:self.mask_dim + self.sh_dim].detach().cpu().numpy()
                hand_ortho = hand_params_single[:, self.mask_dim + self.sh_dim:self.mask_dim + self.sh_dim + self.ortho_dim].detach().cpu().numpy()

                object_mask = object_params_single[:, :self.mask_dim].detach().cpu().numpy()
                object_sh = object_params_single[:, self.mask_dim:self.mask_dim + self.sh_dim].detach().cpu().numpy()
                object_ortho = object_params_single[:, self.mask_dim + self.sh_dim:self.mask_dim + self.sh_dim + self.ortho_dim].detach().cpu().numpy()

                pos_out = positions_single.detach().cpu().numpy()
                cmap = cmap_params_single.detach().cpu().numpy()

                if self.statistics is not None:
                    pos_out = pos_out * self.statistics["std_positions"].cpu().numpy() + self.statistics["mean_positions"].cpu().numpy()
                    hand_mask_out = hand_mask * self.statistics["std_hand_mask"].cpu().numpy() + self.statistics["mean_hand_mask"].cpu().numpy()
                    hand_sh_out = hand_sh * self.statistics["std_hand_sh"].cpu().numpy() + self.statistics["mean_hand_sh"].cpu().numpy()
                    hand_ortho_out = hand_ortho * self.statistics["std_hand_ortho"].cpu().numpy() + self.statistics["mean_hand_ortho"].cpu().numpy()
                    obj_mask_out = object_mask * self.statistics["std_obj_mask"].cpu().numpy() + self.statistics["mean_obj_mask"].cpu().numpy()
                    obj_sh_out = object_sh * self.statistics["std_obj_sh"].cpu().numpy() + self.statistics["mean_obj_sh"].cpu().numpy()
                    obj_ortho_out = object_ortho * self.statistics["std_obj_ortho"].cpu().numpy() + self.statistics["mean_obj_ortho"].cpu().numpy()
                    cmap_out = cmap * self.statistics["std_hand_cmap"].cpu().numpy() + self.statistics["mean_hand_cmap"].cpu().numpy()
                else:
                    hand_mask_out = hand_mask
                    hand_sh_out = hand_sh
                    hand_ortho_out = hand_ortho
                    obj_mask_out = object_mask
                    obj_sh_out = object_sh
                    obj_ortho_out = object_ortho
                    cmap_out = cmap

                generate_dict = {
                    "positions": pos_out,
                    "hand_params": {
                        "mask_params": hand_mask_out,
                        "sh_params": hand_sh_out,
                        "ortho_poses": hand_ortho_out,
                        "cmap_params": cmap_out,
                    },
                    "object_params": {
                        "mask_params": obj_mask_out,
                        "sh_params": obj_sh_out,
                        "ortho_poses": obj_ortho_out,
                        "cmap_params": np.zeros_like(cmap_out),
                    },
                    "anchor_pe": anchor_pe_single.detach().cpu().numpy(),
                }

                if latent_t is not None:
                    generate_dict["iou_latent"] = latent_t[i].detach().cpu().numpy()

                filename_prefix = f"{self.hand_type}_" if self.hand_type is not None else ""
                np.save(os.path.join(self.output_dir, f"{filename_prefix}gen_{data_name}.npy"), generate_dict)

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
            cycle_momentum=False,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
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

    def q_sample_1d(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        sqrt_ab = torch.sqrt(self.alpha_bars_buf[t]).view(-1, 1)
        sqrt_one_minus_ab = torch.sqrt(1.0 - self.alpha_bars_buf[t]).view(-1, 1)
        return sqrt_ab * x0 + sqrt_one_minus_ab * noise

    def p_sample_step(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        pred_noise: torch.Tensor,
        guidance_grad: Optional[torch.Tensor] = None,
        guidance_scale: float = 1.0,
        use_spherical_gaussian: bool = False,
    ) -> torch.Tensor:
        alpha_t = self.alphas_buf[t].view(-1, 1, 1)
        beta_t = self.betas_buf[t].view(-1, 1, 1)
        alpha_bar_t = self.alpha_bars_buf[t].view(-1, 1, 1)
        prev_alpha_bar = self.alpha_bars_prev_buf[t].view(-1, 1, 1)
        coef = beta_t / torch.sqrt(1.0 - alpha_bar_t)
        mean = (1.0 / torch.sqrt(alpha_t)) * (x_t - coef * pred_noise)
        posterior_var = beta_t * (1.0 - prev_alpha_bar) / (1.0 - alpha_bar_t)

        if guidance_grad is not None:
            if use_spherical_gaussian:
                d_sample = -pred_noise
                grad_norm = torch.norm(guidance_grad, dim=-1, keepdim=True) + 1e-8
                d_star = -guidance_grad / grad_norm
                d_m = d_sample + guidance_scale * (d_star - d_sample)
                mean = (1.0 / torch.sqrt(alpha_t)) * (x_t + coef * d_m)
            else:
                mean = mean - guidance_scale * posterior_var * guidance_grad

        posterior_std = torch.sqrt(posterior_var)
        noise = torch.randn_like(x_t)
        nonzero_mask = (t > 0).float().view(-1, 1, 1).expand_as(x_t)
        posterior_std = posterior_std.expand_as(x_t)
        x_prev = mean + nonzero_mask * posterior_std * noise
        return x_prev

    def p_sample_step_1d(self, x_t: torch.Tensor, t: torch.Tensor, pred_noise: torch.Tensor) -> torch.Tensor:
        alpha_t = self.alphas_buf[t].view(-1, 1)
        beta_t = self.betas_buf[t].view(-1, 1)
        alpha_bar_t = self.alpha_bars_buf[t].view(-1, 1)
        prev_alpha_bar = self.alpha_bars_prev_buf[t].view(-1, 1)
        coef = beta_t / torch.sqrt(1.0 - alpha_bar_t)
        mean = (1.0 / torch.sqrt(alpha_t)) * (x_t - coef * pred_noise)
        posterior_var = beta_t * (1.0 - prev_alpha_bar) / (1.0 - alpha_bar_t)

        posterior_std = torch.sqrt(posterior_var)
        noise = torch.randn_like(x_t)
        nonzero_mask = (t > 0).float().view(-1, 1).expand_as(x_t)
        x_prev = mean + nonzero_mask * posterior_std * noise
        return x_prev
