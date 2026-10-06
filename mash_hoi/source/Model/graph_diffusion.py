import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from lightning.pytorch.utilities import grad_norm
from tqdm import tqdm

from mash_hoi.source.Module.module_loss import CollisionLossModule
from mash_hoi.source.Module.module_network import DiffusionTransformerV3Graph
from mash_hoi.source.Utils.muon import Muon, build_muon_param_groups


def _stats_to_nested(stats: Dict[str, torch.Tensor]) -> Dict:
    if "positions" in stats:
        return stats

    def _cpu(x: torch.Tensor) -> torch.Tensor:
        return x.detach().to(device="cpu")

    out: Dict = {
        "positions": {"mean": _cpu(stats["mean_positions"]), "std": _cpu(stats["std_positions"])},
        "hand": {
            "mask_params": {"mean": _cpu(stats["mean_hand_mask"]), "std": _cpu(stats["std_hand_mask"])},
            "sh_params": {"mean": _cpu(stats["mean_hand_sh"]), "std": _cpu(stats["std_hand_sh"])},
            "ortho_poses": {"mean": _cpu(stats["mean_hand_ortho"]), "std": _cpu(stats["std_hand_ortho"])},
        },
        "object": {
            "mask_params": {"mean": _cpu(stats["mean_obj_mask"]), "std": _cpu(stats["std_obj_mask"])},
            "sh_params": {"mean": _cpu(stats["mean_obj_sh"]), "std": _cpu(stats["std_obj_sh"])},
            "ortho_poses": {"mean": _cpu(stats["mean_obj_ortho"]), "std": _cpu(stats["std_obj_ortho"])},
        },
    }

    if "mean_hand_cmap" in stats and "std_hand_cmap" in stats:
        out["hand"]["cmap_params"] = {"mean": _cpu(stats["mean_hand_cmap"]), "std": _cpu(stats["std_hand_cmap"])}
    if "mean_obj_cmap" in stats and "std_obj_cmap" in stats:
        out["object"]["cmap_params"] = {"mean": _cpu(stats["mean_obj_cmap"]), "std": _cpu(stats["std_obj_cmap"])}

    if "mean_cmap" in stats and "std_cmap" in stats:
        out["cmap_params"] = {
            "sh_params": {"mean": _cpu(stats["mean_cmap"]), "std": _cpu(stats["std_cmap"])}
        }
    elif "mean_obj_cmap" in stats and "std_obj_cmap" in stats:
        out["cmap_params"] = {
            "sh_params": {"mean": _cpu(stats["mean_obj_cmap"]), "std": _cpu(stats["std_obj_cmap"])}
        }

    return out


def _nested_to_numpy(d: Dict) -> Dict:
    def _convert(x):
        if isinstance(x, torch.Tensor):
            return x.detach().to(device="cpu").numpy()
        if isinstance(x, dict):
            return {k: _convert(v) for k, v in x.items()}
        return x

    return _convert(d)


class GraphDiffusion(pl.LightningModule):
    def __init__(
        self,
        diffusion_cfg: Dict,
        model_cfg: Dict,
        mash_cfg: Dict,
        loss_cfg: Dict,
        edge_vae_cfg: Optional[Dict] = None,
    ) -> None:
        super().__init__()
        assert model_cfg.d_in == model_cfg.d_out, "d_in must be equal to d_out"

        self.latent_dim = int(model_cfg.get("latent_dim", 64))
        self.k_neighbors = int(model_cfg.get("k_neighbors", 8))

        if model_cfg.name == "dit_v3_graph":
            self.model = DiffusionTransformerV3Graph(
                d_in=int(model_cfg.d_in),
                d_feat=int(model_cfg.d_feat),
                d_out=int(model_cfg.d_out),
                n_blocks=int(model_cfg.n_blocks),
                latent_dim=self.latent_dim,
                knn_bias_scale=float(model_cfg.get("knn_bias_scale", 2.0)),
            )
        else:
            raise ValueError(f"Unknown model name: {model_cfg.name}")

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"{model_cfg.name} total parameters: {total_params}, trainable parameters: {trainable_params}")

        self.lr = float(diffusion_cfg.lr)
        self.diffuse_steps = int(diffusion_cfg.diffuse_steps)
        beta_start = float(diffusion_cfg.beta_start)
        beta_end = float(diffusion_cfg.beta_end)

        betas = torch.linspace(beta_start, beta_end, self.diffuse_steps, dtype=torch.float32)
        self.register_buffer("betas_buf", betas)
        alphas = 1.0 - betas
        self.register_buffer("alphas_buf", alphas)
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer("alpha_bars_buf", alpha_bars)
        alpha_bars_prev = torch.cat([torch.ones(1, dtype=torch.float32), alpha_bars[:-1]])
        self.register_buffer("alpha_bars_prev_buf", alpha_bars_prev)

        self.register_buffer("sqrt_alpha_bars_buf", torch.sqrt(alpha_bars))
        self.register_buffer("sqrt_one_minus_alpha_bars_buf", torch.sqrt(1.0 - alpha_bars))

        self.output_dir = str(diffusion_cfg.output_dir)

        self.mask_dim = int(mash_cfg.mask_dim)
        self.sh_dim = int(mash_cfg.sh_dim)
        self.ortho_dim = int(mash_cfg.ortho_dim)
        self.hand_dim = self.mask_dim + self.sh_dim + self.ortho_dim
        self.cmap_dim = int(mash_cfg.get("cmap_dim", int(model_cfg.d_in) - self.hand_dim))

        self.statistics: Optional[Dict] = None

        self.lambda_collision = float(loss_cfg.lambda_collision)
        self.lambda_latent = float(loss_cfg.get("lambda_latent", 1.0))
        self.guidance_scale = float(loss_cfg.guidance_scale)
        self.guidance_threshold = float(loss_cfg.guidance_threshold)
        self.use_spherical_gaussian = bool(loss_cfg.use_spherical_gaussian)
        self.collision_timestep_threshold = int(loss_cfg.get("collision_timestep_threshold", self.diffuse_steps // 2))

        self.collision_loss_module: Optional[CollisionLossModule] = None
        if self.lambda_collision > 0.0:
            self.collision_loss_module = CollisionLossModule(
                anchor_num=int(mash_cfg.get("anchor_num", 256)),
                mask_dim=self.mask_dim,
                sh_dim=self.sh_dim,
                ortho_dim=self.ortho_dim,
                mask_degree_max=int(mash_cfg.get("mask_degree_max", 3)),
                sh_degree_max=int(mash_cfg.get("sh_degree_max", 2)),
                sample_phi_num=int(mash_cfg.get("sample_phi_num", 8)),
                sample_theta_num=int(mash_cfg.get("sample_theta_num", 8)),
                device="cuda" if torch.cuda.is_available() else "cpu",
                statistics=None,
                include_self_intersection=bool(loss_cfg.get("include_self_intersection", True)),
                use_multiprocessing=bool(loss_cfg.get("use_multiprocessing", False)),
                num_workers=int(loss_cfg.get("num_workers", 4)),
                loss_type=str(loss_cfg.get("loss_type", "fcl")),
                erf_weight=float(loss_cfg.get("erf_weight", 1.0)),
                cmap_loss_weight=float(loss_cfg.get("cmap_loss_weight", 1.0)),
                srf_weight=float(loss_cfg.get("srf_weight", 1.0)),
                srf_threshold=float(loss_cfg.get("srf_threshold", 0.02)),
                srf_k_neighbors=int(loss_cfg.get("srf_k_neighbors", 10)),
            )

        self.edge_vae = None
        self.use_edge_latent = False
        if edge_vae_cfg is not None and edge_vae_cfg.get("checkpoint_path") is not None:
            self._load_edge_vae(edge_vae_cfg)
            self.use_edge_latent = True
            print(f"Loaded Edge VAE from: {edge_vae_cfg['checkpoint_path']}")
        else:
            print("Edge VAE not configured, latent diffusion disabled")

        if bool(model_cfg.get("use_compile", False)):
            self.model = torch.compile(self.model)

    def _load_edge_vae(self, edge_vae_cfg: Dict):
        from mash_hoi.source.Model.vae_edge import EdgeVAE

        self.edge_vae = EdgeVAE(
            input_dim=int(edge_vae_cfg.get("input_dim", 22)),
            n_anchors=int(edge_vae_cfg.get("n_anchors", 256)),
            d_model=int(edge_vae_cfg.get("d_model", 128)),
            nhead=int(edge_vae_cfg.get("nhead", 4)),
            num_encoder_layers=int(edge_vae_cfg.get("num_encoder_layers", 4)),
            latent_dim=int(edge_vae_cfg.get("latent_dim", 64)),
            pos_dim=int(edge_vae_cfg.get("pos_dim", 64)),
            phys_dim=int(edge_vae_cfg.get("phys_dim", 3)),
            dropout=float(edge_vae_cfg.get("dropout", 0.1)),
            row_hidden_dims=edge_vae_cfg.get("row_hidden_dims", [128, 64]),
            col_hidden_dims=edge_vae_cfg.get("col_hidden_dims", [256, 128, 64]),
        )

        state_dict = torch.load(edge_vae_cfg["checkpoint_path"], map_location="cpu")
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
        self.edge_vae.load_state_dict(state_dict, strict=False)

        self.edge_vae.eval()
        for p in self.edge_vae.parameters():
            p.requires_grad = False

    def update_statistics(self, statistics: Dict):
        self.statistics = _nested_to_numpy(_stats_to_nested(statistics))
        if self.collision_loss_module is not None:
            self.collision_loss_module.statistics = self.statistics

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
        return self.model(
            all_features,
            timesteps,
            noisy_latent=noisy_latent,
            hand_knn_indices=hand_knn_indices,
            obj_knn_indices=obj_knn_indices,
        )

    def training_step(self, batch: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        positions, hand_params, object_params, cmap_params, tem_hand, hand_knn, obj_knn, _data_name, obj_points, obj_normals = batch

        B = positions.shape[0]
        device = positions.device

        if torch.isnan(hand_params).any() or torch.isinf(hand_params).any():
            hand_params = torch.nan_to_num(hand_params, nan=0.0, posinf=1.0, neginf=-1.0)
        if torch.isnan(object_params).any() or torch.isinf(object_params).any():
            object_params = torch.nan_to_num(object_params, nan=0.0, posinf=1.0, neginf=-1.0)
        if torch.isnan(cmap_params).any() or torch.isinf(cmap_params).any():
            cmap_params = torch.nan_to_num(cmap_params, nan=0.0, posinf=1.0, neginf=-1.0)

        anchor_feat = torch.cat([hand_params, cmap_params], dim=-1)
        timesteps = torch.randint(0, self.diffuse_steps, (B,), device=device, dtype=torch.long)
        anchor_noise = torch.randn_like(anchor_feat)
        noisy_anchor = self.q_sample(anchor_feat, timesteps, anchor_noise)

        noisy_latent = None
        latent_noise = None
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
            pred_anchor_noise = torch.nan_to_num(pred_anchor_noise, nan=0.0, posinf=10.0, neginf=-10.0)

        recon_loss = torch.nn.functional.mse_loss(pred_anchor_noise, anchor_noise)

        latent_loss = torch.tensor(0.0, device=device, dtype=hand_params.dtype)
        if self.use_edge_latent and latent_noise is not None:
            latent_loss = torch.nn.functional.mse_loss(pred_latent_noise, latent_noise)

        collision_loss = torch.tensor(0.0, device=device, dtype=hand_params.dtype)
        if self.lambda_collision > 0.0 and self.collision_loss_module is not None and self.statistics is not None:
            valid_mask = timesteps < self.collision_timestep_threshold
            if valid_mask.any():
                sqrt_alpha_bar_t = self.sqrt_alpha_bars_buf[timesteps].view(B, 1, 1)
                sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bars_buf[timesteps].view(B, 1, 1)
                pred_x0 = (noisy_anchor - sqrt_one_minus_alpha_bar_t * pred_anchor_noise) / sqrt_alpha_bar_t
                pred_x0 = torch.nan_to_num(pred_x0, nan=0.0, posinf=10.0, neginf=-10.0)

                pred_hand_params = pred_x0[:, :, : self.hand_dim]
                pred_cmap_params = pred_x0[:, :, self.hand_dim : self.hand_dim + self.cmap_dim]

                valid_timesteps = timesteps[valid_mask].float()
                timestep_weights = torch.exp(-valid_timesteps / self.collision_timestep_threshold)
                timestep_weights = timestep_weights / timestep_weights.sum()

                collision_loss = self.collision_loss_module(
                    hand_params=pred_hand_params[valid_mask],
                    object_params=object_params[valid_mask],
                    positions=positions[valid_mask],
                    denormalize=True,
                    cmap_params=pred_cmap_params[valid_mask],
                    sample_weights=timestep_weights,
                )

        loss = recon_loss + self.lambda_latent * latent_loss + self.lambda_collision * collision_loss
        if torch.isnan(loss) or torch.isinf(loss):
            loss = torch.tensor(0.0, device=device, dtype=hand_params.dtype, requires_grad=True)

        tr = getattr(self, "_trainer", None)
        if tr is not None and hasattr(tr, "_results") and hasattr(tr, "optimizers"):
            self.log("recon_loss", recon_loss, prog_bar=True, on_step=True, sync_dist=False)
            self.log("latent_loss", latent_loss, prog_bar=True, on_step=True, sync_dist=False)
            self.log("collision_loss", collision_loss, prog_bar=True, on_step=True, sync_dist=False)
            self.log("lr", float(tr.optimizers[0].param_groups[0]["lr"]), prog_bar=False, on_step=True, sync_dist=False)

        return loss

    def on_before_optimizer_step(self, optimizer) -> None:
        for _, param in self.model.named_parameters():
            if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                param.grad.zero_()

        if self.global_step % 1000 == 0:
            self.log_dict(grad_norm(self.model, norm_type=2))

    def validation_step(self, batch: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        positions, hand_params, object_params, cmap_params, tem_hand, hand_knn, obj_knn, _data_name, obj_points, obj_normals = batch

        B = positions.shape[0]
        device = positions.device

        anchor_feat = torch.cat([hand_params, cmap_params], dim=-1)
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
        return torch.tensor(mse_step, device=device, dtype=torch.float32)

    def test_step(self, batch: Tuple[torch.Tensor, ...]) -> None:
        with torch.no_grad():
            positions, object_params, tem_hand, hand_knn, obj_knn, data_name_list, obj_points, obj_normals = batch

            B, N, _ = positions.shape
            device = positions.device

            x_t = torch.randn(B, N, self.hand_dim + self.cmap_dim, device=device)
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

                guidance_grad = None
                if self.collision_loss_module is not None and self.statistics is not None and self.guidance_scale > 0:
                    with torch.enable_grad():
                        x_in = x_t.detach().requires_grad_(True)
                        sqrt_alpha_bar_t = self.sqrt_alpha_bars_buf[t_tensor].view(B, 1, 1)
                        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bars_buf[t_tensor].view(B, 1, 1)
                        x0_hat = (x_in - sqrt_one_minus_alpha_bar_t * pred_anchor_noise) / sqrt_alpha_bar_t

                        hand_params_hat = x0_hat[:, :, : self.hand_dim]
                        cmap_hat = x0_hat[:, :, self.hand_dim : self.hand_dim + self.cmap_dim]

                        loss = self.collision_loss_module(
                            hand_params=hand_params_hat,
                            object_params=object_params,
                            positions=positions,
                            denormalize=True,
                            cmap_params=cmap_hat,
                        )

                        loss_value = float(loss.detach().item())
                        if loss_value > self.guidance_threshold and loss.requires_grad:
                            grad = torch.autograd.grad(loss, x_in, retain_graph=False, create_graph=False, allow_unused=True)[0]
                            if grad is not None:
                                guidance_grad = grad

                x_t = self.p_sample_step(
                    x_t,
                    t_tensor,
                    pred_anchor_noise,
                    guidance_grad=guidance_grad,
                    guidance_scale=self.guidance_scale,
                    use_spherical_gaussian=self.use_spherical_gaussian,
                )
                if latent_t is not None:
                    latent_t = self.p_sample_step_1d(latent_t, t_tensor, pred_latent_noise)
                pbar.update(1)

            os.makedirs(self.output_dir, exist_ok=True)
            for i in torch.randperm(B):
                data_name = Path(str(data_name_list[i])).stem
                hand_cmap_single = x_t[i]
                object_params_single = object_params[i]
                positions_single = positions[i]

                hand_params_single = hand_cmap_single[:, : self.hand_dim]
                cmap_params_single = hand_cmap_single[:, self.hand_dim :]

                hand_mask = hand_params_single[:, : self.mask_dim].detach().cpu().numpy()
                hand_sh = hand_params_single[:, self.mask_dim : self.mask_dim + self.sh_dim].detach().cpu().numpy()
                hand_ortho = hand_params_single[:, self.mask_dim + self.sh_dim :].detach().cpu().numpy()

                obj_mask = object_params_single[:, : self.mask_dim].detach().cpu().numpy()
                obj_sh = object_params_single[:, self.mask_dim : self.mask_dim + self.sh_dim].detach().cpu().numpy()
                obj_ortho = object_params_single[:, self.mask_dim + self.sh_dim :].detach().cpu().numpy()

                cmap = cmap_params_single.detach().cpu().numpy()

                if self.statistics is None:
                    pos_out = positions_single.detach().cpu().numpy()
                    hand_mask_out = hand_mask
                    hand_sh_out = hand_sh
                    hand_ortho_out = hand_ortho
                    obj_mask_out = obj_mask
                    obj_sh_out = obj_sh
                    obj_ortho_out = obj_ortho
                    cmap_out = cmap
                else:
                    pos_out = (
                        positions_single.detach().cpu().numpy() * self.statistics["positions"]["std"]
                        + self.statistics["positions"]["mean"]
                    )
                    hand_mask_out = (
                        hand_mask * self.statistics["hand"]["mask_params"]["std"] + self.statistics["hand"]["mask_params"]["mean"]
                    )
                    hand_sh_out = (
                        hand_sh * self.statistics["hand"]["sh_params"]["std"] + self.statistics["hand"]["sh_params"]["mean"]
                    )
                    hand_ortho_out = (
                        hand_ortho * self.statistics["hand"]["ortho_poses"]["std"] + self.statistics["hand"]["ortho_poses"]["mean"]
                    )
                    obj_mask_out = (
                        obj_mask * self.statistics["object"]["mask_params"]["std"] + self.statistics["object"]["mask_params"]["mean"]
                    )
                    obj_sh_out = (
                        obj_sh * self.statistics["object"]["sh_params"]["std"] + self.statistics["object"]["sh_params"]["mean"]
                    )
                    obj_ortho_out = (
                        obj_ortho * self.statistics["object"]["ortho_poses"]["std"] + self.statistics["object"]["ortho_poses"]["mean"]
                    )
                    if "cmap_params" in self.statistics:
                        cmap_out = cmap * self.statistics["cmap_params"]["sh_params"]["std"] + self.statistics["cmap_params"]["sh_params"]["mean"]
                    else:
                        cmap_out = cmap

                generate_dict = {
                    "positions": pos_out,
                    "hand_params": {
                        "mask_params": hand_mask_out,
                        "sh_params": hand_sh_out,
                        "ortho_poses": hand_ortho_out,
                        "cmap_params": np.zeros_like(cmap_out),
                    },
                    "object_params": {
                        "mask_params": obj_mask_out,
                        "sh_params": obj_sh_out,
                        "ortho_poses": obj_ortho_out,
                        "cmap_params": cmap_out,
                    },
                }

                if latent_t is not None:
                    generate_dict["iou_latent"] = latent_t[i].detach().cpu().numpy()

                np.save(os.path.join(self.output_dir, f"gen_{data_name}.npy"), generate_dict)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        param_groups = build_muon_param_groups(self.model)
        optimizer = Muon(param_groups, lr=self.lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.lr,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.1,
            cycle_momentum=False,
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1}}

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
                grad_norm_val = torch.norm(guidance_grad, dim=-1, keepdim=True) + 1e-8
                d_star = -guidance_grad / grad_norm_val
                d_m = d_sample + guidance_scale * (d_star - d_sample)
                mean = (1.0 / torch.sqrt(alpha_t)) * (x_t + coef * d_m)
            else:
                mean = mean - guidance_scale * posterior_var * guidance_grad

        posterior_std = torch.sqrt(posterior_var)
        noise = torch.randn_like(x_t)
        nonzero_mask = (t > 0).float().view(-1, 1, 1).expand_as(x_t)
        return mean + nonzero_mask * posterior_std.expand_as(x_t) * noise

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
        return mean + nonzero_mask * posterior_std * noise
