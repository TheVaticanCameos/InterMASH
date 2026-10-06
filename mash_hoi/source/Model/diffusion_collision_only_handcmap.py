import os
import torch
import pytorch_lightning as pl
import numpy as np
from tqdm import tqdm
from pathlib import Path
from lightning.pytorch.utilities import grad_norm
from mash_hoi.source.Module.module_network import DiffusionTransformerV3Fixed
from mash_hoi.source.Utils.muon import build_muon_param_groups, Muon
from mash_hoi.source.Module.module_loss import MultiHandCollisionLossModuleOnlyHand

from typing import Tuple, Optional, Dict


class MultiHandDiffusionCollision(pl.LightningModule):
    def __init__(
        self,
        diffusion_cfg: Dict,
        model_cfg: Dict,
        mash_cfg: Dict,
        loss_cfg: Dict,
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
        
        # Precompute sqrt coefficients for efficiency
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)
        self.register_buffer("sqrt_alpha_bars_buf", self.sqrt_alpha_bars)
        self.register_buffer("sqrt_one_minus_alpha_bars_buf", self.sqrt_one_minus_alpha_bars)

        self.output_dir = diffusion_cfg.output_dir

        self.mask_dim = mash_cfg.mask_dim
        self.sh_dim = mash_cfg.sh_dim
        self.ortho_dim = mash_cfg.ortho_dim
        self.cmap_dim = mash_cfg.cmap_dim

        self.hand_dim = self.mask_dim + self.sh_dim + self.ortho_dim
        self.object_dim = self.hand_dim

        self.statistics: Optional[Dict] = None
        
        # Collision loss module
        # Get loss configuration        
        self.lambda_collision = loss_cfg.lambda_collision
        # Physics-Guided Sampling parameters
        self.guidance_scale = loss_cfg.guidance_scale
        self.guidance_threshold = loss_cfg.guidance_threshold
        # Optimization: Spherical Gaussian Constraint (PhysGuidance.md Section 4)
        self.use_spherical_gaussian = loss_cfg.use_spherical_gaussian
        
        self.collision_timestep_threshold = loss_cfg.get('collision_timestep_threshold', self.diffuse_steps // 2)
        self.collision_loss_module: Optional[MultiHandCollisionLossModuleOnlyHand] = None
        if self.lambda_collision > 0.0:
            
            self.collision_loss_module = MultiHandCollisionLossModuleOnlyHand(
                anchor_num=mash_cfg.get('anchor_num', 256),
                mask_dim=self.mask_dim,
                sh_dim=self.sh_dim,
                ortho_dim=self.ortho_dim,
                mask_degree_max=mash_cfg.get('mask_degree_max', 3),
                sh_degree_max=mash_cfg.get('sh_degree_max', 2),
                sample_phi_num=mash_cfg.get('sample_phi_num', 8),
                sample_theta_num=mash_cfg.get('sample_theta_num', 8),
                device="cuda" if torch.cuda.is_available() else "cpu",
                include_self_intersection=loss_cfg.get('include_self_intersection', True),
                use_multiprocessing=loss_cfg.get('use_multiprocessing', False),
                num_workers=loss_cfg.get('num_workers', 4),
                loss_type=loss_cfg.get('loss_type', 'fcl'),
                erf_weight=loss_cfg.get('erf_weight', 1.0),
                cmap_loss_weight=loss_cfg.get('cmap_loss_weight', 1.0),
                srf_weight=loss_cfg.get('srf_weight', 1.0),
                srf_threshold=loss_cfg.get('srf_threshold', 0.02),
                srf_k_neighbors=loss_cfg.get('srf_k_neighbors', 10),
            )

        if model_cfg.use_complie:
            self.model = torch.compile(self.model)
        
        self.hand_type: str | None = None

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
        timesteps: torch.Tensor
    ) -> torch.Tensor:
        all_features = torch.cat([noisy_feat, object_feat, pos, tem_hand], dim=-1)
        pred_noise = self.model(all_features, timesteps)
        return pred_noise

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str, Dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        positions, hand_params, object_params, tem_hand, data_name, stats = batch
        B = positions.shape[0]
        device = hand_params.device
        
        # Input data sanity check
        if torch.isnan(hand_params).any() or torch.isinf(hand_params).any():
            print("[ERROR] Input hand_params has NaN/Inf! Replacing with 0")
            hand_params = torch.nan_to_num(hand_params, nan=0.0, posinf=1.0, neginf=-1.0)
        if torch.isnan(object_params).any() or torch.isinf(object_params).any():
            print("[ERROR] Input object_params has NaN/Inf! Replacing with 0")
            object_params = torch.nan_to_num(object_params, nan=0.0, posinf=1.0, neginf=-1.0)

        hand_geom = hand_params[:, :, :self.hand_dim]
        hand_cmap = hand_params[:, :, self.hand_dim:]
        object_geom = object_params[:, :, :self.object_dim]
        object_cmap = object_params[:, :, self.object_dim:]

        feat = torch.cat([hand_geom, hand_cmap], dim=-1)   # MASK_DIM+SH_DIM+ORTHO_DIM+CMAP_DIM
        timesteps = torch.randint(0, self.diffuse_steps, (B,), device=device, dtype=torch.long)
        noise = torch.randn_like(feat)
        x_noisy = self.q_sample(feat, timesteps, noise)

        # Check inputs before forward pass
        if torch.isnan(x_noisy).any() or torch.isinf(x_noisy).any():
            print("[ERROR] x_noisy has NaN/Inf before model forward!")
            print(f"  x_noisy stats: min={x_noisy.min().item()}, max={x_noisy.max().item()}")
        if torch.isnan(object_params).any() or torch.isinf(object_params).any():
            print("[ERROR] object_params has NaN/Inf before model forward!")
        
        pred_noise = self.forward(
            noisy_feat=x_noisy,
            object_feat=object_geom,
            pos=positions,
            tem_hand=tem_hand,
            timesteps=timesteps
        )
        
        # Safety check: Replace NaN/Inf in model output before computing loss
        if torch.isnan(pred_noise).any() or torch.isinf(pred_noise).any():
            print("[ERROR] Model output pred_noise has NaN/Inf!")
            print(f"  global_step: {self.global_step}, current_epoch: {self.current_epoch}")
            print(f"  pred_noise stats: min={torch.nanmin(pred_noise).item() if not torch.isnan(pred_noise).all() else 'all NaN'}, "
                  f"max={torch.nanmax(pred_noise).item() if not torch.isnan(pred_noise).all() else 'all NaN'}")
            print(f"  NaN count: {torch.isnan(pred_noise).sum().item()}/{pred_noise.numel()}")
            print(f"  Inf count: {torch.isinf(pred_noise).sum().item()}/{pred_noise.numel()}")
            print("  Clamping to [-10, 10]...")
            pred_noise = torch.nan_to_num(pred_noise, nan=0.0, posinf=10.0, neginf=-10.0)
        
        recon_loss = torch.nn.functional.mse_loss(pred_noise, noise)

        # ==========================================================================================
        # Physics-Aware Training (PhysGuidance.md Section 3)
        # Core Idea: Compute physical constraints on the estimated clean sample (hat_x0) 
        # to bridge the gap between noisy training data and physical rules.
        # ==========================================================================================
        # Collision loss (constraint-aware loss similar to DDPM)
        # Only compute collision loss at small timesteps where pred_x0 is more accurate
        collision_loss = torch.tensor(0.0, device=device, dtype=hand_params.dtype)
        if self.lambda_collision > 0.0 and self.collision_loss_module is not None:
            # Create mask for samples with timestep < threshold
            # timesteps: [B], values in [0, diffuse_steps-1]
            valid_timesteps_mask = timesteps < self.collision_timestep_threshold  # [B]
            
            if valid_timesteps_mask.any():
                # Compute pred_x0 (denoised prediction) from x_noisy and pred_noise
                # pred_x0 = (x_noisy - sqrt(1 - alpha_bar_t) * pred_noise) / sqrt(alpha_bar_t)
                # Use precomputed sqrt coefficients for efficiency
                sqrt_alpha_bar_t = self.sqrt_alpha_bars_buf[timesteps].view(B, 1, 1)  # [B, 1, 1]
                sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bars_buf[timesteps].view(B, 1, 1)  # [B, 1, 1]
                pred_x0 = (x_noisy - sqrt_one_minus_alpha_bar_t * pred_noise) / sqrt_alpha_bar_t
                
                # Safety check: Replace NaN/Inf in pred_x0 before collision loss
                if torch.isnan(pred_x0).any() or torch.isinf(pred_x0).any():
                    print("[WARNING] pred_x0 has NaN/Inf after denoising! Clamping to [-10, 10]")
                    pred_x0 = torch.nan_to_num(pred_x0, nan=0.0, posinf=10.0, neginf=-10.0)
                
                # Extract denoised hand params and cmap params from pred_x0
                # pred_x0 contains [hand_params, cmap_params], where cmap_params are sh_params for contact map
                pred_hand_params = pred_x0[:, :, :self.hand_dim]  # [B, N, 22]
                pred_hand_cmap_params = pred_x0[:, :, self.hand_dim:self.hand_dim+self.cmap_dim]  # [B, N, 4]
                
                # Per-sample timestep weighting: smaller t → higher weight (pred_x0 more accurate)
                # Use exponential weighting: weight(t) = exp(-t / threshold)
                valid_timesteps = timesteps[valid_timesteps_mask].float()  # [valid_count]
                timestep_weights = torch.exp(-valid_timesteps / self.collision_timestep_threshold)  # [valid_count]
                # Normalize to sum to 1 for proper weighted average
                timestep_weights = timestep_weights / timestep_weights.sum()  # [valid_count]
                
                # Compute collision loss with per-sample timestep weighting
                
                collision_loss = self.collision_loss_module(
                    valid_timesteps_mask,
                    hand_params=pred_hand_params,
                    object_params=object_geom,
                    positions=positions,
                    denormalize=True,
                    hand_cmap_params=pred_hand_cmap_params,
                    sample_weights=timestep_weights,  # Pass per-sample weights
                    stats=stats
                )
        
        loss = recon_loss + self.lambda_collision * collision_loss
        
        # Final safety check: Ensure loss is finite
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"[ERROR] Final loss is NaN/Inf! recon_loss={recon_loss.item()}, collision_loss={collision_loss.item()}")
            # Return a dummy loss to skip this batch
            loss = torch.tensor(0.0, device=device, dtype=hand_params.dtype, requires_grad=True)

        tr = getattr(self, "trainer", None)
        if tr is not None and hasattr(tr, "_results") and hasattr(tr, "optimizers"):
            self.log("recon_loss", recon_loss, prog_bar=True, on_step=True, sync_dist=False)
            self.log("collision_loss", collision_loss, prog_bar=True, on_step=True, sync_dist=False)
            self.log("lr", float(self.trainer.optimizers[0].param_groups[0]['lr']), prog_bar=False, on_step=True, sync_dist=False)
        return loss

    def on_before_optimizer_step(self, optimizer) -> None:
        # Check for NaN gradients
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    print(f"[ERROR] NaN/Inf gradient in {name}!")
                    print(f"  grad max: {param.grad.max().item()}, min: {param.grad.min().item()}")
                    # Zero out the gradient to prevent corruption
                    param.grad.zero_()
        
        if self.global_step % 1000 == 0:
            norms = grad_norm(self.model, norm_type=2)
            self.log_dict(norms)

    def validation_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str, Dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        positions, hand_params, object_params, tem_hand, data_name, stats = batch
        B = positions.shape[0]
        device = hand_params.device
        hand_geom = hand_params[:, :, :self.hand_dim]
        hand_cmap = hand_params[:, :, self.hand_dim:]
        object_geom = object_params[:, :, :self.object_dim]
        object_cmap = object_params[:, :, self.object_dim:]
        feat = torch.cat([hand_geom, hand_cmap], dim=-1)
        x_t = torch.randn_like(feat)

        pbar = tqdm(range(self.diffuse_steps), desc="Validation")
        mse_step = None
        for t in reversed(range(self.diffuse_steps)):
            t_tensor = torch.full((B,), t, device=device, dtype=torch.long)
            pred_noise = self.forward(
                noisy_feat=x_t,
                object_feat=object_geom,
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
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str, Dict[str, torch.Tensor]]
    ) -> None:
        with torch.no_grad():
            positions, object_params, tem_hand, data_name_list, stats = batch
            B, N, _ = positions.shape
            device = positions.device

            object_geom = object_params[:, :, :self.object_dim]

            x_t = torch.randn(B, N, self.hand_dim + self.cmap_dim, device=device)

            pbar = tqdm(range(self.diffuse_steps), desc="Test")
            for t in reversed(range(self.diffuse_steps)):
                t_tensor = torch.full((B,), t, device=device, dtype=torch.long)
                pred_noise = self.forward(
                    noisy_feat=x_t,
                    object_feat=object_geom,
                    pos=positions,
                    tem_hand=tem_hand,
                    timesteps=t_tensor
                )

                # ==========================================================================================
                # Physics-Guided Sampling (PhysGuidance.md Section 4)
                # Core Idea: Use the gradient of the physical loss to guide the sampling process
                # towards physically valid regions.
                # ==========================================================================================
                guidance_grad = None    # [STILL IN DEVELOP]
                # if self.collision_loss_module is not None and self.guidance_scale > 0:
                #     with torch.enable_grad():
                #         x_in = x_t.detach().requires_grad_(True)
                        
                #         # 1. Estimate clean sample hat_x0 (Step 2 in PhysGuidedSample.md)
                #         # We use detached pred_noise for efficiency (standard approximation in guidance)
                #         sqrt_alpha_bar_t = self.sqrt_alpha_bars_buf[t_tensor].view(B, 1, 1)
                #         sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bars_buf[t_tensor].view(B, 1, 1)
                #         x0_hat = (x_in - sqrt_one_minus_alpha_bar_t * pred_noise) / sqrt_alpha_bar_t
                        
                #         hand_params_hat = x0_hat[:, :, :self.hand_dim]
                        
                #         # 2. Compute physical loss on estimated clean sample (Step 3)
                #         loss = self.collision_loss_module(
                #             hand_params=hand_params_hat,
                #             object_params=object_params,
                #             positions=positions,
                #             denormalize=True,
                #             stats=stats
                #         )
                        
                #         # 3. Compute gradient of loss w.r.t noisy input x_t (Step 3)
                #         # grad = \nabla_{x_t} L_{PA}(\hat{x}_0(x_t))
                #         loss_value = float(loss.detach().item())
                #         if loss_value > self.guidance_threshold and loss.requires_grad:
                #             grad = torch.autograd.grad(
                #                 loss, 
                #                 x_in, 
                #                 retain_graph=False, 
                #                 create_graph=False, 
                #                 allow_unused=True
                #             )[0]
                #             if grad is not None:
                #                 guidance_grad = grad

                # 4. Modify mean/noise and Sample (Step 4 & 5)
                # Supports both standard guidance and Spherical Gaussian Constraint
                x_t = self.p_sample_step(
                    x_t, 
                    t_tensor, 
                    pred_noise, 
                    guidance_grad=guidance_grad, 
                    guidance_scale=self.guidance_scale,
                    use_spherical_gaussian=self.use_spherical_gaussian
                )
                pbar.update(1)

            os.makedirs(self.output_dir, exist_ok=True)
            for i in torch.randperm(B):
                data_name = Path(data_name_list[i]).stem
                hand_cmap_params_single = x_t[i]
                object_params_single = object_params[i]
                positions_single = positions[i]

                hand_params_single = hand_cmap_params_single[:, :self.hand_dim]
                
                hand_mask = hand_params_single[:, :self.mask_dim].detach().cpu().numpy()
                hand_sh = hand_params_single[:, self.mask_dim:self.mask_dim + self.sh_dim].detach().cpu().numpy()
                hand_ortho = hand_params_single[:, self.mask_dim + self.sh_dim:].detach().cpu().numpy()
                hand_cmap = hand_cmap_params_single[:, self.hand_dim:self.hand_dim + self.cmap_dim]

                object_mask = object_params_single[:, :self.mask_dim].detach().cpu().numpy()
                object_sh = object_params_single[:, self.mask_dim:self.mask_dim + self.sh_dim].detach().cpu().numpy()
                object_ortho = object_params_single[:, self.mask_dim + self.sh_dim:].detach().cpu().numpy()
                object_cmap = torch.zeros_like(hand_cmap)

                hand_cmap = hand_cmap.detach().cpu().numpy()
                object_cmap = object_cmap.detach().cpu().numpy()

                if self.statistics is None:
                    pos_out = positions_single.detach().cpu().numpy()
                    hand_mask_out = hand_mask
                    hand_sh_out = hand_sh
                    hand_ortho_out = hand_ortho
                    obj_mask_out = object_mask
                    obj_sh_out = object_sh
                    obj_ortho_out = object_ortho
                    hand_cmap_out = hand_cmap
                    object_cmap_out = object_cmap
                else:
                    pos_out = positions_single.detach().cpu().numpy() * self.statistics["std_positions"].cpu().numpy() + self.statistics["mean_positions"].cpu().numpy()
                    hand_mask_out = hand_mask * self.statistics["std_hand_mask"].cpu().numpy() + self.statistics["mean_hand_mask"].cpu().numpy()
                    hand_sh_out = hand_sh * self.statistics["std_hand_sh"].cpu().numpy() + self.statistics["mean_hand_sh"].cpu().numpy()
                    hand_ortho_out = hand_ortho * self.statistics["std_hand_ortho"].cpu().numpy() + self.statistics["mean_hand_ortho"].cpu().numpy()
                    hand_cmap_out = hand_cmap * self.statistics["std_hand_cmap"].cpu().numpy() + self.statistics["mean_hand_cmap"].cpu().numpy()

                    obj_mask_out = object_mask * self.statistics["std_obj_mask"].cpu().numpy() + self.statistics["mean_obj_mask"].cpu().numpy()
                    obj_sh_out = object_sh * self.statistics["std_obj_sh"].cpu().numpy() + self.statistics["mean_obj_sh"].cpu().numpy()
                    obj_ortho_out = object_ortho * self.statistics["std_obj_ortho"].cpu().numpy() + self.statistics["mean_obj_ortho"].cpu().numpy()
                    object_cmap_out = object_cmap * self.statistics["std_obj_cmap"].cpu().numpy() + self.statistics["mean_obj_cmap"].cpu().numpy()

                generate_dict = {
                    "positions": pos_out,
                    "hand_params": {
                        "mask_params": hand_mask_out,
                        "sh_params": hand_sh_out,
                        "ortho_poses": hand_ortho_out,
                        "cmap_params": hand_cmap_out
                    },
                    "object_params": {
                        "mask_params": obj_mask_out,
                        "sh_params": obj_sh_out,
                        "ortho_poses": obj_ortho_out,
                        "cmap_params": object_cmap_out
                    }
                }
                np.save(os.path.join(self.output_dir, f"gen_{self.hand_type}_{data_name}.npy"), generate_dict)

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

    def p_sample_step(
        self, 
        x_t: torch.Tensor, 
        t: torch.Tensor, 
        pred_noise: torch.Tensor,
        guidance_grad: Optional[torch.Tensor] = None,
        guidance_scale: float = 1.0,
        use_spherical_gaussian: bool = False
    ) -> torch.Tensor:
        alpha_t = self.alphas_buf[t].view(-1, 1, 1)
        beta_t = self.betas_buf[t].view(-1, 1, 1)
        alpha_bar_t = self.alpha_bars_buf[t].view(-1, 1, 1)
        prev_alpha_bar = self.alpha_bars_prev_buf[t].view(-1, 1, 1)
        coef = beta_t / torch.sqrt(1.0 - alpha_bar_t)
        mean = (1.0 / torch.sqrt(alpha_t)) * (x_t - coef * pred_noise)
        posterior_var = beta_t * (1.0 - prev_alpha_bar) / (1.0 - alpha_bar_t)

        # Physics-Guided Sampling
        if guidance_grad is not None:
            if use_spherical_gaussian:
                # Spherical Gaussian Constraint: modify direction instead of directly adjusting mean
                d_sample = -pred_noise
                grad_norm = torch.norm(guidance_grad, dim=-1, keepdim=True) + 1e-8
                d_star = -guidance_grad / grad_norm
                d_m = d_sample + guidance_scale * (d_star - d_sample)
                mean = (1.0 / torch.sqrt(alpha_t)) * (x_t + coef * d_m)
            else:
                # Standard Guidance (Eq 13): mu_tilde = mu - s * Sigma * grad
                mean = mean - guidance_scale * posterior_var * guidance_grad

        posterior_std = torch.sqrt(posterior_var)
        noise = torch.randn_like(x_t)
        nonzero_mask = (t > 0).float().view(-1, 1, 1).expand_as(x_t)
        posterior_std = posterior_std.expand_as(x_t)
        x_prev = mean + nonzero_mask * posterior_std * noise
        return x_prev
