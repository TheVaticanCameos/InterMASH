import torch
import torch.nn as nn

from .network_general import SinusoidalPositionEmbeddings, AdaLN, GeGLU


class PointNet(nn.Module):
    def __init__(self, d_in: int, d_feat: int, d_out: int) -> None:
        super().__init__()
        self.t_emb = SinusoidalPositionEmbeddings(d_feat)
        self.ada_ln = AdaLN(d_feat)
        self.up = nn.Sequential(nn.Linear(d_in, d_feat), nn.SiLU())
        self.blocks = nn.ModuleList([
            GeGLU(d_feat * 2, d_feat) for _ in range(8)
        ])
        self.proj = nn.Linear(d_feat, d_out)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Input x: (B, N, d_in)
              t: (B, 1)
                d_in = 22(hand) + 22(object) + 3(position)
        
        Output: (B, N, d_out)
            d_out = 22(hand)
        """
        t_emb = self.t_emb(t)   # (B, d_feat)
        x = self.up(x)  # x: (B, N, d_feat)
        x = self.ada_ln(x, t_emb)
        for block in self.blocks:
            global_features = x.mean(dim=1, keepdim=True).repeat(1, x.shape[1], 1)
            x = block(torch.cat([x, global_features], dim=-1)) + x
        return self.proj(x)
