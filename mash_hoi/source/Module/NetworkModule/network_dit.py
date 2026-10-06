import torch
import torch.nn as nn

from typing import Optional, Tuple

from .network_general import AdaLN, SelfAttention, CrossAttention, GeGLU, SinusoidalPositionEmbeddings, GraphSelfAttention
from mash_hoi.source.Utils.constant import MASK_DIM, SH_DIM, ORTHO_DIM, CMAP_DIM


class DiTBlockV1(nn.Module):
    def __init__(self, d_feat: int) -> None:
        super().__init__()
        self.ada_ln1 = AdaLN(d_feat)
        self.self_attn = SelfAttention(d_feat, num_heads=8, phys_dim=3)
        self.ada_ln2 = AdaLN(d_feat)
        self.mlp = GeGLU(d_feat, d_feat)

    def forward(self, x: torch.Tensor, cond_emb: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """
        Input 
            x: (B, N, d_feat)
            cond_emb: (B, N, d_feat)
            pos: (B, N, 3)
        Output
            (B, N, d_feat)
        """
        x = self.ada_ln1(x, cond_emb)
        x = self.self_attn(x, pos) + x
        x = self.ada_ln2(x, cond_emb)
        x = self.mlp(x) + x
        return x


class DiTBlockV2(nn.Module):
    def __init__(self, d_feat: int, num_heads: int = 8, head_dim: Optional[int] = None, phys_dim: int = 3) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_feat)
        self.ln2 = nn.LayerNorm(d_feat)
        self.ln3 = nn.LayerNorm(d_feat)
        self.ada_ln1 = AdaLN(d_feat)
        self.self_attn = SelfAttention(d_feat, num_heads=num_heads, phys_dim=phys_dim, head_dim=head_dim)
        self.cross_attn = CrossAttention(d_feat, num_heads=num_heads, phys_dim=phys_dim, head_dim=head_dim)
        self.mlp = GeGLU(d_feat, d_feat)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, pos: torch.Tensor, object_emb: torch.Tensor) -> torch.Tensor:
        """
        Input 
            x: (B, N, d_feat)
            t_emb: (B, N, d_feat)
            pos: (B, N, 3)
            object_emb: (B, N, d_feat)
        Output
            (B, N, d_feat)
        """
        g1, s1, g2, s2, g3, s3, alpha = self.ada_ln1(t_emb)
        norm_x = self.ln1(x) * (1 + g1) + s1
        x = self.self_attn(norm_x, pos) + x
        norm_x = self.ln2(x) * (1 + g2) + s2
        x = self.cross_attn(norm_x, object_emb, pos) + x
        norm_x = self.ln3(x) * (1 + g3) + s3
        x = self.mlp(norm_x) * (1 + alpha) + x
        return x


class DiTBlockV3(nn.Module):
    def __init__(self, d_feat: int, num_heads: int = 8, head_dim: Optional[int] = None, phys_dim: int = 3) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_feat)
        self.ln2 = nn.LayerNorm(d_feat)
        self.ln3 = nn.LayerNorm(d_feat)
        self.ln4 = nn.LayerNorm(d_feat)
        self.ada_ln1 = AdaLN(d_feat, chunk_num=9)
        self.self_attn = SelfAttention(d_feat, num_heads=num_heads, phys_dim=phys_dim, head_dim=head_dim)
        self.cross_attn_obj = CrossAttention(d_feat, num_heads=num_heads, phys_dim=phys_dim, head_dim=head_dim)
        self.cross_attn_tem = CrossAttention(d_feat, num_heads=num_heads, phys_dim=phys_dim, head_dim=head_dim)
        self.mlp = GeGLU(d_feat, d_feat)

    def forward(
        self, 
        x: torch.Tensor, 
        t_emb: torch.Tensor, 
        pos: torch.Tensor, 
        object_emb: torch.Tensor, 
        tem_emb: torch.Tensor, 
        pos_tem: torch.Tensor
    ) -> torch.Tensor:
        """
        Input 
            x: (B, N, d_feat)
            t_emb: (B, N, d_feat)
            pos: (B, N, 3)
            object_emb: (B, N, d_feat)
            tem_emb: (B, N, d_feat)
            pos_tem: (B, N, 3)
        Output
            (B, N, d_feat)
        """
        g1, s1, g2, s2, g3, s3, g4, s4, alpha = self.ada_ln1(t_emb)
        norm_x = self.ln1(x) * (1 + g1) + s1
        x = self.self_attn(norm_x, pos) + x
        norm_x = self.ln2(x) * (1 + g2) + s2
        x = self.cross_attn_tem(norm_x, tem_emb, pos_tem) + x
        norm_x = self.ln3(x) * (1 + g3) + s3
        x = self.cross_attn_obj(norm_x, object_emb, pos) + x
        norm_x = self.ln4(x) * (1 + g4) + s4
        x = self.mlp(norm_x) * (1 + alpha) + x
        return x


class DiTBlockV3Fixed(nn.Module):
    """
    Diffusion Transformer Block for fixed anchor setting
    """
    def __init__(self, d_feat: int, num_heads: int = 8, head_dim: Optional[int] = None, phys_dim: int = 3) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_feat)
        self.ln2 = nn.LayerNorm(d_feat)
        self.ln3 = nn.LayerNorm(d_feat)
        self.ln4 = nn.LayerNorm(d_feat)
        self.ada_ln = AdaLN(d_feat, chunk_num=9)
        self.self_attn = SelfAttention(d_feat, num_heads=num_heads, phys_dim=phys_dim, head_dim=head_dim)
        self.cross_attn_obj = CrossAttention(d_feat, num_heads=num_heads, phys_dim=phys_dim, head_dim=head_dim)
        self.cross_attn_tem = CrossAttention(d_feat, num_heads=num_heads, phys_dim=phys_dim, head_dim=head_dim)
        self.mlp = GeGLU(d_feat, d_feat)

    def forward(
        self, 
        x: torch.Tensor, 
        t_emb: torch.Tensor, 
        pos: torch.Tensor, 
        object_emb: torch.Tensor, 
        tem_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Input 
            x: (B, N, d_feat), cmap_emb cat hand_emb
            t_emb: (B, N, d_feat)
            pos: (B, N, 3)
            object_emb: (B, N, d_feat)
            tem_emb: (B, N, d_feat)
        Output
            (B, N, d_feat)
        """
        g1, s1, g2, s2, g3, s3, g4, s4, alpha = self.ada_ln(t_emb)
        norm_x = self.ln1(x) * (1 + g1) + s1
        x = self.self_attn(norm_x, pos) + x
        norm_x = self.ln2(x) * (1 + g2) + s2
        x = self.cross_attn_tem(norm_x, tem_emb, pos) + x
        norm_x = self.ln3(x) * (1 + g3) + s3
        x = self.cross_attn_obj(norm_x, object_emb, pos) + x
        norm_x = self.ln4(x) * (1 + g4) + s4
        x = self.mlp(norm_x) * (1 + alpha) + x
        return x


class Embedder(nn.Module):
    def __init__(self, d_in: int, d_feat: int, num_heads: int = 8, head_dim: Optional[int] = None) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_in, d_feat)
        self.silu = nn.SiLU()
        self.layer_norm = nn.LayerNorm(d_feat)
        self.self_attn = SelfAttention(d_feat, num_heads=num_heads, phys_dim=3, head_dim=head_dim)
        self.linear_2 = nn.Linear(d_feat, d_feat)
    
    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """
        Input 
            x: (B, N, d_in)
            pos: (B, N, 3)
        Output: 
            (B, N, d_feat)
        """
        x = self.linear_1(x)
        x = self.silu(x)
        x = self.layer_norm(x)
        x = self.self_attn(x, pos) + x
        x = self.linear_2(x)
        return x


class GraphEmbedder(nn.Module):
    """
    Graph-aware Embedder that uses GraphSelfAttention with KNN bias.
    
    Each patch attends more strongly to its K nearest neighbors during embedding.
    """
    def __init__(
        self, 
        d_in: int, 
        d_feat: int, 
        num_heads: int = 8, 
        head_dim: Optional[int] = None,
        knn_bias_scale: float = 2.0,
    ) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_in, d_feat)
        self.silu = nn.SiLU()
        self.layer_norm = nn.LayerNorm(d_feat)
        self.graph_self_attn = GraphSelfAttention(
            d_feat, 
            num_heads=num_heads, 
            phys_dim=3, 
            head_dim=head_dim,
            knn_bias_scale=knn_bias_scale,
        )
        self.linear_2 = nn.Linear(d_feat, d_feat)
    
    def forward(
        self, 
        x: torch.Tensor, 
        pos: torch.Tensor,
        knn_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Input 
            x: (B, N, d_in)
            pos: (B, N, 3)
            knn_indices: (B, N, K) KNN indices for graph attention, or None
        Output: 
            (B, N, d_feat)
        """
        x = self.linear_1(x)
        x = self.silu(x)
        x = self.layer_norm(x)
        x = self.graph_self_attn(x, pos, knn_indices=knn_indices, has_edge_token=False) + x
        x = self.linear_2(x)
        return x


class DiffusionTransformerV1(nn.Module):
    def __init__(self, d_in: int, d_feat: int, d_out: int, n_blocks: int) -> None:
        super().__init__()
        self.t_embedding = nn.Sequential(
            SinusoidalPositionEmbeddings(d_feat), 
            nn.SiLU(),
            nn.Linear(d_feat, d_feat),
            nn.SiLU())
        self.object_embedding = nn.Sequential(  # TODO: this object embedding could be reinforced
            nn.Linear(d_in, d_feat), 
            nn.SiLU(),
            nn.Linear(d_feat, d_feat),
            nn.SiLU())
        self.hand_up = nn.Sequential(
            nn.Linear(d_in, d_feat), 
            nn.SiLU())
        self.blocks = nn.ModuleList([
            DiTBlockV1(d_feat) for _ in range(n_blocks)
        ])
        self.proj = nn.Linear(d_feat, d_out)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Input x: (B, N, d_in)
              t: (B,)
                d_in = 22(hand) + 22(object) + 3(position)
        
        Output: (B, N, d_out)
            d_out = 22(hand)
        """
        hand = x[:, :, :22]
        object = x[:, :, 22:44]
        pos = x[:, :, 44:]
        t_emb = self.t_embedding(t)   # (B, d_feat)
        t_emb = t_emb.unsqueeze(1).repeat(1, pos.size(1), 1)    # (B, N, d_feat)
        object_emb = self.object_embedding(object)
        cond_emb = t_emb + object_emb
        hand_emb = self.hand_up(hand)

        for block in self.blocks:
            global_features = hand_emb.mean(dim=1, keepdim=True).repeat(1, hand_emb.shape[1], 1)
            hand_emb = block(hand_emb + global_features, cond_emb, pos) + hand_emb
            
        return self.proj(hand_emb)


class DiffusionTransformerV2(nn.Module):
    def __init__(self, d_in: int, d_feat: int, d_out: int, n_blocks: int) -> None:
        super().__init__()
        self.t_embedding = nn.Sequential(
            SinusoidalPositionEmbeddings(d_feat), 
            nn.SiLU(),
            nn.Linear(d_feat, d_feat))
        self.object_embedding = Embedder(d_in=22, d_feat=d_feat)
        self.hand_up = nn.Sequential(
            nn.Linear(d_in, d_feat))
        self.blocks = nn.ModuleList([
            DiTBlockV2(d_feat) for _ in range(n_blocks)
        ])
        self.proj = nn.Sequential(
            nn.Linear(d_feat, d_out)
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Input x: (B, N, d_in)
              t: (B,)
                d_in = 22(hand) + 22(object) + 3(position)
        
        Output: (B, N, d_out)
            d_out = 22(hand)
        """
        hand = x[:, :, :22]
        object = x[:, :, 22:44]
        pos = x[:, :, 44:]
        t_emb = self.t_embedding(t)   # (B, d_feat)
        t_emb = t_emb.unsqueeze(1).repeat(1, pos.size(1), 1)    # (B, N, d_feat)
        object_emb = self.object_embedding(object, pos)
        hand_emb = self.hand_up(hand)

        for i, block in enumerate(self.blocks):
            hand_emb = block(hand_emb, t_emb, pos, object_emb)
            
        return self.proj(hand_emb)


class DiffusionTransformerV2CMap(nn.Module):
    def __init__(self, d_in: int, d_feat: int, d_out: int, n_blocks: int, num_heads: int = 8, head_dim: Optional[int] = None) -> None:
        super().__init__()
        self.t_embedding = nn.Sequential(
            SinusoidalPositionEmbeddings(d_feat), 
            nn.SiLU(),
            nn.Linear(d_feat, d_feat))
        self.object_embedding = Embedder(d_in=22, d_feat=d_feat, num_heads=num_heads, head_dim=head_dim)
        self.hand_up = nn.Sequential(
            nn.Linear(22, d_feat // 2))
        self.cmap_up = nn.Sequential(
            nn.Linear(4, d_feat // 2))
        self.blocks = nn.ModuleList([
            DiTBlockV2(d_feat, num_heads=num_heads, head_dim=head_dim) for _ in range(n_blocks)
        ])
        self.proj = nn.Sequential(
            nn.Linear(d_feat, d_out)
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Input x: (B, N, d_in)
              t: (B,)
                d_in = 22(hand) + 4(cmap) + 22(object) + 3(position)
        
        Output: (B, N, d_out)
            d_out = 22(hand) + 4(cmap)
        """
        hand = x[:, :, :22]
        cmap = x[:, :, 22:26]
        object = x[:, :, 26:48]
        pos = x[:, :, 48:51]
        t_emb = self.t_embedding(t)
        t_emb = t_emb.unsqueeze(1).expand(-1, pos.size(1), -1)
        object_emb = self.object_embedding(object, pos)
        hand_emb = self.hand_up(hand)
        
        cmap_emb = self.cmap_up(cmap)
        feat = torch.cat([hand_emb, cmap_emb], dim=-1)

        for i, block in enumerate(self.blocks):
            feat = block(feat, t_emb, pos, object_emb)
            
        return self.proj(feat)


class DiffusionTransformerV3(nn.Module):
    def __init__(self, d_in: int, d_feat: int, d_out: int, n_blocks: int, num_heads: int = 8, head_dim: Optional[int] = None) -> None:
        super().__init__()
        self.t_embedding = nn.Sequential(
            SinusoidalPositionEmbeddings(d_feat), 
            nn.SiLU(),
            nn.Linear(d_feat, d_feat))
        self.object_embedding = Embedder(d_in=22, d_feat=d_feat, num_heads=num_heads, head_dim=head_dim)
        self.template_embedding = Embedder(d_in=22, d_feat=d_feat, num_heads=num_heads, head_dim=head_dim)
        self.hand_up = nn.Sequential(
            nn.Linear(22, d_feat // 2))
        self.cmap_up = nn.Sequential(
            nn.Linear(4, d_feat // 2))
        self.blocks = nn.ModuleList([
            DiTBlockV3(d_feat, num_heads=num_heads, head_dim=head_dim) for _ in range(n_blocks)
        ])
        self.proj = nn.Sequential(
            nn.Linear(d_feat, d_out)
        )
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Input x: (B, N, d_in)
              t: (B,)
                d_in = 22(hand) + 4(cmap) + 22(object) + 3(pos) + 22(template) + 3(tem_pos) = 76

        Output: (B, N, d_out)
                d_out = 22(hand) + 4(cmap)
        """
        hand = x[:, :, :22]
        cmap = x[:, :, 22:26]
        object = x[:, :, 26:48]
        pos = x[:, :, 48:51]
        template = x[:, :, 51:73]
        tem_pos = x[:, :, 73:]

        t_emb = self.t_embedding(t)
        t_emb = t_emb.unsqueeze(1).expand(-1, pos.size(1), -1)
        object_emb = self.object_embedding(object, pos)
        tem_emb = self.template_embedding(template, tem_pos)
        hand_emb = self.hand_up(hand)
        cmap_emb = self.cmap_up(cmap)

        feat = torch.cat([hand_emb, cmap_emb], dim=-1)

        for i, block in enumerate(self.blocks):
            feat = block(feat, t_emb, pos, object_emb, tem_emb, tem_pos)

        return self.proj(feat)


class DiffusionTransformerV3Fixed(nn.Module):
    def __init__(self, d_in: int, d_feat: int, d_out: int, n_blocks: int, num_heads: int = 8, head_dim: Optional[int] = None, use_hand_cmap: bool = False) -> None:
        super().__init__()
        self.use_hand_cmap = use_hand_cmap
        self.t_embedding = nn.Sequential(
            SinusoidalPositionEmbeddings(d_feat), 
            nn.SiLU(),
            nn.Linear(d_feat, d_feat)
        )
        self.object_embedding = Embedder(d_in=22, d_feat=d_feat, num_heads=num_heads, head_dim=head_dim)
        self.template_embedding = Embedder(d_in=22, d_feat=d_feat, num_heads=num_heads, head_dim=head_dim)
        self.hand_up = nn.Sequential(
            nn.Linear(22, d_feat // 2),
            nn.SiLU(),
            nn.Linear(d_feat // 2, d_feat // 2)
        )
        # if use_hand_cmap:
        #     # With hand_cmap: hand(d_feat//2) + hand_cmap(d_feat//4) + object_cmap(d_feat//4) = d_feat
        #     self.cmap_up = nn.Sequential(
        #         nn.Linear(4, d_feat // 2),
        #         nn.SiLU(),
        #         nn.Linear(d_feat // 2, d_feat // 4)
        #     )
        # else:
        #     # Without hand_cmap: hand(d_feat//2) + object_cmap(d_feat//2) = d_feat
        #     self.cmap_up = nn.Sequential(
        #         nn.Linear(4, d_feat // 2),
        #         nn.SiLU(),
        #         nn.Linear(d_feat // 2, d_feat // 2)
        #     )
        self.cmap_up = nn.Sequential(
            nn.Linear(4, d_feat // 2),
            nn.SiLU(),
            nn.Linear(d_feat // 2, d_feat // 2)
        )
        self.blocks = nn.ModuleList([
            DiTBlockV3Fixed(d_feat, num_heads=num_heads, head_dim=head_dim) for _ in range(n_blocks)
        ])
        self.proj = nn.Sequential(
            nn.Linear(d_feat, d_feat),
            nn.SiLU(),
            nn.Linear(d_feat, d_out)
        )
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Input x: (B, N, d_in)
              t: (B,)
              If use_hand_cmap=True:
                d_in = 22(hand) + 4(hand_cmap) + 4(object_cmap) + 22(object) + 3(pos) + 22(template) = 77
                d_out = 22(hand) + 4(hand_cmap) + 4(object_cmap) = 30
              If use_hand_cmap=False:
                d_in = 22(hand) + 4(object_cmap) + 22(object) + 3(pos) + 22(template) = 73
                d_out = 22(hand) + 4(object_cmap) = 26

        Output: (B, N, d_out)
        """
        if self.use_hand_cmap:
            hand = x[:, :, :22]
            hand_cmap = x[:, :, 22:26]
            object_cmap = x[:, :, 26:30]
            object = x[:, :, 30:52]
            pos = x[:, :, 52:55]
            template = x[:, :, 55:77]
        else:
            hand = x[:, :, :22]
            object_cmap = x[:, :, 22:26]
            object = x[:, :, 26:48]
            pos = x[:, :, 48:51]
            template = x[:, :, 51:73]

        t_emb = self.t_embedding(t)
        t_emb = t_emb.unsqueeze(1).expand(-1, pos.size(1), -1)
        object_emb = self.object_embedding(object, pos)
        tem_emb = self.template_embedding(template, pos)
        hand_emb = self.hand_up(hand)
        object_cmap_emb = self.cmap_up(object_cmap)

        if self.use_hand_cmap:
            hand_cmap_emb = self.cmap_up(hand_cmap)
            feat = torch.cat([hand_emb, hand_cmap_emb, object_cmap_emb], dim=-1)
        else:
            feat = torch.cat([hand_emb, object_cmap_emb], dim=-1)

        for i, block in enumerate(self.blocks):
            feat = block(feat, t_emb, pos, object_emb, tem_emb)

        return self.proj(feat)


class DiTBlockV3Graph(nn.Module):
    """
    DiT block with graph-aware attention for fixed anchor setting.
    
    Uses GraphSelfAttention for hand features and GraphCrossAttention for object cross-attention.
    Supports edge_token at position 0 for global edge latent generation.
    """
    def __init__(
        self, 
        d_feat: int, 
        num_heads: int = 8, 
        head_dim: Optional[int] = None, 
        phys_dim: int = 3,
        knn_bias_scale: float = 2.0,
    ) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_feat)
        self.ln2 = nn.LayerNorm(d_feat)
        self.ln3 = nn.LayerNorm(d_feat)
        self.ln4 = nn.LayerNorm(d_feat)
        self.ada_ln = AdaLN(d_feat, chunk_num=9)
        self.graph_self_attn = GraphSelfAttention(
            d_feat, 
            num_heads=num_heads, 
            phys_dim=phys_dim, 
            head_dim=head_dim,
            knn_bias_scale=knn_bias_scale,
        )
        # Note: Object cross-attention uses regular CrossAttention (not graph-aware)
        # because obj_knn_indices describes object-object neighbors, not hand-object relationships
        self.cross_attn_obj = CrossAttention(d_feat, num_heads=num_heads, phys_dim=phys_dim, head_dim=head_dim)
        self.cross_attn_tem = CrossAttention(d_feat, num_heads=num_heads, phys_dim=phys_dim, head_dim=head_dim)
        self.mlp = GeGLU(d_feat, d_feat)

    def forward(
        self, 
        x: torch.Tensor, 
        t_emb: torch.Tensor, 
        pos: torch.Tensor, 
        object_emb: torch.Tensor, 
        tem_emb: torch.Tensor,
        hand_knn_indices: Optional[torch.Tensor] = None,
        obj_knn_indices: Optional[torch.Tensor] = None,
        has_edge_token: bool = True,
    ) -> torch.Tensor:
        """
        Input 
            x: (B, N+1, d_feat) if has_edge_token else (B, N, d_feat)
               edge_token at position 0, anchor tokens at positions 1:N+1
            t_emb: (B, N+1, d_feat) or (B, N, d_feat)
            pos: (B, N, 3) anchor positions only
            object_emb: (B, N, d_feat)
            tem_emb: (B, N, d_feat)
            hand_knn_indices: (B, N, K) hand KNN indices for self-attention
            obj_knn_indices: (B, N, K) object KNN indices for cross-attention
            has_edge_token: whether x includes edge_token at position 0
        Output
            (B, N+1, d_feat) or (B, N, d_feat)
        """
        g1, s1, g2, s2, g3, s3, g4, s4, alpha = self.ada_ln(t_emb)
        
        # Graph self-attention (with hand KNN bias)
        norm_x = self.ln1(x) * (1 + g1) + s1
        x = self.graph_self_attn(norm_x, pos, knn_indices=hand_knn_indices, has_edge_token=has_edge_token) + x
        
        # For cross-attention, we need to handle edge_token separately
        if has_edge_token:
            # Split edge_token and anchor tokens
            edge_token = x[:, :1, :]  # [B, 1, d_feat]
            anchor_x = x[:, 1:, :]   # [B, N, d_feat]
            
            edge_t_emb = t_emb[:, :1, :]
            anchor_t_emb = t_emb[:, 1:, :]
            
            # Cross-attention for anchor tokens only
            g2_anchor = g2[:, 1:, :]
            s2_anchor = s2[:, 1:, :]
            g3_anchor = g3[:, 1:, :]
            s3_anchor = s3[:, 1:, :]
            g4_anchor = g4[:, 1:, :]
            s4_anchor = s4[:, 1:, :]
            alpha_anchor = alpha[:, 1:, :]
            
            norm_anchor = self.ln2(anchor_x) * (1 + g2_anchor) + s2_anchor
            anchor_x = self.cross_attn_tem(norm_anchor, tem_emb, pos) + anchor_x
            
            # Cross-attention with object (each hand patch attends to all object patches)
            norm_anchor = self.ln3(anchor_x) * (1 + g3_anchor) + s3_anchor
            anchor_x = self.cross_attn_obj(norm_anchor, object_emb, pos) + anchor_x
            
            norm_anchor = self.ln4(anchor_x) * (1 + g4_anchor) + s4_anchor
            anchor_x = self.mlp(norm_anchor) * (1 + alpha_anchor) + anchor_x
            
            # edge_token goes through MLP only (no cross-attention with object/template)
            g4_edge = g4[:, :1, :]
            s4_edge = s4[:, :1, :]
            alpha_edge = alpha[:, :1, :]
            norm_edge = self.ln4(edge_token) * (1 + g4_edge) + s4_edge
            edge_token = self.mlp(norm_edge) * (1 + alpha_edge) + edge_token
            
            # Concatenate back
            x = torch.cat([edge_token, anchor_x], dim=1)
        else:
            # Standard processing without edge_token
            norm_x = self.ln2(x) * (1 + g2) + s2
            x = self.cross_attn_tem(norm_x, tem_emb, pos) + x
            norm_x = self.ln3(x) * (1 + g3) + s3
            x = self.cross_attn_obj(norm_x, object_emb, pos) + x
            norm_x = self.ln4(x) * (1 + g4) + s4
            x = self.mlp(norm_x) * (1 + alpha) + x
        
        return x


class DiffusionTransformerV3Graph(nn.Module):
    """
    Graph-aware Diffusion Transformer with edge_token for edge latent generation.
    
    Features:
    - Graph self-attention with KNN bias for spatial relationships
    - edge_token for generating global edge latent code
    - Dual output heads: per-anchor noise and edge latent noise
    """
    def __init__(
        self, 
        d_in: int, 
        d_feat: int, 
        d_out: int, 
        n_blocks: int, 
        latent_dim: int = 64,
        num_heads: int = 8, 
        head_dim: Optional[int] = None,
        knn_bias_scale: float = 2.0,
    ) -> None:
        super().__init__()
        self.d_feat = d_feat
        self.latent_dim = latent_dim
        
        # Timestep embedding
        self.t_embedding = nn.Sequential(
            SinusoidalPositionEmbeddings(d_feat), 
            nn.SiLU(),
            nn.Linear(d_feat, d_feat)
        )
        
        # Input embeddings
        # Object embedding uses GraphEmbedder with obj_knn for graph-aware feature extraction
        self.object_embedding = GraphEmbedder(
            d_in=22, d_feat=d_feat, num_heads=num_heads, head_dim=head_dim, knn_bias_scale=knn_bias_scale
        )
        self.template_embedding = Embedder(d_in=22, d_feat=d_feat, num_heads=num_heads, head_dim=head_dim)
        self.hand_up = nn.Sequential(
            nn.Linear(22, d_feat // 2),
            nn.SiLU(),
            nn.Linear(d_feat // 2, d_feat // 2)
        )
        self.cmap_up = nn.Sequential(
            nn.Linear(4, d_feat // 2),
            nn.SiLU(),
            nn.Linear(d_feat // 2, d_feat // 2)
        )
        
        # edge latent embedding (for noisy latent input)
        self.latent_up = nn.Sequential(
            nn.Linear(latent_dim, d_feat),
            nn.SiLU(),
            nn.Linear(d_feat, d_feat)
        )
        
        # Learnable edge_token
        self.edge_token = nn.Parameter(torch.randn(1, 1, d_feat) * 0.02)
        
        # Graph-aware DiT blocks
        self.blocks = nn.ModuleList([
            DiTBlockV3Graph(
                d_feat, 
                num_heads=num_heads, 
                head_dim=head_dim,
                knn_bias_scale=knn_bias_scale,
            ) for _ in range(n_blocks)
        ])
        
        # Multiple output heads
        self.proj_to_mask = nn.Sequential(
            nn.Linear(d_feat, d_feat),
            nn.SiLU(),
            nn.Linear(d_feat, MASK_DIM)
        )
        self.proj_to_sh = nn.Sequential(
            nn.Linear(d_feat, d_feat),
            nn.SiLU(),
            nn.Linear(d_feat, SH_DIM)
        )
        self.proj_to_ortho = nn.Sequential(
            nn.Linear(d_feat, d_feat),
            nn.SiLU(),
            nn.Linear(d_feat, ORTHO_DIM)
        )
        self.proj_to_object_cmap = nn.Sequential(
            nn.Linear(d_feat, d_feat),
            nn.SiLU(),
            nn.Linear(d_feat, CMAP_DIM)
        )
        self.latent_proj = nn.Sequential(
            nn.Linear(d_feat, d_feat),
            nn.SiLU(),
            nn.Linear(d_feat, latent_dim)  # 64 for edge latent
        )
    
    def forward(
        self, 
        x: torch.Tensor, 
        t: torch.Tensor,
        noisy_latent: Optional[torch.Tensor] = None,
        hand_knn_indices: Optional[torch.Tensor] = None,
        obj_knn_indices: Optional[torch.Tensor] = None,
    ) -> tuple:
        """
        Input:
            x: (B, N, d_in) where d_in = 22(hand) + 4(cmap) + 22(object) + 3(pos) + 22(template) = 73
            t: (B,) timesteps
            noisy_latent: (B, latent_dim) noisy edge latent code, or None
            hand_knn_indices: (B, N, K) hand KNN indices for graph self-attention
            obj_knn_indices: (B, N, K) object KNN indices for object embedding

        Output:
            anchor_noise: (B, N, d_out) predicted noise for per-anchor features
            latent_noise: (B, latent_dim) predicted noise for edge latent
        """
        B, N, _ = x.shape
        
        # Parse input
        hand = x[:, :, :22]
        cmap = x[:, :, 22:26]
        obj = x[:, :, 26:48]
        pos = x[:, :, 48:51]
        template = x[:, :, 51:73]

        # Embeddings
        t_emb = self.t_embedding(t)  # [B, d_feat]
        t_emb_anchor = t_emb.unsqueeze(1).expand(-1, N, -1)  # [B, N, d_feat]
        
        # Object embedding with graph attention (using obj_knn)
        object_emb = self.object_embedding(obj, pos, knn_indices=obj_knn_indices)  # [B, N, d_feat]
        tem_emb = self.template_embedding(template, pos)  # [B, N, d_feat]
        hand_emb = self.hand_up(hand)  # [B, N, d_feat//2]
        cmap_emb = self.cmap_up(cmap)  # [B, N, d_feat//2]
        
        anchor_feat = torch.cat([hand_emb, cmap_emb], dim=-1)  # [B, N, d_feat]
        
        # Prepare edge_token
        edge_tokens = self.edge_token.expand(B, -1, -1)  # [B, 1, d_feat]
        if noisy_latent is not None:
            # Add noisy latent information to edge_token
            latent_emb = self.latent_up(noisy_latent)  # [B, d_feat]
            edge_tokens = edge_tokens + latent_emb.unsqueeze(1)  # [B, 1, d_feat]
        
        # Concatenate edge_token with anchor tokens
        feat = torch.cat([edge_tokens, anchor_feat], dim=1)  # [B, N+1, d_feat]
        
        # Prepare t_emb for all tokens (edge_token + anchors)
        t_emb_edge = t_emb.unsqueeze(1)  # [B, 1, d_feat]
        t_emb_full = torch.cat([t_emb_edge, t_emb_anchor], dim=1)  # [B, N+1, d_feat]

        # Process through DiT blocks
        for block in self.blocks:
            feat = block(
                feat, 
                t_emb_full, 
                pos, 
                object_emb, 
                tem_emb,
                hand_knn_indices=hand_knn_indices,
                obj_knn_indices=obj_knn_indices,
                has_edge_token=True,
            )

        # Split outputs
        edge_out = feat[:, 0, :]  # [B, d_feat]
        anchor_out = feat[:, 1:, :]  # [B, N, d_feat]
        
        # Project to output dimensions
        anchor_mask_noise = self.proj_to_mask(anchor_out)
        anchor_sh_noise = self.proj_to_sh(anchor_out)
        anchor_ortho_noise = self.proj_to_ortho(anchor_out)
        object_cmap_noise = self.proj_to_object_cmap(anchor_out)
        anchor_noise = torch.cat(
            [anchor_mask_noise, anchor_sh_noise, anchor_ortho_noise, object_cmap_noise],
            dim=-1,
        )
        latent_noise = self.latent_proj(edge_out)  # [B, latent_dim]
        
        return anchor_noise, latent_noise


class DiffusionTransformerV4Graph(nn.Module):
    """ [VERSION 4]: with anchor P.E. as supervision
    Graph-aware Diffusion Transformer with edge_token for edge latent generation.
    
    Features:
    - Graph self-attention with KNN bias for spatial relationships
    - edge_token for generating global edge latent code
    - Dual output heads: per-anchor noise and edge latent noise
    - Anchor P.E. as supervision
    """
    def __init__(
        self, 
        d_in: int, 
        d_feat: int, 
        d_out: int, 
        n_blocks: int, 
        pe_dim: int = 48,
        latent_dim: int = 64,
        num_heads: int = 8, 
        head_dim: Optional[int] = None,
        knn_bias_scale: float = 2.0,
    ) -> None:
        super().__init__()
        self.d_feat = d_feat
        self.latent_dim = latent_dim
        self.pe_dim = pe_dim

        expected_d_out = MASK_DIM + SH_DIM + ORTHO_DIM + CMAP_DIM + pe_dim
        if d_out != expected_d_out:
            raise ValueError(f"d_out must be {expected_d_out} (mask+sh+ortho+cmap+pe), got {d_out}")

        if d_feat % 3 != 0:
            raise ValueError(f"d_feat must be divisible by 3, got {d_feat}")
        local_feat = d_feat // 3
        
        # Timestep embedding
        self.t_embedding = nn.Sequential(
            SinusoidalPositionEmbeddings(d_feat), 
            nn.SiLU(),
            nn.Linear(d_feat, d_feat)
        )
        
        # Input embeddings
        # Object embedding uses GraphEmbedder with obj_knn for graph-aware feature extraction
        self.object_embedding = GraphEmbedder(
            d_in=22, d_feat=d_feat, num_heads=num_heads, head_dim=head_dim, knn_bias_scale=knn_bias_scale
        )
        self.template_embedding = Embedder(d_in=22, d_feat=d_feat, num_heads=num_heads, head_dim=head_dim)
        self.hand_up = nn.Sequential(
            nn.Linear(22, local_feat),
            nn.SiLU(),
            nn.Linear(local_feat, local_feat)
        )
        self.cmap_up = nn.Sequential(
            nn.Linear(4, local_feat),
            nn.SiLU(),
            nn.Linear(local_feat, local_feat)
        )
        self.pe_up = nn.Sequential(
            nn.Linear(pe_dim, local_feat),
            nn.SiLU(),
            nn.Linear(local_feat, local_feat)
        )
        
        # edge latent embedding (for noisy latent input)
        self.latent_up = nn.Sequential(
            nn.Linear(latent_dim, d_feat),
            nn.SiLU(),
            nn.Linear(d_feat, d_feat)
        )
        
        # Learnable edge_token
        self.edge_token = nn.Parameter(torch.randn(1, 1, d_feat) * 0.02)
        
        # Graph-aware DiT blocks
        self.blocks = nn.ModuleList([
            DiTBlockV3Graph(
                d_feat, 
                num_heads=num_heads, 
                head_dim=head_dim,
                knn_bias_scale=knn_bias_scale,
            ) for _ in range(n_blocks)
        ])
        
        self.proj_to_mask = nn.Sequential(
            nn.Linear(d_feat, d_feat),
            nn.SiLU(),
            nn.Linear(d_feat, MASK_DIM)
        )
        self.proj_to_sh = nn.Sequential(
            nn.Linear(d_feat, d_feat),
            nn.SiLU(),
            nn.Linear(d_feat, SH_DIM)
        )
        self.proj_to_ortho = nn.Sequential(
            nn.Linear(d_feat, d_feat),
            nn.SiLU(),
            nn.Linear(d_feat, ORTHO_DIM)
        )
        self.proj_to_object_cmap = nn.Sequential(
            nn.Linear(d_feat, d_feat),
            nn.SiLU(),
            nn.Linear(d_feat, CMAP_DIM)
        )
        self.proj_to_anchor_pe = nn.Sequential(
            nn.Linear(d_feat, d_feat),
            nn.SiLU(),
            nn.Linear(d_feat, pe_dim)
        )
        self.latent_proj = nn.Sequential(
            nn.Linear(d_feat, d_feat),
            nn.SiLU(),
            nn.Linear(d_feat, latent_dim)  # 64 for edge latent
        )
    
    def forward(
        self, 
        x: torch.Tensor, 
        t: torch.Tensor,
        noisy_latent: Optional[torch.Tensor] = None,
        hand_knn_indices: Optional[torch.Tensor] = None,
        obj_knn_indices: Optional[torch.Tensor] = None,
    ) -> tuple:
        """
        Input:
            x: (B, N, d_in) where d_in = 22(hand) + 4(cmap) + 22(object) + 3(pos) + 22(template) = 73
            t: (B,) timesteps
            noisy_latent: (B, latent_dim) noisy edge latent code, or None
            hand_knn_indices: (B, N, K) hand KNN indices for graph self-attention
            obj_knn_indices: (B, N, K) object KNN indices for object embedding

        Output:
            anchor_noise: (B, N, d_out) predicted noise for per-anchor features
            latent_noise: (B, latent_dim) predicted noise for edge latent
        """
        B, N, _ = x.shape
        
        # Parse input
        hand = x[:, :, :22]
        cmap = x[:, :, 22:26]
        pe = x[:, :, 26:26 + self.pe_dim]
        obj_start = 26 + self.pe_dim
        obj_end = obj_start + 22
        pos_start = obj_end
        pos_end = pos_start + 3
        tem_start = pos_end
        tem_end = tem_start + 22

        obj = x[:, :, obj_start:obj_end]
        pos = x[:, :, pos_start:pos_end]
        template = x[:, :, tem_start:tem_end]

        # Embeddings
        t_emb = self.t_embedding(t)  # [B, d_feat]
        t_emb_anchor = t_emb.unsqueeze(1).expand(-1, N, -1)  # [B, N, d_feat]
        
        # Object embedding with graph attention (using obj_knn)
        object_emb = self.object_embedding(obj, pos, knn_indices=obj_knn_indices)  # [B, N, d_feat]
        tem_emb = self.template_embedding(template, pos)  # [B, N, d_feat]
        hand_emb = self.hand_up(hand)  # [B, N, d_feat]
        cmap_emb = self.cmap_up(cmap)  # [B, N, d_feat]
        pe_emb = self.pe_up(pe)  # [B, N, d_feat]
        anchor_feat = torch.cat([hand_emb, cmap_emb, pe_emb], dim=-1)  # [B, N, d_feat]
        
        # Prepare edge_token
        edge_tokens = self.edge_token.expand(B, -1, -1)  # [B, 1, d_feat]
        if noisy_latent is not None:
            # Add noisy latent information to edge_token
            latent_emb = self.latent_up(noisy_latent)  # [B, d_feat]
            edge_tokens = edge_tokens + latent_emb.unsqueeze(1)  # [B, 1, d_feat]
        
        # Concatenate edge_token with anchor tokens
        feat = torch.cat([edge_tokens, anchor_feat], dim=1)  # [B, N+1, d_feat]
        
        # Prepare t_emb for all tokens (edge_token + anchors)
        t_emb_edge = t_emb.unsqueeze(1)  # [B, 1, d_feat]
        t_emb_full = torch.cat([t_emb_edge, t_emb_anchor], dim=1)  # [B, N+1, d_feat]

        # Process through DiT blocks
        for block in self.blocks:
            feat = block(
                feat, 
                t_emb_full, 
                pos, 
                object_emb, 
                tem_emb,
                hand_knn_indices=hand_knn_indices,
                obj_knn_indices=obj_knn_indices,
                has_edge_token=True,
            )

        # Split outputs
        edge_out = feat[:, 0, :]  # [B, d_feat]
        anchor_out = feat[:, 1:, :]  # [B, N, d_feat]
        
        anchor_mask_noise = self.proj_to_mask(anchor_out)
        anchor_sh_noise = self.proj_to_sh(anchor_out)
        anchor_ortho_noise = self.proj_to_ortho(anchor_out)
        object_cmap_noise = self.proj_to_object_cmap(anchor_out)
        anchor_pe_noise = self.proj_to_anchor_pe(anchor_out)
        anchor_noise = torch.cat(
            [anchor_mask_noise, anchor_sh_noise, anchor_ortho_noise, object_cmap_noise, anchor_pe_noise],
            dim=-1,
        )
        latent_noise = self.latent_proj(edge_out)  # [B, latent_dim]
        
        return anchor_noise, latent_noise


class DiffusionTransformerV4(nn.Module):
    def __init__(self, d_in: int, d_feat: int, d_out: int, n_blocks: int, num_heads: int = 8, head_dim: Optional[int] = None) -> None:
        super().__init__()
        self.t_embedding = nn.Sequential(
            SinusoidalPositionEmbeddings(d_feat), 
            nn.SiLU(),
            nn.Linear(d_feat, d_feat)
        )
        self.object_embedding = Embedder(d_in=22, d_feat=d_feat, num_heads=num_heads, head_dim=head_dim)
        self.template_embedding = Embedder(d_in=22, d_feat=d_feat, num_heads=num_heads, head_dim=head_dim)
        self.hand_up = nn.Sequential(
            nn.Linear(22, d_feat // 2),
            nn.SiLU(),
            nn.Linear(d_feat // 2, d_feat // 2)
        )
        self.cmap_up = nn.Sequential(
            nn.Linear(4, d_feat // 2),
            nn.SiLU(),
            nn.Linear(d_feat // 2, d_feat // 2)
        )
        self.blocks = nn.ModuleList([
            DiTBlockV3Fixed(d_feat, num_heads=num_heads, head_dim=head_dim) for _ in range(n_blocks)
        ])
        self.proj_to_mask = nn.Sequential(
            nn.Linear(d_feat, d_feat),
            nn.SiLU(),
            nn.Linear(d_feat, MASK_DIM)
        )
        self.proj_to_sh = nn.Sequential(
            nn.Linear(d_feat, d_feat),
            nn.SiLU(),
            nn.Linear(d_feat, SH_DIM)
        )
        self.proj_to_ortho = nn.Sequential(
            nn.Linear(d_feat, d_feat),
            nn.SiLU(),
            nn.Linear(d_feat, ORTHO_DIM)
        )
        self.proj_to_object_cmap = nn.Sequential(
            nn.Linear(d_feat, d_feat),
            nn.SiLU(),
            nn.Linear(d_feat, CMAP_DIM)
        )
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor]:
        """
        Input x: (B, N, d_in)
              t: (B,)
                d_in = 22(hand) + 4(cmap) + 22(object) + 3(pos) + 22(template) = 73

        Output: (B, N, MASK_DIM), (B, N, SH_DIM), (B, N, ORTHO_DIM), (B, N, CMAP_DIM)
        """
        hand = x[:, :, :22]
        cmap = x[:, :, 22:26]
        object = x[:, :, 26:48]
        pos = x[:, :, 48:51]
        template = x[:, :, 51:73]

        t_emb = self.t_embedding(t)
        t_emb = t_emb.unsqueeze(1).expand(-1, pos.size(1), -1)
        object_emb = self.object_embedding(object, pos)
        tem_emb = self.template_embedding(template, pos)
        hand_emb = self.hand_up(hand)
        cmap_emb = self.cmap_up(cmap)

        feat = torch.cat([hand_emb, cmap_emb], dim=-1)

        for i, block in enumerate(self.blocks):
            feat = block(feat, t_emb, pos, object_emb, tem_emb)
        
        mask = self.proj_to_mask(feat)
        sh = self.proj_to_sh(feat)
        ortho = self.proj_to_ortho(feat)
        object_cmap = self.proj_to_object_cmap(feat)
        
        return mask, sh, ortho, object_cmap
