from typing import List, Optional
import torch
import torch.nn as nn
import math
from liger_kernel.ops.geglu import LigerGELUMulFunction


def sinusoidal_frequencies(dim: int) -> torch.Tensor:
    max_freq = math.log(dim // 2 - 1, 2)
    freqs = torch.pi * torch.exp2(torch.linspace(0, max_freq, dim // 2, dtype=torch.float32))
    return freqs


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    q_embed = (q * cos) + (rotate_half(q) * sin)
    return q_embed


def roundup_16(x: int) -> int:
    return int((x + 15) // 16 * 16)


class RoPE(nn.Module):
    def __init__(
        self,
        phys_dim: int = 3,
        d_model: int = 384
    ) -> None:
        super().__init__()
        self.m = phys_dim
        self.d_model = d_model
        assert d_model % (phys_dim * 2) == 0, f"d_model {d_model} must be divisible by 2*phys_dim={2*phys_dim}"
        dim_per_phys = d_model // phys_dim
        self.register_buffer("freqs", sinusoidal_frequencies(dim_per_phys))
    
    @torch.autocast(device_type='cuda', enabled=False)  # Disable autocast for precision in RoPE_M
    def forward(self, fx: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the RoPE layer to the input tensor.

        Args:
            fx (torch.Tensor): Functional values at the input points, shape (batch_size, heads, seq_len, d_model).
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, m).

        Returns:
            torch.Tensor: Output tensor with rotary embeddings applied.
        """
        assert x.shape[-1] == self.m, f"Input tensor x must have last dimension {self.m}, but got {x.shape[-1]}"
        assert fx.shape[-1] == self.d_model, (
            f"Functional tensor fx must have last dimension {self.d_model}, but got {fx.shape[-1]}"
        )
        batch_size, seq_len, _ = x.shape
        freqs = self.freqs * x.unsqueeze(-1)  # (d_model // (m*2), ) * (batch_size, seq_len, m, 1) ==> (b, s, m, d_model // (m*2))
        freqs = freqs.view(batch_size, 1, seq_len, -1) # (b, 1, s, d_model // (m*2))
        freqs = torch.cat([freqs, freqs], dim=-1) # (b, 1, s, d_model // 2 * 2) ==> (b, 1, s, d_model)
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
        out = apply_rotary_pos_emb(fx, cos, sin)
        return out.contiguous()


class SelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, phys_dim: int = 3, head_dim: Optional[int] = None) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = head_dim if head_dim is not None else d_model // num_heads
        self.phys_dim = phys_dim

        self.qkv = nn.Linear(d_model, 3 * (self.head_dim * num_heads), bias=False)
        self.out_proj = nn.Linear(self.head_dim * num_heads, d_model)
        self.rope = RoPE(phys_dim=phys_dim, d_model=self.head_dim)

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, d_model) Input features
            pos: (B, N, phys_dim) 3D position coordinates
        Returns:
            out: (B, N, d_model)
        """
        
        B, N, _ = x.shape
        qkv = self.qkv(x)  # (B, N, 3*d_model)
        q, k, v = qkv.chunk(3, dim=-1)  # Each has shape (B, N, d_model)

        # Split into multiple heads (B, N, num_heads, head_dim) -> (B, num_heads, N, head_dim)
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE rotary positional embeddings
        q = self.rope(q, pos)  # (B, num_heads, N, head_dim)
        k = self.rope(k, pos)

        # # Compute attention weights
        # scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, num_heads, N, N)
        # attn = torch.softmax(scores, dim=-1)

        # # Compute the weighted sum
        # out = torch.matmul(attn, v)  # (B, num_heads, N, head_dim)
        # out = out.transpose(1, 2).contiguous().view(B, N, self.d_model)  # Merge attention heads
        out = nn.functional.scaled_dot_product_attention(q, k, v, None)
        out = self.out_proj(out.transpose(1, 2).reshape(B, N, -1))
        return out


class GraphSelfAttention(nn.Module):
    """
    Self-attention with KNN graph bias for graph-aware processing.
    
    Supports an optional edge_token at position 0 that can attend to all tokens freely.
    Anchor tokens (positions 1:N+1) have attention bias toward their K nearest neighbors.
    """
    def __init__(
        self, 
        d_model: int, 
        num_heads: int, 
        phys_dim: int = 3, 
        head_dim: Optional[int] = None,
        knn_bias_scale: float = 2.0,
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = head_dim if head_dim is not None else d_model // num_heads
        self.phys_dim = phys_dim
        self.knn_bias_scale = knn_bias_scale

        self.qkv = nn.Linear(d_model, 3 * (self.head_dim * num_heads), bias=False)
        self.out_proj = nn.Linear(self.head_dim * num_heads, d_model)
        self.rope = RoPE(phys_dim=phys_dim, d_model=self.head_dim)

    def build_knn_bias(
        self, 
        knn_indices: torch.Tensor, 
        seq_len: int,
        has_edge_token: bool = True,
    ) -> torch.Tensor:
        """
        Build attention bias matrix from KNN indices.
        
        Args:
            knn_indices: [B, N, K] KNN indices for anchor tokens
            seq_len: Total sequence length (N+1 if has_edge_token else N)
            has_edge_token: Whether edge_token is at position 0
            
        Returns:
            bias: [B, seq_len, seq_len] attention bias (additive)
        """
        B, N, K = knn_indices.shape
        device = knn_indices.device
        dtype = torch.float32
        
        # Create bias matrix for anchor tokens [B, N, N]
        anchor_bias = torch.zeros(B, N, N, device=device, dtype=dtype)
        
        # Expand indices for scatter: [B, N, K]
        # For each anchor i, set bias[i, knn_indices[i, :]] = scale
        batch_idx = torch.arange(B, device=device).view(B, 1, 1).expand(-1, N, K)
        row_idx = torch.arange(N, device=device).view(1, N, 1).expand(B, -1, K)
        anchor_bias[batch_idx, row_idx, knn_indices] = self.knn_bias_scale
        
        if has_edge_token:
            # Pad for edge_token: [B, N+1, N+1]
            # edge_token (row 0) has no bias (attends freely)
            # edge_token (col 0) has no bias (all tokens can attend to it freely)
            bias = torch.zeros(B, seq_len, seq_len, device=device, dtype=dtype)
            bias[:, 1:, 1:] = anchor_bias
        else:
            bias = anchor_bias
            
        return bias

    def forward(
        self, 
        x: torch.Tensor, 
        pos: torch.Tensor,
        knn_indices: Optional[torch.Tensor] = None,
        has_edge_token: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, N+1, d_model) input features, with edge_token at position 0
            pos: (B, N, phys_dim) 3D positions for anchor tokens only
            knn_indices: (B, N, K) KNN indices for anchor tokens, or None
            has_edge_token: Whether x includes edge_token at position 0
            
        Returns:
            out: (B, N+1, d_model)
        """
        B, seq_len, _ = x.shape
        N = pos.shape[1]  # Number of anchor tokens
        
        qkv = self.qkv(x)  # (B, seq_len, 3*head_dim*num_heads)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape to multi-head: (B, seq_len, num_heads, head_dim) -> (B, num_heads, seq_len, head_dim)
        q = q.view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE to anchor tokens only
        if has_edge_token:
            # edge_token at position 0 doesn't have position, use zero position
            edge_pos = torch.zeros(B, 1, self.phys_dim, device=pos.device, dtype=pos.dtype)
            full_pos = torch.cat([edge_pos, pos], dim=1)  # [B, N+1, 3]
        else:
            full_pos = pos
            
        q = self.rope(q, full_pos)
        k = self.rope(k, full_pos)

        # Build KNN attention bias
        attn_bias = None
        if knn_indices is not None:
            attn_bias = self.build_knn_bias(knn_indices, seq_len, has_edge_token)
            # Expand for num_heads: [B, 1, seq_len, seq_len]
            attn_bias = attn_bias.unsqueeze(1)

        # Scaled dot-product attention with bias
        out = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
        out = self.out_proj(out.transpose(1, 2).reshape(B, seq_len, -1))
        return out


class CrossAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, phys_dim: int = 3, head_dim: Optional[int] = None) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = head_dim if head_dim is not None else d_model // num_heads
        self.phys_dim = phys_dim

        self.q = nn.Linear(d_model, (self.num_heads * self.head_dim), bias=False)
        self.kv = nn.Linear(d_model, (2 * self.num_heads * self.head_dim), bias=False)
        self.out_proj = nn.Linear((self.num_heads * self.head_dim), d_model)
        self.rope = RoPE(phys_dim=phys_dim, d_model=self.head_dim)
    
    def forward(self, hand: torch.Tensor, object: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hand: (B, N, d_model) Hand features
            object: (B, N, d_model) Object features
            pos: (B, N, phys_dim) 3D position coordinates
        Returns:
            out: (B, N, d_model)
        """
        B, N, _ = hand.shape
        q = self.q(hand)  # (B, N, d_model)
        kv = self.kv(object)  # (B, N, 2*d_model)
        k, v = kv.chunk(2, dim=-1)  # Each has shape (B, N, d_model)

        # Split into multiple heads (B, N, num_heads, head_dim) -> (B, num_heads, N, head_dim)
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE rotary positional embeddings
        q = self.rope(q, pos)  # (B, num_heads, N, head_dim)
        k = self.rope(k, pos)

        # # Compute attention weights
        # scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, num_heads, N, N)
        # attn = torch.softmax(scores, dim=-1)

        # # Compute the weighted sum
        # out = torch.matmul(attn, v)  # (B, num_heads, N, head_dim)
        # out = out.transpose(1, 2).contiguous().view(B, N, self.d_model)  # Merge attention heads
        out = nn.functional.scaled_dot_product_attention(q, k, v, None)
        out = self.out_proj(out.transpose(1, 2).reshape(B, N, -1))
        return out


class GraphCrossAttention(nn.Module):
    """
    Cross-attention with KNN graph bias for graph-aware processing.
    
    Each query token attends more strongly to its K nearest key tokens (based on KNN indices).
    Supports an optional edge_token at query position 0 that can attend to all keys freely.
    """
    def __init__(
        self, 
        d_model: int, 
        num_heads: int, 
        phys_dim: int = 3, 
        head_dim: Optional[int] = None,
        knn_bias_scale: float = 2.0,
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = head_dim if head_dim is not None else d_model // num_heads
        self.phys_dim = phys_dim
        self.knn_bias_scale = knn_bias_scale

        self.q = nn.Linear(d_model, (self.num_heads * self.head_dim), bias=False)
        self.kv = nn.Linear(d_model, (2 * self.num_heads * self.head_dim), bias=False)
        self.out_proj = nn.Linear((self.num_heads * self.head_dim), d_model)
        self.rope = RoPE(phys_dim=phys_dim, d_model=self.head_dim)

    def build_knn_bias(
        self, 
        knn_indices: torch.Tensor, 
        query_len: int,
        key_len: int,
        has_edge_token: bool = True,
    ) -> torch.Tensor:
        """
        Build cross-attention bias matrix from KNN indices.
        
        Args:
            knn_indices: [B, N, K] KNN indices for anchor tokens (query → key neighbors)
            query_len: Query sequence length (N+1 if has_edge_token else N)
            key_len: Key sequence length (N, no edge_token in keys)
            has_edge_token: Whether edge_token is at query position 0
            
        Returns:
            bias: [B, query_len, key_len] attention bias (additive)
        """
        B, N, K = knn_indices.shape
        device = knn_indices.device
        dtype = torch.float32
        
        # Create bias matrix for anchor tokens [B, N, N]
        anchor_bias = torch.zeros(B, N, key_len, device=device, dtype=dtype)
        
        # For each anchor i, set bias[i, knn_indices[i, :]] = scale
        batch_idx = torch.arange(B, device=device).view(B, 1, 1).expand(-1, N, K)
        row_idx = torch.arange(N, device=device).view(1, N, 1).expand(B, -1, K)
        anchor_bias[batch_idx, row_idx, knn_indices] = self.knn_bias_scale
        
        if has_edge_token:
            # Pad for edge_token: [B, N+1, key_len]
            # edge_token (row 0) has no bias (attends freely to all keys)
            full_bias = torch.zeros(B, query_len, key_len, device=device, dtype=dtype)
            full_bias[:, 1:, :] = anchor_bias  # Anchor tokens start at position 1
            return full_bias
        else:
            return anchor_bias
    
    def forward(
        self, 
        query: torch.Tensor, 
        key_value: torch.Tensor, 
        pos: torch.Tensor,
        knn_indices: Optional[torch.Tensor] = None,
        has_edge_token: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            query: (B, Q, d_model) query features (Q = N+1 if has_edge_token else N)
            key_value: (B, N, d_model) key/value features (object features)
            pos: (B, N, phys_dim) 3D positions for keys
            knn_indices: (B, N, K) KNN indices for anchor tokens, or None
            has_edge_token: Whether query has edge_token at position 0
        Returns:
            out: (B, Q, d_model)
        """
        B, Q, _ = query.shape
        N = key_value.shape[1]
        
        q = self.q(query)  # (B, Q, d_model)
        kv = self.kv(key_value)  # (B, N, 2*d_model)
        k, v = kv.chunk(2, dim=-1)  # each (B, N, d_model)

        # Reshape to multi-head: (B, Q/N, num_heads, head_dim) -> (B, num_heads, Q/N, head_dim)
        q = q.view(B, Q, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE (only to anchor tokens in query if has_edge_token)
        if has_edge_token:
            # Query: edge_token at [0], anchors at [1:]
            # Create dummy position for edge_token
            edge_pos = pos[:, :1, :]  # Use first position as placeholder
            query_pos = torch.cat([edge_pos, pos], dim=1)  # [B, Q, 3]
            q = self.rope(q, query_pos)
        else:
            q = self.rope(q, pos)
        k = self.rope(k, pos)

        # Build attention bias if KNN indices provided
        attn_mask = None
        if knn_indices is not None:
            bias = self.build_knn_bias(knn_indices, Q, N, has_edge_token)
            # Expand for multi-head: [B, 1, Q, N]
            attn_mask = bias.unsqueeze(1)

        # Scaled dot-product attention with optional bias
        out = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = self.out_proj(out.transpose(1, 2).reshape(B, Q, -1))
        return out


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        assert self.dim % 2 == 0, "Sin P.E. dim must be even"
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Input t: (B)
        Output: (B, dim)
        """
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]          # (B, half_dim)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class GeGLU(nn.Module):
    def __init__(self, d_in, d_out, d_hidden = None, bias_down=False):
        super().__init__()
        # self.pre_ln = nn.LayerNorm(d_in)
        d_hidden = roundup_16((d_in + d_out) * (4/3)) if d_hidden is None else d_hidden
        self.up = nn.Linear(d_in, d_hidden, bias=False)
        self.gate = nn.Linear(d_in, d_hidden, bias=False)
        self.down = nn.Linear(d_hidden, d_out, bias=bias_down)
        self.impl = LigerGELUMulFunction()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        up = self.up(x)
        gate = self.gate(x)
        dtype = up.dtype
        device_type = up.device.type
        gate_c = gate.contiguous()
        up_c = up.contiguous()
        if device_type == "cuda":
            out = self.impl.apply(gate_c.float(), up_c.float()).to(dtype)
        else:
            out = (torch.nn.functional.gelu(gate_c) * up_c).to(dtype)
        return self.down(out)


class AdaLN(nn.Module):
    def __init__(self, d_model: int, chunk_num: int = 7, scale: int = 4):
        super().__init__()
        self.chunk_num = chunk_num
        self.mlp = GeGLU(d_model, d_model * chunk_num, d_model * scale, bias_down=True)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.mlp.down.weight)
        nn.init.zeros_(self.mlp.down.bias)

    def forward(self, cond: torch.Tensor) -> List[torch.Tensor]:
        params = self.mlp(cond).chunk(self.chunk_num, dim=-1)
        return params
