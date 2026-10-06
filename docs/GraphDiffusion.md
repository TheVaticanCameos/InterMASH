# Graph-Aware Diffusion with Edge Latent Token

本文档介绍 Graph-Aware Diffusion 模型的设计思路、实现细节以及训练/测试方法。

## 1. 动机 (Motivation)

### 1.1 问题背景

当前的 diffusion 模型将 256 个 anchor 点独立处理，没有考虑：
1. **空间邻近关系**：相邻的 patch 之间应该有更强的相关性
2. **手-物体交互关系**：Edge Weight 矩阵 (256×256) 描述了手的每个 patch 与物体每个 patch 的交互强度

### 1.2 解决方案

我们提出三个关键改进：

1. **K-Neighbor Graph Attention**：在 self-attention 中加入 KNN bias，让每个 patch 更关注其空间上的 K 个最近邻
   - **手部图注意力**：手部 patch 之间的空间邻近关系
   - **物体图编码**：物体 patch 之间的空间邻近关系
2. **Edge Latent Token**：添加一个可学习的 `edge_token`，通过 attention 机制聚合全局信息，生成 Edge Weight VAE 的 latent code (64维)

## 2. 架构设计 (Architecture)

### 2.1 整体结构

```
输入特征处理:
├── object_embedding (GraphEmbedder + obj_knn)    # 物体特征图编码
│   └── 每个 object patch 融合相邻 object patches 信息
├── template_embedding (Embedder)                  # 模板特征编码
└── hand_up + cmap_up                             # 手部特征投影

         ↓

构建输入序列 [B, N+1, D]:
├── edge_token [B, 1, D]          # 可学习的全局 token
│   ├── 初始化为 learnable parameter
│   └── 如果有 noisy_latent，加入其嵌入
└── anchor_feat [B, 256, D]       # 手部特征 (hand_emb + cmap_emb)

         ↓ 通过多层 DiT Blocks

DiT Blocks (×n) with Graph Attention:
├── graph_self_attn (hand_knn)                    # 手部图自注意力
│   ├── edge_token 自由 attend 到所有 anchor tokens
│   ├── anchor tokens attend 到其 K 近邻（KNN bias）
│   └── anchor tokens 也可以 attend 到 edge_token
├── cross_attn_tem                                # anchor → 模板交叉注意力
│   └── edge_token 不参与（仅 MLP）
├── cross_attn_obj                                # anchor → 物体交叉注意力
│   ├── 物体特征已经是图感知的（通过 GraphEmbedder）
│   └── edge_token 不参与（仅 MLP）
└── mlp                                           # Feed-forward

         ↓ 分离 edge_token 和 anchor tokens

Output:
├── Per-Anchor Noise [B, 256, 26]  # 从 anchor tokens 预测
│   └── anchor_proj(anchor_out)
└── Edge Latent Noise [B, 64]      # 从 edge_token 预测
    └── latent_proj(edge_out)
```

**关键点**：
- `edge_token` 位于序列开头（位置 0），通过 self-attention 聚合全局信息
- 在 cross-attention 阶段，`edge_token` 只经过 MLP，不参与手→物体/模板的交互
- 最终 `edge_token` 的输出被投影为 Edge Weight latent 的 noise 预测

### 2.2 Graph Attention 机制

模型在两个位置使用图注意力机制：

#### 2.2.1 物体特征编码（GraphEmbedder）

在物体特征编码阶段，使用 `GraphEmbedder` 对物体 patch 进行图感知编码：

```python
# GraphEmbedder 使用 GraphSelfAttention
object_emb = GraphEmbedder(obj_params, pos, knn_indices=obj_knn_indices)
```

**作用**：
- 每个 object patch 在编码时融合其 K 个最近邻 object patches 的信息
- 增强物体特征的局部一致性
- 为后续的 cross-attention 提供更好的物体表示

#### 2.2.2 手部图自注意力（DiT Blocks）

在 DiT blocks 中，手部特征通过 `GraphSelfAttention` 进行图感知处理：

```python
# 标准 attention: softmax(QK^T / sqrt(d)) V
# Graph attention: softmax(QK^T / sqrt(d) + bias) V

# bias[i,j] = scale if j in hand_KNN(i) else 0
```

**作用**：
- 每个 hand patch 更关注其空间上的 K 个最近邻 hand patches
- 保持手部结构的空间局部性
- 同时仍然可以 attend 到所有其他 hand patches（通过 attention 机制）

#### 2.2.3 KNN 使用总结

| KNN 类型 | 使用位置 | 作用 |
|---------|---------|------|
| `hand_knn_indices` | DiT blocks 的 `graph_self_attn` | 手部 patch 内部图约束 |
| `obj_knn_indices` | `object_embedding` (GraphEmbedder) | 物体 patch 内部图约束 |

这种设计使得：
- **物体特征**在编码阶段就具有图结构感知
- **手部特征**在处理过程中通过图注意力保持空间关系
- 两者在 cross-attention 中自然融合

### 2.3 Edge Latent Token（核心设计）

#### 2.3.1 什么是 Edge Latent Token？

`edge_token` 是一个**可学习的参数** `[1, 1, d_feat]`，在模型初始化时随机初始化：

```python
self.edge_token = nn.Parameter(torch.randn(1, 1, d_feat) * 0.02)
```

它被添加到所有 anchor tokens 的前面，形成 `[B, N+1, d_feat]` 的输入序列：
- 位置 0：`edge_token` [B, 1, d_feat]
- 位置 1 到 N+1：anchor tokens [B, 256, d_feat]

#### 2.3.2 为什么可以用 Learnable Token？

这是一个关键的设计选择，基于以下原理：

**1. Attention 机制的全局聚合能力**

在 Transformer 的 self-attention 中，每个 token 都可以 attend 到序列中的所有其他 token。对于 `edge_token`：
- **作为 Query**：`edge_token` 可以 attend 到所有 256 个 anchor tokens，聚合全局信息
- **作为 Key/Value**：所有 anchor tokens 也可以 attend 到 `edge_token`，向其传递信息

这种双向的信息流使得 `edge_token` 能够：
- 从所有 anchor 中收集关于手-物体交互的全局模式
- 学习如何最优地聚合这些信息（而不是使用固定的聚合方式，如平均池化）

**2. 可学习性带来的灵活性**

与固定聚合方式（如全局平均池化）不同，learnable token 允许模型：
- **学习最优的聚合策略**：模型可以学习哪些 anchor 更重要，如何加权组合它们
- **适应不同的交互模式**：不同的抓取姿态需要不同的聚合方式，learnable token 可以适应这种变化
- **端到端优化**：通过反向传播，token 的参数会朝着最小化 loss 的方向更新

**3. 与 Diffusion 过程的结合**

在 diffusion 训练中：
- **训练阶段**：`edge_token` 学习预测 Edge Weight latent 的噪声
- **推理阶段**：`edge_token` 生成合理的 Edge Weight latent，指导全局交互模式

通过这种方式，`edge_token` 不仅聚合信息，还学习生成符合物理约束的全局交互表示。

#### 2.3.3 Attention 机制详解

在 `GraphSelfAttention` 中，attention bias 矩阵的设计如下：

```python
# Attention bias matrix [B, N+1, N+1]
bias = [
    [0,    0,    0,    ..., 0   ]  # edge_token row: 可以 attend 到所有 tokens
    [0,    KNN,  KNN,  ..., KNN]  # anchor 1: 对 K 个邻居有 bias
    [0,    KNN,  KNN,  ..., KNN]  # anchor 2: 对 K 个邻居有 bias
    ...
]
```

关键点：
- **edge_token (row 0)**：没有 KNN bias，可以自由 attend 到所有 anchor tokens
- **edge_token (col 0)**：所有 anchor tokens 都可以自由 attend 到 edge_token
- **Anchor tokens**：对空间上的 K 个最近邻有额外的 attention bias

这样设计的好处：
1. `edge_token` 不受空间约束，可以聚合全局信息
2. Anchor tokens 仍然保持空间局部性（通过 KNN bias）
3. 两者通过 attention 机制自然融合

#### 2.3.4 与 Noisy Latent 的结合

在 forward 过程中，如果提供了 `noisy_latent`（来自 diffusion 过程）：

```python
# 1. 扩展 learnable edge_token
edge_tokens = self.edge_token.expand(B, -1, -1)  # [B, 1, d_feat]

# 2. 如果有 noisy_latent，将其嵌入并加到 edge_token 上
if noisy_latent is not None:
    latent_emb = self.latent_up(noisy_latent)  # [B, d_feat]
    edge_tokens = edge_tokens + latent_emb.unsqueeze(1)  # [B, 1, d_feat]
```

这样设计的原因：
- **Learnable token 提供基础表示**：学习如何聚合信息
- **Noisy latent 提供当前状态**：来自 diffusion 过程的当前 latent 状态
- **两者相加**：结合了可学习的聚合策略和当前的 diffusion 状态

#### 2.3.5 输出投影

经过所有 DiT blocks 后，`edge_token` 的输出被投影到 latent 维度：

```python
edge_out = feat[:, 0, :]  # [B, d_feat] - edge_token 的输出
latent_noise = self.latent_proj(edge_out)  # [B, 64] - 预测的 latent noise
```

这个预测的 noise 用于：
- **训练**：与真实的 latent noise 计算 MSE loss
- **推理**：在 diffusion 的去噪过程中逐步生成 Edge Weight latent

#### 2.3.6 与 Edge VAE 的配合

Edge Latent Token 生成的 latent code 与 Edge Weight VAE 配合使用：

1. **训练时**：
   - Edge VAE encoder 将 GT 的 hand/object params 编码为 latent
   - Diffusion 模型学习预测这个 latent 的 noise
   - `edge_token` 学习如何从 anchor features 中提取信息来预测 latent noise

2. **推理时**：
   - Diffusion 模型从随机噪声开始，逐步去噪生成 latent
   - `edge_token` 在每个 timestep 聚合当前 anchor 状态，预测下一步的 noise
   - 最终生成的 latent 可以用 Edge VAE decoder 解码为完整的 Edge Weight 矩阵

#### 2.3.7 设计优势总结

使用 learnable `edge_token` 相比固定聚合方式的优势：

| 特性 | Learnable Token | 固定聚合（如平均池化） |
|------|----------------|---------------------|
| **灵活性** | ✅ 可学习最优聚合策略 | ❌ 固定权重 |
| **适应性** | ✅ 适应不同交互模式 | ❌ 无法适应 |
| **端到端** | ✅ 与 diffusion 联合优化 | ❌ 分离优化 |
| **表达能力** | ✅ 强大的全局表示 | ⚠️ 表达能力有限 |

这种设计使得模型能够学习生成符合物理约束的全局交互模式，同时保持与局部 anchor 特征的紧密联系。

### 2.4 KNN 计算

模型需要两种 KNN indices：

#### 2.4.1 Hand KNN

用于手部 patch 的图自注意力：

1. 从模板手部 MASH 参数采样点云 (8×8=64 点/patch)
2. 计算每个手部 patch 的中心点
3. 基于中心点距离计算 K=8 最近邻
4. 预计算并缓存为 `hand_knn` [B, N, K]

**注意**：使用模板手部参数（而非 GT 手部参数）计算，确保训练和推理时的一致性。

#### 2.4.2 Object KNN

用于物体特征的图编码：

1. 从物体 MASH 参数采样点云 (8×8=64 点/patch)
2. 计算每个物体 patch 的中心点
3. 基于中心点距离计算 K=8 最近邻
4. 预计算并缓存为 `obj_knn` [B, N, K]

#### 2.4.3 KNN Cache

两种 KNN indices 都预计算并缓存在 H5 文件中：

```
.cache/knn_cache/
├── shadow_knn_cache.h5
│   ├── hand_knn [N, K]    # 手部 KNN
│   └── obj_knn [N, K]     # 物体 KNN
└── ...
```

首次运行时会自动计算，后续直接加载以加速训练。

## 3. 文件结构

```
mash_hoi/source/
├── Module/
│   ├── NetworkModule/
│   │   ├── network_general.py     # GraphSelfAttention, GraphCrossAttention
│   │   └── network_dit.py         # GraphEmbedder, DiTBlockV3Graph, DiffusionTransformerV3Graph
│   └── DataModule/
│       └── data_dro_graph.py      # 带 KNN 缓存的数据集（hand_knn + obj_knn）
├── Model/
│   └── graph_diffusion_multihand.py  # Graph Diffusion 模型
├── Config/DRO/
│   └── ddpm_graph_dro.yaml        # 配置文件
└── Demo/ExpDRO-graphdiff/
    └── train.py                   # 训练脚本
```

**关键组件**：
- `GraphSelfAttention`: 图感知自注意力（用于手部和物体）
- `GraphEmbedder`: 图感知特征编码器（用于物体特征）
- `DiTBlockV3Graph`: 图感知 DiT block（使用 hand_knn）
- `DiffusionTransformerV3Graph`: 完整的图感知 diffusion transformer

## 4. 训练 (Training)

### 4.1 数据准备

确保数据集目录结构如下：
```
/path/to/Sphere-CMapDataset/
├── shadow_train_data.h5
├── shadow_statistics.npy
├── allegro_train_data.h5
├── allegro_statistics.npy
├── barrett_train_data.h5
├── barrett_statistics.npy
└── .cache/
    └── knn_cache/
        ├── shadow_knn_cache.h5    # 自动生成
        ├── allegro_knn_cache.h5
        └── barrett_knn_cache.h5
```

### 4.2 配置文件

编辑 `mash_hoi/source/Config/DRO/ddpm_graph_dro.yaml`：

```yaml
model_cfg:
  name: dit_v3_graph
  d_in: 26           # 每个 anchor: 22 (hand) + 4 (cmap)
  d_feat: 192        # 特征维度
  d_out: 26
  n_blocks: 8
  latent_dim: 64     # Edge Weight latent 维度
  k_neighbors: 8     # K 近邻数量

edge_vae_cfg:
  checkpoint_path: null  # Edge Weight VAE 训练完成后设置
  latent_dim: 64
  row_hidden_dims: [128, 64]       # Stage 1: Column classifier
  col_hidden_dims: [256, 128, 64]  # Stage 2: Row predictor

loss_cfg:
  lambda_latent: 1.0     # Edge Weight latent loss 权重
  lambda_collision: 1.0  # 碰撞 loss 权重
```

### 4.3 训练命令

```bash
cd mash_hoi/source/Demo/ExpDRO-graphdiff

# 单 GPU 训练
python train.py trainer.devices=[0]

# 多 GPU 训练
python train.py trainer.devices=[0,1,2,3]

# 快速调试
python train.py trainer.fast_dev_run=True
```

### 4.4 训练流程

1. **首次运行**：自动预计算两种 KNN cache（`hand_knn` 和 `obj_knn`，约需几分钟）
   - `hand_knn`: 基于模板手部参数计算，用于 graph self-attention
   - `obj_knn`: 基于物体参数计算，用于 object embedding 的图编码
2. **Diffusion 训练**：
   - 对每个 anchor 的 `[hand_params, cmap_params]` 进行 diffusion
   - 同时对全局 `edge_weight_latent` 进行 diffusion
   - 使用 graph attention 增强空间关系建模：
     - 物体特征通过 `GraphEmbedder` 进行图感知编码
     - 手部特征通过 `GraphSelfAttention` 保持空间局部性

### 4.5 监控

使用 TensorBoard 监控训练：

```bash
tensorboard --logdir=mash_hoi/runs
```

关键指标：
- `recon_loss`: 重建损失
- `latent_loss`: Edge Weight latent 损失（需 Edge VAE）
- `collision_loss`: 碰撞损失

## 5. 测试 (Testing)

### 5.1 测试命令

```bash
python train.py \
    trainer.devices=[0] \
    mode=test \
    tester.ckpt_path=/path/to/checkpoint.ckpt
```

### 5.2 生成结果

测试结果保存在 `diffusion_cfg.output_dir` 指定的目录：

```
output_dir/
├── gen_sample_0.npy
├── gen_sample_1.npy
└── ...
```

每个 `.npy` 文件包含：
- `positions`: anchor 位置
- `hand_params`: 生成的手部 MASH 参数
- `object_params`: 物体 MASH 参数
- `cmap_params`: contact map 参数
- `edge_weight_latent`: Edge Weight latent code（可用 Edge VAE decoder 解码）

### 5.3 边权重矩阵重建

当 Edge Weight VAE 可用时，可以从 latent 重建完整边权重矩阵：

```python
from mash_hoi.source.Model.vae_edge import EdgeVAE

# 加载 VAE
vae = EdgeVAE.load_from_checkpoint(vae_ckpt_path)

# 重建边权重矩阵（需要 positions 参数）
edge_weight_matrix = vae.efficient_reconstruct_full_matrix(
    hand_params=generated_hand_params,
    object_params=object_params,
    positions=positions,  # [B, N, 3] 必需参数
    n_anchors=256,
    nonzero_threshold=0.1
)  # [B, 256, 256]
```

注意：`positions` 参数是必需的，用于 RoPE 位置编码。

## 6. 技术细节

### 6.1 GraphEmbedder 实现

`GraphEmbedder` 用于物体特征的图感知编码：

```python
class GraphEmbedder(nn.Module):
    def __init__(self, d_in, d_feat, num_heads, knn_bias_scale=2.0):
        self.graph_self_attn = GraphSelfAttention(
            d_feat, num_heads=num_heads, knn_bias_scale=knn_bias_scale
        )
    
    def forward(self, x, pos, knn_indices=None):
        # x: [B, N, 22] object params
        # pos: [B, N, 3] positions
        # knn_indices: [B, N, K] object KNN indices
        x = self.graph_self_attn(x, pos, knn_indices=knn_indices, has_edge_token=False) + x
        return x
```

**关键点**：
- 使用 `GraphSelfAttention` 而非普通 `SelfAttention`
- 通过 `obj_knn_indices` 让每个 object patch 更关注其空间邻居
- 在特征编码阶段就融入图结构信息

### 6.2 Edge Latent Token 的实现

在 `DiffusionTransformerV3Graph` 中：

```python
# 初始化
self.edge_token = nn.Parameter(torch.randn(1, 1, d_feat) * 0.02)

# Forward 中
edge_tokens = self.edge_token.expand(B, -1, -1)  # [B, 1, d_feat]
if noisy_latent is not None:
    latent_emb = self.latent_up(noisy_latent)  # [B, d_feat]
    edge_tokens = edge_tokens + latent_emb.unsqueeze(1)
feat = torch.cat([edge_tokens, anchor_feat], dim=1)  # [B, N+1, d_feat]
```

### 6.3 Graph Attention 中的处理

在 `GraphSelfAttention` 中，`edge_token` 的 attention bias 为 0，意味着：
- `edge_token` 可以 attend 到所有 anchor tokens（无空间约束）
- 所有 anchor tokens 可以 attend 到 `edge_token`（传递全局信息）

对于 anchor tokens，attention bias 矩阵为：
```python
# bias[i,j] = knn_bias_scale if j in KNN(i) else 0
```

这样设计使得：
- 每个 patch 对空间邻居有更强的注意力
- 同时保持全局 attention 的能力（不限制只能 attend 到邻居）

### 6.3 训练损失

总损失包含三部分：

```python
loss = recon_loss + lambda_latent * latent_loss + lambda_collision * collision_loss
```

- `recon_loss`：Anchor features 的重建损失
- `latent_loss`：Edge Weight latent 的重建损失（需要 Edge VAE）
- `collision_loss`：碰撞约束损失

## 7. 注意事项

1. **Edge Weight VAE 依赖**：Edge Weight latent diffusion 需要预训练的 Edge VAE。如果 VAE 未训练，设置 `edge_vae_cfg.checkpoint_path: null`，模型会跳过 latent loss。

2. **Positions 参数**：Edge VAE 的所有方法（`encode`, `forward`, `efficient_reconstruct_full_matrix`）都需要 `positions` 参数用于 RoPE 位置编码。

3. **KNN Cache**：
   - 首次运行会自动计算两种 KNN cache（`hand_knn` 和 `obj_knn`）
   - 使用模板手部参数计算 `hand_knn`，确保训练和推理一致性
   - 如需重新计算，删除 `.cache/knn_cache/` 目录

4. **显存占用**：Graph attention 的 bias 矩阵会增加少量显存。如遇显存不足，可减小 `batch_size` 或 `k_neighbors`。

5. **混合精度**：推荐使用 `precision: "16-mixed"` 加速训练。注意某些操作（如 `torch.linalg.eigh`）需要 float32。

6. **Learnable Token 初始化**：`edge_token` 使用小方差初始化（`* 0.02`），确保训练初期的稳定性。

7. **KNN 使用位置**：
   - `hand_knn_indices`: 用于 DiT blocks 中的 graph self-attention
   - `obj_knn_indices`: 用于 `object_embedding` (GraphEmbedder) 的图编码
   - 两者都是描述**内部**连接关系（手部内部、物体内部），而非跨实体关系

