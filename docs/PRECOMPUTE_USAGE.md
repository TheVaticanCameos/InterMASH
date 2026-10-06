# Pre-computed Object Point Clouds and KNN Usage Guide

## Overview

Pre-compute object point clouds, normals, and KNN indices to accelerate training with graph-aware models.

## Step 1: Pre-compute Object Data and KNN

Run the pre-computation script before training:

```bash
python mash_hoi/Preprocess/precompute_object_pointclouds.py \
    --data_root /path/to/your/dataset/root \
    --template_mash_path /path/to/template.npy
```

**Required arguments:**
- `--data_root`: Root directory of the dataset
- `--template_mash_path`: Path to shadow hand template MASH file

**Optional arguments:**
- `--sample_phi_num 32` (default: 32)
- `--sample_theta_num 8` (default: 8)
- `--device cuda` (default: cuda)

**Output files:**
- `{data_root}/object_pointclouds.npz` - Structured point clouds [N, S, 3] and normals for all objects
- `{data_root}/object_knn.npz` - KNN indices [N, K] for each object
- `{data_root}/template_hand_knn.npz` - KNN indices [N, K] for template hand

## Step 2: Training

### Option A: Standard DGN Dataset (without graph structure)

```python
from mash_hoi.source.Module.DataModule.data_dgn_sphere import SphereDGNDataModule

data_module = SphereDGNDataModule(
    data_root="/path/to/your/dataset/root",
    template_mash_path="/path/to/template.npy",
    batch_size=32,
    num_workers=4
)
```

### Option B: Graph-Aware DGN Dataset (with KNN)

For graph-aware models that use KNN-based attention:

```python
from mash_hoi.source.Module.DataModule.data_dgn_graph import GraphDGNDataModule

data_module = GraphDGNDataModule(
    data_root="/path/to/your/dataset/root",
    template_mash_path="/path/to/template.npy",
    batch_size=32,
    num_workers=4,
    k_neighbors=8  # Number of nearest neighbors
)
```

The dataset returns:
- `positions`: [N, 3] anchor positions
- `hand_params`: [N, 22] hand parameters
- `object_params`: [N, 22] object parameters
- `cmap_params`: [N, 4] contact map parameters
- `template_hand_params`: [N, 22] template hand
- `hand_knn`: [N, K] hand KNN indices
- `obj_knn`: [N, K] object KNN indices
- `data_name`: str
- `obj_points`: [N, S, 3] structured object point cloud
- `obj_normals`: [N, S, 3] structured object normals

## Troubleshooting

**Error: Pre-computed data not found**
- Run the pre-computation script before training (see Step 1)

## File Structure

```
dataset_root/
├── train_data.h5
├── test_data.h5
├── statistics.npy
├── object_pointclouds.npz       # Structured point clouds [N, S, 3]
├── object_knn.npz               # Object KNN indices [N, K]
└── template_hand_knn.npz        # Template hand KNN indices [N, K]
```

## Notes

- **Structured Format**: Point clouds are stored as [N, S, 3] where N is the number of patches and S is the samples per patch
- **KNN Computation**: KNN indices are computed based on patch centers, enabling efficient graph-aware attention
- **Caching**: The dataset uses lazy loading and in-memory caching for efficient data access
- **Compatibility**: `data_dgn_sphere.py` continues to work without KNN data, while `data_dgn_graph.py` requires all three files
