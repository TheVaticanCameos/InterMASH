#!/usr/bin/env python3
"""
Compute statistics and visualize dataset parameter distributions.

Usage example:
  python mash-hoi/analyze_dataset.py \
    --root_dir /data2/yxz/DATASET/GrabDataset/GRAB-HOI-Dataset/train \
    --output_dir mash-hoi/runs

The script creates a results directory under output_dir containing JSON statistics and PNG histograms.
"""

import os
import json
import argparse
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy.typing as npt

# Fixed bins: divide [-1, 1] into 20 equal intervals
BINS_EDGES: npt.NDArray[np.float64] = np.linspace(-1.0, 1.0, 21)


def _safe_concat(list_of_arrays):
    """Safely concatenate a list of arrays, allowing an empty list."""
    if len(list_of_arrays) == 0:
        return None
    return np.concatenate(list_of_arrays, axis=0)


def _compute_stats(x: np.ndarray, eps: float = 1e-8) -> Dict[str, float]:
    """Compute basic statistics. x: [N, D] or [N]"""
    if x is None:
        return {}
    # Flatten to one dimension for overall statistics; visualize individual dimensions separately
    xf = x.reshape(-1)
    return {
        "count": int(xf.shape[0]),
        "mean": float(np.nanmean(xf)),
        "std": float(np.nanstd(xf) + eps),
        "min": float(np.nanmin(xf)),
        "max": float(np.nanmax(xf)),
        "p01": float(np.nanpercentile(xf, 1)),
        "p05": float(np.nanpercentile(xf, 5)),
        "p50": float(np.nanpercentile(xf, 50)),
        "p95": float(np.nanpercentile(xf, 95)),
        "p99": float(np.nanpercentile(xf, 99)),
    }


def _plot_hist_per_dim(data: np.ndarray, title_prefix: str, save_path: str):
    """Plot a histogram for each dimension of data and save the figure.

    data: [N, D]
    title_prefix: Prefix for plot titles
    save_path: Output PNG path
    """
    if data is None:
        return
    if data.ndim == 1:
        data = data[:, None]

    N, D = data.shape
    cols = 4
    rows = int(np.ceil(D / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3), squeeze=False)
    fig.suptitle(f"{title_prefix} (N={N}, D={D}, normalized to [-1,1])", fontsize=14)

    for d in range(D):
        r, c = divmod(d, cols)
        ax = axes[r][c]
        dim_data = data[:, d]
        # Filter invalid values and use weights to express bin heights as percentages
        finite_mask = np.isfinite(dim_data)
        dim_data_valid = dim_data[finite_mask]
        if dim_data_valid.size > 0:
            weights = np.ones_like(dim_data_valid) * (100.0 / dim_data_valid.size)
            ax.hist(
                dim_data_valid,
                bins=BINS_EDGES,
                weights=weights,
                color="#4C78A8",
                alpha=0.85,
            )
        # Format the y-axis as percentages
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
        ax.set_ylabel("%")
        ax.set_title(f"{title_prefix}[{d}]")
        ax.grid(True, alpha=0.2)

    # Remove empty subplots
    for d in range(D, rows * cols):
        r, c = divmod(d, cols)
        axes[r][c].axis("off")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


def _normalize_minus1_1(x: np.ndarray):
    """Normalize each dimension to [-1, 1] and return (normalized array, dictionary of (min, max) values).

    Handle NaN/Inf by computing min/max and transforming only finite values; leave non-finite values unchanged.
    If min == max for a dimension, normalize that dimension to 0.
    """
    if x is None:
        return None, None
    if x.ndim == 1:
        x = x[:, None]
    x_norm = x.astype(np.float64, copy=True)
    N, D = x_norm.shape
    minmax = {"min": [], "max": []}

    for d in range(D):
        col = x_norm[:, d]
        mask = np.isfinite(col)
        if mask.any():
            cmin = np.nanmin(col[mask])
            cmax = np.nanmax(col[mask])
        else:
            # If all values are non-finite, set the range to zero and output zeros
            cmin, cmax = 0.0, 0.0
        minmax["min"].append(float(cmin))
        minmax["max"].append(float(cmax))

        if cmax == cmin:
            # Constant dimension -> map to 0
            x_norm[mask, d] = 0.0
        else:
            # Apply min-max scaling to [0,1], then map linearly to [-1,1]
            x_norm[mask, d] = (col[mask] - cmin) / (cmax - cmin) * 2.0 - 1.0

    return x_norm, minmax


def _bincount_per_dim(x_norm: np.ndarray, bins_edges: np.ndarray) -> Dict[str, list]:
    """Compute per-dimension counts and percentages using fixed bins_edges and return a JSON-compatible list."""
    if x_norm is None:
        return {"counts": [], "percent": []}
    if x_norm.ndim == 1:
        x_norm = x_norm[:, None]
    N, D = x_norm.shape
    counts = np.zeros((D, len(bins_edges) - 1), dtype=int)
    percent = np.zeros_like(counts, dtype=float)
    for d in range(D):
        col = x_norm[:, d]
        mask = np.isfinite(col)
        if mask.any():
            hist, _ = np.histogram(col[mask], bins=bins_edges)
        else:
            hist = np.zeros(len(bins_edges) - 1, dtype=int)
        counts[d] = hist
        total = hist.sum()
        percent[d] = (hist / total * 100.0) if total > 0 else np.zeros_like(hist, dtype=float)
    return {"counts": counts.tolist(), "percent": percent.tolist()}


def analyze(root_dir: str, out_base_dir: str) -> Tuple[str, Dict[str, Dict]]:
    """Iterate over the dataset, aggregate statistics, and save visualizations.

    Return (results directory, statistics dictionary)
    """
    # Results directory
    ts = datetime.now().strftime("%m%d_%H%M%S")
    out_dir = os.path.join(out_base_dir, f"analysis_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    # Collect all data
    hand_mask_list = []
    hand_sh_list = []
    hand_ortho_list = []
    obj_mask_list = []
    obj_sh_list = []
    obj_ortho_list = []
    pos_list = []

    data_paths = [
        os.path.join(root_dir, n)
        for n in sorted(os.listdir(root_dir))
        if n.endswith(".npy")
    ]

    if len(data_paths) == 0:
        raise FileNotFoundError(f"在目录 `{root_dir}` 未找到 .npy 数据文件")

    for p in data_paths:
        try:
            data = np.load(p, allow_pickle=True)[()]
        except Exception as e:
            print(f"跳过无法读取文件: {p} ({e})")
            continue

        # positions
        if "positions" in data:
            pos_list.append(np.asarray(data["positions"]))

        # hand
        hp = data.get("hand_params", {})
        if len(hp) > 0:
            if "mask_params" in hp:
                hand_mask_list.append(np.asarray(hp["mask_params"]))
            if "sh_params" in hp:
                hand_sh_list.append(np.asarray(hp["sh_params"]))
            if "ortho_poses" in hp:
                hand_ortho_list.append(np.asarray(hp["ortho_poses"]))

        # object
        op = data.get("object_params", {})
        if len(op) > 0:
            if "mask_params" in op:
                obj_mask_list.append(np.asarray(op["mask_params"]))
            if "sh_params" in op:
                obj_sh_list.append(np.asarray(op["sh_params"]))
            if "ortho_poses" in op:
                obj_ortho_list.append(np.asarray(op["ortho_poses"]))

    positions = _safe_concat(pos_list)
    hand_mask = _safe_concat(hand_mask_list)
    hand_sh = _safe_concat(hand_sh_list)
    hand_ortho = _safe_concat(hand_ortho_list)
    obj_mask = _safe_concat(obj_mask_list)
    obj_sh = _safe_concat(obj_sh_list)
    obj_ortho = _safe_concat(obj_ortho_list)

    # Normalize to [-1, 1]
    positions_n, positions_minmax = _normalize_minus1_1(positions)
    hand_mask_n, hand_mask_minmax = _normalize_minus1_1(hand_mask)
    hand_sh_n, hand_sh_minmax = _normalize_minus1_1(hand_sh)
    hand_ortho_n, hand_ortho_minmax = _normalize_minus1_1(hand_ortho)
    obj_mask_n, obj_mask_minmax = _normalize_minus1_1(obj_mask)
    obj_sh_n, obj_sh_minmax = _normalize_minus1_1(obj_sh)
    obj_ortho_n, obj_ortho_minmax = _normalize_minus1_1(obj_ortho)

    # Compute overall statistics
    stats: Dict[str, Dict] = {
        "positions": {
            "original": _compute_stats(positions),
            "minmax": positions_minmax,
            "normalized_stats": _compute_stats(positions_n),
            "binned": _bincount_per_dim(positions_n, BINS_EDGES),
        },
        "hand": {
            "mask_params": {
                "original": _compute_stats(hand_mask),
                "minmax": hand_mask_minmax,
                "normalized_stats": _compute_stats(hand_mask_n),
                "binned": _bincount_per_dim(hand_mask_n, BINS_EDGES),
            },
            "sh_params": {
                "original": _compute_stats(hand_sh),
                "minmax": hand_sh_minmax,
                "normalized_stats": _compute_stats(hand_sh_n),
                "binned": _bincount_per_dim(hand_sh_n, BINS_EDGES),
            },
            "ortho_poses": {
                "original": _compute_stats(hand_ortho),
                "minmax": hand_ortho_minmax,
                "normalized_stats": _compute_stats(hand_ortho_n),
                "binned": _bincount_per_dim(hand_ortho_n, BINS_EDGES),
            },
        },
        "object": {
            "mask_params": {
                "original": _compute_stats(obj_mask),
                "minmax": obj_mask_minmax,
                "normalized_stats": _compute_stats(obj_mask_n),
                "binned": _bincount_per_dim(obj_mask_n, BINS_EDGES),
            },
            "sh_params": {
                "original": _compute_stats(obj_sh),
                "minmax": obj_sh_minmax,
                "normalized_stats": _compute_stats(obj_sh_n),
                "binned": _bincount_per_dim(obj_sh_n, BINS_EDGES),
            },
            "ortho_poses": {
                "original": _compute_stats(obj_ortho),
                "minmax": obj_ortho_minmax,
                "normalized_stats": _compute_stats(obj_ortho_n),
                "binned": _bincount_per_dim(obj_ortho_n, BINS_EDGES),
            },
        },
        "bins_edges": BINS_EDGES.tolist(),
    }

    # Save statistics as JSON
    stats_path = os.path.join(out_dir, "stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    # Visualize per-dimension histograms
    _plot_hist_per_dim(positions_n, "positions", os.path.join(out_dir, "positions_hist.png"))
    _plot_hist_per_dim(hand_mask_n, "hand.mask_params", os.path.join(out_dir, "hand_mask_hist.png"))
    _plot_hist_per_dim(hand_sh_n, "hand.sh_params", os.path.join(out_dir, "hand_sh_hist.png"))
    _plot_hist_per_dim(hand_ortho_n, "hand.ortho_poses", os.path.join(out_dir, "hand_ortho_hist.png"))
    _plot_hist_per_dim(obj_mask_n, "object.mask_params", os.path.join(out_dir, "object_mask_hist.png"))
    _plot_hist_per_dim(obj_sh_n, "object.sh_params", os.path.join(out_dir, "object_sh_hist.png"))
    _plot_hist_per_dim(obj_ortho_n, "object.ortho_poses", os.path.join(out_dir, "object_ortho_hist.png"))

    print(f"分析完成：统计保存到 {stats_path}，图像保存到 {out_dir}")
    return out_dir, stats


def main():
    parser = argparse.ArgumentParser(description="统计并可视化数据集参数分布")
    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="数据集目录，包含若干 .npy 文件（每个文件为一个样本字典）",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join("mash-hoi", "runs"),
        help="输出基目录（将自动创建 analysis_时间戳 子目录）",
    )
    args = parser.parse_args()

    analyze(args.root_dir, args.output_dir)


if __name__ == "__main__":
    main()