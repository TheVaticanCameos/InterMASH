import os
import sys
import pickle
from typing import Any, Dict, Tuple

import numpy as np


def _as_rows(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        arr = x
    elif isinstance(x, (list, tuple)):
        if len(x) == 0:
            return np.zeros((0, 0), dtype=np.float32)
        if isinstance(x[0], np.ndarray):
            arr = np.stack([np.asarray(v) for v in x], axis=0)
        elif isinstance(x[0], (list, tuple)):
            arr = np.asarray(x)
        else:
            arr = np.asarray(x)
    else:
        arr = np.asarray(x)

    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"Unexpected data ndim: {arr.ndim}, shape={arr.shape}")
    return arr


def _collect_rows_from_pickle(data: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
    meta: Dict[str, Any] = {"format": type(data).__name__}

    if isinstance(data, dict):
        if "sample_qpos" in data and isinstance(data["sample_qpos"], dict):
            qpos_dict = data["sample_qpos"]
            keys = sorted(qpos_dict.keys())
            arrays = []
            for k in keys:
                arr_k = _as_rows(qpos_dict[k])
                arrays.append(arr_k)
            meta["sample_qpos_key_count"] = len(keys)
            meta["sample_qpos_key_head"] = keys[:5]
            if len(arrays) == 0:
                return np.zeros((0, 0), dtype=np.float32), meta
            rows = np.concatenate(arrays, axis=0).astype(np.float32, copy=False)
            meta["row_count"] = int(rows.shape[0])
            meta["row_dim"] = int(rows.shape[1]) if rows.ndim == 2 else None
            return rows, meta

        for candidate_key in ("qpos", "data", "samples", "X", "features"):
            if candidate_key in data:
                arr = _as_rows(data[candidate_key]).astype(np.float32, copy=False)
                meta["key"] = candidate_key
                return arr, meta

        raise ValueError(f"Unrecognized dict keys: {sorted(data.keys())[:50]}")

    arr = _as_rows(data).astype(np.float32, copy=False)
    return arr, meta


def _load_rows_from_pkl_path(pkl_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    rows, meta = _collect_rows_from_pickle(data)
    meta = dict(meta)
    meta["path"] = pkl_path
    return rows, meta


def _sample_rows(rows: np.ndarray, n: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rows = np.asarray(rows)
    if rows.ndim != 2:
        raise ValueError(f"Unexpected rows shape: {rows.shape}")
    n_total = int(rows.shape[0])
    if n <= 0 or n_total <= n:
        idx = np.arange(n_total, dtype=np.int64)
        return rows, idx
    rng = np.random.default_rng(seed=seed)
    idx = rng.choice(n_total, size=int(n), replace=False).astype(np.int64, copy=False)
    idx = np.sort(idx)
    return rows[idx], idx


def _tsne_embed_2d(features: np.ndarray, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    from sklearn.manifold import TSNE

    X = np.asarray(features, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"Unexpected features shape: {X.shape}")

    n_total = X.shape[0]
    idx = np.arange(n_total, dtype=np.int64)
    n = n_total
    if n < 2:
        return np.zeros((n, 2), dtype=np.float64), idx

    max_points = int(os.environ.get("TSNE_MAX_POINTS", "0"))
    if max_points > 0 and n > max_points:
        rng = np.random.default_rng(seed=random_state)
        idx = rng.choice(n_total, size=max_points, replace=False)
        idx = np.sort(idx)
        X = X[idx]
        n = X.shape[0]

    perplexity = float(os.environ.get("TSNE_PERPLEXITY", "30"))
    perplexity = min(perplexity, max(5.0, (n - 1) / 3.0))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=random_state,
    )
    Y = tsne.fit_transform(X)
    return Y, idx


def _save_tsne_png(emb2d: np.ndarray, out_path: str, point_size: float = 3.0) -> None:
    if os.environ.get("DISPLAY") is None:
        import matplotlib

        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    emb2d = np.asarray(emb2d, dtype=np.float64)
    if emb2d.ndim != 2 or emb2d.shape[1] != 2:
        raise ValueError(f"Unexpected emb2d shape: {emb2d.shape}")

    fig = plt.figure(figsize=(8, 8), dpi=220)
    ax = fig.add_subplot(111)
    ax.scatter(emb2d[:, 0], emb2d[:, 1], s=float(point_size), linewidths=0, alpha=0.85)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout(pad=0.1)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def _save_compare_tsne_png(
    emb2d_ours: np.ndarray,
    emb2d_dga: np.ndarray,
    out_path: str,
    point_size: float = 3.0,
) -> None:
    if os.environ.get("DISPLAY") is None:
        import matplotlib

        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    emb2d_ours = np.asarray(emb2d_ours, dtype=np.float64)
    emb2d_dga = np.asarray(emb2d_dga, dtype=np.float64)
    if emb2d_ours.ndim != 2 or emb2d_ours.shape[1] != 2:
        raise ValueError(f"Unexpected emb2d_ours shape: {emb2d_ours.shape}")
    if emb2d_dga.ndim != 2 or emb2d_dga.shape[1] != 2:
        raise ValueError(f"Unexpected emb2d_dga shape: {emb2d_dga.shape}")

    fig = plt.figure(figsize=(8, 8), dpi=240)
    ax = fig.add_subplot(111)
    ax.scatter(
        emb2d_dga[:, 0],
        emb2d_dga[:, 1],
        s=float(point_size),
        c="#2ca02c",
        linewidths=0,
        alpha=0.75,
        label="DGA",
    )
    ax.scatter(
        emb2d_ours[:, 0],
        emb2d_ours[:, 1],
        s=float(point_size),
        c="#d62728",
        linewidths=0,
        alpha=0.75,
        label="Ours",
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout(pad=0.1)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def _load_gt_joint_positions_from_pt_path(
    pt_path: str,
    sample_n: int,
    seed: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    import torch

    obj = torch.load(pt_path, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict) or "metadata" not in obj:
        raise ValueError(f"Unexpected GT pt format: {type(obj)}")

    metadata = obj["metadata"]
    if not isinstance(metadata, list) or len(metadata) == 0:
        raise ValueError(f"Unexpected GT metadata type/len: {type(metadata)} len={len(metadata) if hasattr(metadata,'__len__') else 'NA'}")

    n_total = len(metadata)
    n_take = int(sample_n) if int(sample_n) > 0 else n_total
    n_take = min(n_take, n_total)
    rng = np.random.default_rng(seed=seed)
    idx = rng.choice(n_total, size=n_take, replace=False).astype(np.int64, copy=False)

    rows = np.empty((n_take, 24), dtype=np.float32)
    for i, j in enumerate(idx.tolist()):
        item = metadata[j]
        if not isinstance(item, dict) or "joint_positions" not in item:
            raise ValueError(f"Unexpected GT metadata item at {j}: {type(item)}")
        jp = item["joint_positions"]
        if not torch.is_tensor(jp):
            raise ValueError(f"Unexpected joint_positions type at {j}: {type(jp)}")
        jp = jp.detach().cpu().to(dtype=torch.float32).reshape(-1)
        if jp.numel() != 24:
            raise ValueError(f"Unexpected joint_positions dim at {j}: {tuple(jp.shape)}")
        rows[i] = jp.numpy()

    meta: Dict[str, Any] = {
        "format": "pt",
        "path": pt_path,
        "metadata_len": int(n_total),
        "sample_n": int(n_take),
        "dim": 24,
    }
    return rows, meta


def _save_compare3_tsne_png(
    emb2d_ours: np.ndarray,
    emb2d_dga: np.ndarray,
    emb2d_gt: np.ndarray,
    out_path: str,
    point_size: float = 3.0,
) -> None:
    if os.environ.get("DISPLAY") is None:
        import matplotlib

        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    emb2d_ours = np.asarray(emb2d_ours, dtype=np.float64)
    emb2d_dga = np.asarray(emb2d_dga, dtype=np.float64)
    emb2d_gt = np.asarray(emb2d_gt, dtype=np.float64)
    if emb2d_ours.ndim != 2 or emb2d_ours.shape[1] != 2:
        raise ValueError(f"Unexpected emb2d_ours shape: {emb2d_ours.shape}")
    if emb2d_dga.ndim != 2 or emb2d_dga.shape[1] != 2:
        raise ValueError(f"Unexpected emb2d_dga shape: {emb2d_dga.shape}")
    if emb2d_gt.ndim != 2 or emb2d_gt.shape[1] != 2:
        raise ValueError(f"Unexpected emb2d_gt shape: {emb2d_gt.shape}")

    fig = plt.figure(figsize=(8, 8), dpi=260)
    ax = fig.add_subplot(111)
    ax.scatter(
        emb2d_gt[:, 0],
        emb2d_gt[:, 1],
        s=float(point_size),
        c="#1f77b4",
        linewidths=0,
        alpha=0.70,
        label="GT",
    )
    ax.scatter(
        emb2d_dga[:, 0],
        emb2d_dga[:, 1],
        s=float(point_size),
        c="#2ca02c",
        linewidths=0,
        alpha=0.70,
        label="DGA",
    )
    ax.scatter(
        emb2d_ours[:, 0],
        emb2d_ours[:, 1],
        s=float(point_size),
        c="#d62728",
        linewidths=0,
        alpha=0.70,
        label="Ours",
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout(pad=0.1)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def _compare_tsne(
    dga_pkl_path: str,
    ours_pkl_path: str,
    out_path: str,
    sample_n: int,
    random_state: int,
    point_size: float,
    dga_slice_start: int,
    ours_slice_start: int,
) -> None:
    rows_dga, meta_dga = _load_rows_from_pkl_path(dga_pkl_path)
    rows_ours, meta_ours = _load_rows_from_pkl_path(ours_pkl_path)

    if rows_dga.ndim != 2 or rows_dga.shape[0] == 0:
        raise ValueError(f"Empty DGA rows: {rows_dga.shape}")
    if rows_ours.ndim != 2 or rows_ours.shape[0] == 0:
        raise ValueError(f"Empty Ours rows: {rows_ours.shape}")

    if rows_dga.shape[1] <= dga_slice_start:
        raise ValueError(f"DGA dim too small: {rows_dga.shape}, slice_start={dga_slice_start}")
    if rows_ours.shape[1] <= ours_slice_start:
        raise ValueError(f"Ours dim too small: {rows_ours.shape}, slice_start={ours_slice_start}")

    rows_dga_s, idx_dga = _sample_rows(rows_dga, n=sample_n, seed=random_state + 11)
    rows_ours_s, idx_ours = _sample_rows(rows_ours, n=sample_n, seed=random_state + 17)

    feat_dga = rows_dga_s[:, int(dga_slice_start) :].astype(np.float32, copy=False)
    feat_ours = rows_ours_s[:, int(ours_slice_start) :].astype(np.float32, copy=False)
    if feat_dga.shape[1] != feat_ours.shape[1]:
        raise ValueError(
            f"Feature dims mismatch: DGA {feat_dga.shape} vs Ours {feat_ours.shape}. "
            f"Try aligning slice starts (e.g., both 9)."
        )

    X = np.concatenate([feat_dga, feat_ours], axis=0)
    emb2d, idx = _tsne_embed_2d(X, random_state=random_state)
    emb2d = emb2d[idx]

    n_dga = feat_dga.shape[0]
    emb2d_dga = emb2d[:n_dga]
    emb2d_ours = emb2d[n_dga:]
    _save_compare_tsne_png(emb2d_ours=emb2d_ours, emb2d_dga=emb2d_dga, out_path=out_path, point_size=point_size)

    print(f"DGA:  {dga_pkl_path}")
    print(f"  rows={rows_dga.shape} sample={rows_dga_s.shape} slice=[{dga_slice_start}:] -> {feat_dga.shape[1]}")
    print(f"  meta={meta_dga}")
    print(f"Ours: {ours_pkl_path}")
    print(f"  rows={rows_ours.shape} sample={rows_ours_s.shape} slice=[{ours_slice_start}:] -> {feat_ours.shape[1]}")
    print(f"  meta={meta_ours}")
    print(f"Saved: {out_path}")


def _compare3_tsne(
    dga_pkl_path: str,
    ours_pkl_path: str,
    gt_pt_path: str,
    out_path: str,
    sample_n: int,
    random_state: int,
    point_size: float,
    dga_slice_start: int,
    ours_slice_start: int,
) -> None:
    rows_dga, meta_dga = _load_rows_from_pkl_path(dga_pkl_path)
    rows_ours, meta_ours = _load_rows_from_pkl_path(ours_pkl_path)
    rows_gt, meta_gt = _load_gt_joint_positions_from_pt_path(gt_pt_path, sample_n=sample_n, seed=random_state + 23)

    if rows_dga.ndim != 2 or rows_dga.shape[0] == 0:
        raise ValueError(f"Empty DGA rows: {rows_dga.shape}")
    if rows_ours.ndim != 2 or rows_ours.shape[0] == 0:
        raise ValueError(f"Empty Ours rows: {rows_ours.shape}")
    if rows_gt.ndim != 2 or rows_gt.shape[0] == 0:
        raise ValueError(f"Empty GT rows: {rows_gt.shape}")

    if rows_dga.shape[1] <= dga_slice_start:
        raise ValueError(f"DGA dim too small: {rows_dga.shape}, slice_start={dga_slice_start}")
    if rows_ours.shape[1] <= ours_slice_start:
        raise ValueError(f"Ours dim too small: {rows_ours.shape}, slice_start={ours_slice_start}")

    rows_dga_s, _ = _sample_rows(rows_dga, n=sample_n, seed=random_state + 11)
    rows_ours_s, _ = _sample_rows(rows_ours, n=sample_n, seed=random_state + 17)

    feat_dga = rows_dga_s[:, int(dga_slice_start) :].astype(np.float32, copy=False)
    feat_ours = rows_ours_s[:, int(ours_slice_start) :].astype(np.float32, copy=False)
    feat_gt = rows_gt.astype(np.float32, copy=False)

    if feat_dga.shape[1] != feat_ours.shape[1] or feat_dga.shape[1] != feat_gt.shape[1]:
        raise ValueError(
            f"Feature dims mismatch: DGA {feat_dga.shape}, Ours {feat_ours.shape}, GT {feat_gt.shape}. "
            f"Try aligning slice starts (e.g., both 9)."
        )

    X = np.concatenate([feat_gt, feat_dga, feat_ours], axis=0)
    emb2d, idx = _tsne_embed_2d(X, random_state=random_state)
    emb2d = emb2d[idx]

    n_gt = feat_gt.shape[0]
    n_dga = feat_dga.shape[0]
    emb2d_gt = emb2d[:n_gt]
    emb2d_dga = emb2d[n_gt : n_gt + n_dga]
    emb2d_ours = emb2d[n_gt + n_dga :]
    _save_compare3_tsne_png(
        emb2d_ours=emb2d_ours,
        emb2d_dga=emb2d_dga,
        emb2d_gt=emb2d_gt,
        out_path=out_path,
        point_size=point_size,
    )

    print(f"GT:   {gt_pt_path}")
    print(f"  rows={rows_gt.shape} sample={rows_gt.shape} dim=24")
    print(f"  meta={meta_gt}")
    print(f"DGA:  {dga_pkl_path}")
    print(f"  rows={rows_dga.shape} sample={rows_dga_s.shape} slice=[{dga_slice_start}:] -> {feat_dga.shape[1]}")
    print(f"  meta={meta_dga}")
    print(f"Ours: {ours_pkl_path}")
    print(f"  rows={rows_ours.shape} sample={rows_ours_s.shape} slice=[{ours_slice_start}:] -> {feat_ours.shape[1]}")
    print(f"  meta={meta_ours}")
    print(f"Saved: {out_path}")


def main() -> None:
    random_state = int(os.environ.get("TSNE_SEED", "42"))
    point_size = float(os.environ.get("TSNE_POINT_SIZE", "3.0"))
    sample_n = int(os.environ.get("TSNE_SAMPLE_N", "1000"))

    if len(sys.argv) > 1 and sys.argv[1] == "--compare3":
        dga_pkl_path = (
            sys.argv[2]
            if len(sys.argv) > 2
            else "/home/yxz/MyProjects/DexGrasp-Anything/Results/eval/final/2025-12-24_10-00-09/res_diffuser.pkl"
        )
        ours_pkl_path = (
            sys.argv[3]
            if len(sys.argv) > 3
            else "/home/yxz/MyProjects/MASH-HOI/mash_hoi/Results/Mesh/exp_ddpm_phys_dgn/res/part0/res_mash-hoi.pkl"
        )
        gt_pt_path = (
            sys.argv[4]
            if len(sys.argv) > 4
            else "/home/yxz/MyProjects/DATASET/Original-Dataset/Original-DexGraspNet/DexGraspNet/dexgraspnet_shadowhand_downsample.pt"
        )
        out_path = sys.argv[5] if len(sys.argv) > 5 else ""
        if out_path == "":
            out_path = os.path.join(os.path.dirname(ours_pkl_path), "tsne_compare_gt_dga_ours.png")

        dga_slice_start = int(os.environ.get("TSNE_DGA_SLICE_START", "9"))
        ours_slice_start = int(os.environ.get("TSNE_OURS_SLICE_START", "9"))
        _compare3_tsne(
            dga_pkl_path=dga_pkl_path,
            ours_pkl_path=ours_pkl_path,
            gt_pt_path=gt_pt_path,
            out_path=out_path,
            sample_n=sample_n,
            random_state=random_state,
            point_size=point_size,
            dga_slice_start=dga_slice_start,
            ours_slice_start=ours_slice_start,
        )
        return

    if len(sys.argv) > 1 and sys.argv[1] == "--compare":
        dga_pkl_path = (
            sys.argv[2]
            if len(sys.argv) > 2
            else "/home/yxz/MyProjects/DexGrasp-Anything/Results/eval/final/2025-12-24_10-00-09/res_diffuser.pkl"
        )
        ours_pkl_path = (
            sys.argv[3]
            if len(sys.argv) > 3
            else "/home/yxz/MyProjects/MASH-HOI/mash_hoi/Results/Mesh/exp_ddpm_phys_dgn/res/part0/res_mash-hoi.pkl"
        )
        out_path = sys.argv[4] if len(sys.argv) > 4 else ""
        if out_path == "":
            out_path = os.path.join(os.path.dirname(ours_pkl_path), "tsne_compare_dga_ours.png")

        dga_slice_start = int(os.environ.get("TSNE_DGA_SLICE_START", "9"))
        ours_slice_start = int(os.environ.get("TSNE_OURS_SLICE_START", "9"))
        _compare_tsne(
            dga_pkl_path=dga_pkl_path,
            ours_pkl_path=ours_pkl_path,
            out_path=out_path,
            sample_n=sample_n,
            random_state=random_state,
            point_size=point_size,
            dga_slice_start=dga_slice_start,
            ours_slice_start=ours_slice_start,
        )
        return

    pkl_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "/home/yxz/MyProjects/MASH-HOI/mash_hoi/Results/Mesh/exp_ddpm_phys_dgn/res/part0/res_mash-hoi.pkl"
    )
    out_path = sys.argv[2] if len(sys.argv) > 2 else ""
    if out_path == "":
        out_path = os.path.join(os.path.dirname(pkl_path), "tsne_qpos_6plus.png")

    slice_start = int(os.environ.get("TSNE_SLICE_START", "6"))
    rows, meta = _load_rows_from_pkl_path(pkl_path)
    if rows.size == 0 or rows.shape[0] == 0:
        raise ValueError("Loaded empty data from pickle")
    if rows.shape[1] <= slice_start:
        raise ValueError(f"Expected rows with dim > {slice_start}, got shape={rows.shape}")

    features = rows[:, int(slice_start) :]
    emb2d, idx = _tsne_embed_2d(features, random_state=random_state)
    _save_tsne_png(emb2d, out_path, point_size=point_size)

    print(f"Loaded: {pkl_path}")
    print(f"Rows: {rows.shape[0]}  Dim: {rows.shape[1]}  Use: [{slice_start}:] -> {features.shape[1]}")
    print(f"Format: {meta}")
    print(f"tSNE points: {emb2d.shape[0]}  Saved: {out_path}")


if __name__ == "__main__":
    main()
