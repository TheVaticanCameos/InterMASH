import os
import sys
import pickle
from typing import Dict, Tuple, List

import numpy as np


def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))


def _load_sample_qpos_from_pkl(path: str) -> Dict[str, np.ndarray]:
    with open(path, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict) or "sample_qpos" not in data or not isinstance(data["sample_qpos"], dict):
        raise ValueError(f"Unexpected pkl format at {path}")
    result: Dict[str, np.ndarray] = {}
    for k, v in data["sample_qpos"].items():
        arr = np.asarray(v, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] < 9:
            continue
        result[k] = arr
    if not result:
        raise ValueError(f"No valid qpos entries in {path}")
    return result


def _compute_R_from_rot6d(rot6d: np.ndarray) -> np.ndarray:
    project_root = _project_root()
    if project_root not in sys.path:
        sys.path.append(project_root)
    sys.path.append(os.path.join(project_root, "ma-sh"))
    from ma_sh.Method.rotate import compute_rotation_matrix_from_ortho6d
    import torch

    rot = np.asarray(rot6d, dtype=np.float32).reshape(-1, 6)
    rot_t = torch.from_numpy(rot)
    R = compute_rotation_matrix_from_ortho6d(rot_t).detach().cpu().numpy()
    return R.reshape(-1, 3, 3)


def _select_random_key(keys: List[str], prefix: str | None = None, seed: int = 0) -> str:
    rng = np.random.default_rng(seed=seed)
    if prefix is not None:
        candidates = [k for k in keys if k.startswith(prefix)]
        if candidates:
            return str(rng.choice(candidates))
    return str(rng.choice(keys))


def _load_gt_metadata_from_pt(pt_path: str) -> List[dict]:
    import torch

    obj = torch.load(pt_path, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict) or "metadata" not in obj:
        raise ValueError(f"Unexpected GT pt format: {type(obj)}")
    metadata = obj["metadata"]
    if not isinstance(metadata, list):
        raise ValueError(f"Unexpected GT metadata type: {type(metadata)}")
    return metadata


def _load_gt_poses_from_metadata(
    metadata: List[dict],
    object_name: str,
) -> Tuple[np.ndarray, np.ndarray]:
    import torch

    translations: List[np.ndarray] = []
    rotations: List[np.ndarray] = []
    for item in metadata:
        if not isinstance(item, dict):
            continue
        if item.get("object_name", None) != object_name:
            continue
        t = item.get("translations", None)
        R = item.get("rotations", None)
        if t is None or R is None:
            continue
        if not (torch.is_tensor(t) and torch.is_tensor(R)):
            continue
        t_np = t.detach().cpu().to(dtype=torch.float32).numpy().reshape(3)
        R_np = R.detach().cpu().to(dtype=torch.float32).numpy().reshape(3, 3)
        translations.append(t_np)
        rotations.append(R_np)

    if not translations:
        raise ValueError(f"No GT poses found for object_name={object_name}")
    T = np.stack(translations, axis=0)
    R = np.stack(rotations, axis=0)
    return R, T


def _decode_qpos_rot_trans(qpos: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    qpos = np.asarray(qpos, dtype=np.float32)
    if qpos.ndim != 2 or qpos.shape[1] < 9:
        raise ValueError(f"Unexpected qpos shape: {qpos.shape}")
    T = qpos[:, 0:3]
    rot6d = qpos[:, 3:9]
    R = _compute_R_from_rot6d(rot6d)
    return R, T


def _frames_from_RT(R: np.ndarray, T: np.ndarray, axis_len: float = 0.02) -> Tuple[np.ndarray, np.ndarray]:
    R = np.asarray(R, dtype=np.float32)
    T = np.asarray(T, dtype=np.float32)
    if R.ndim != 3 or R.shape[1:] != (3, 3):
        raise ValueError(f"Unexpected R shape: {R.shape}")
    if T.ndim != 2 or T.shape[1] != 3 or T.shape[0] != R.shape[0]:
        raise ValueError(f"Unexpected T shape: {T.shape} for R: {R.shape}")

    B = R.shape[0]
    origins = []
    xs = []
    ys = []
    zs = []

    e0 = np.array([axis_len, 0.0, 0.0], dtype=np.float32)
    e1 = np.array([0.0, axis_len, 0.0], dtype=np.float32)
    e2 = np.array([0.0, 0.0, axis_len], dtype=np.float32)

    for i in range(B):
        Ri = R[i]
        ti = T[i]
        o = Ri @ ti
        x_end = Ri @ (ti + e0)
        y_end = Ri @ (ti + e1)
        z_end = Ri @ (ti + e2)
        origins.append(o)
        xs.append(x_end)
        ys.append(y_end)
        zs.append(z_end)

    origins_np = np.stack(origins, axis=0)
    xs_np = np.stack(xs, axis=0)
    ys_np = np.stack(ys, axis=0)
    zs_np = np.stack(zs, axis=0)

    pts = np.concatenate([origins_np, xs_np, ys_np, zs_np], axis=0)
    colors = np.zeros((pts.shape[0], 3), dtype=np.float32)
    colors[0:B, :] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    colors[B : 2 * B, :] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    colors[2 * B : 3 * B, :] = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    colors[3 * B : 4 * B, :] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return pts, colors


def _plot_frames_matplotlib(
    pts: np.ndarray,
    colors: np.ndarray,
    out_path: str,
    xyz_limits: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]] | None = None,
) -> None:
    if os.environ.get("DISPLAY") is None:
        import matplotlib

        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    pts = np.asarray(pts, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"Unexpected pts shape: {pts.shape}")
    if colors.shape != (pts.shape[0], 3):
        raise ValueError(f"Unexpected colors shape: {colors.shape}")

    fig = plt.figure(figsize=(6, 6), dpi=220)
    ax = fig.add_subplot(111, projection="3d")

    B = pts.shape[0] // 4
    origin = pts[0:B]
    x_end = pts[B : 2 * B]
    y_end = pts[2 * B : 3 * B]
    z_end = pts[3 * B : 4 * B]

    for i in range(B):
        o = origin[i]
        ax.plot([o[0], x_end[i, 0]], [o[1], x_end[i, 1]], [o[2], x_end[i, 2]], color="r", linewidth=0.5)
        ax.plot([o[0], y_end[i, 0]], [o[1], y_end[i, 1]], [o[2], y_end[i, 2]], color="g", linewidth=0.5)
        ax.plot([o[0], z_end[i, 0]], [o[1], z_end[i, 1]], [o[2], z_end[i, 2]], color="b", linewidth=0.5)

    if xyz_limits is not None:
        (xmin, xmax), (ymin, ymax), (zmin, zmax) = xyz_limits
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_zlim(zmin, zmax)
    else:
        xmin, ymin, zmin = pts.min(axis=0)
        xmax, ymax, zmax = pts.max(axis=0)
        margin = 0.05 * max(xmax - xmin, ymax - ymin, zmax - zmin)
        cx = 0.5 * (xmin + xmax)
        cy = 0.5 * (ymin + ymax)
        cz = 0.5 * (zmin + zmax)
        r = 0.5 * max(xmax - xmin, ymax - ymin, zmax - zmin) + margin
        ax.set_xlim(cx - r, cx + r)
        ax.set_ylim(cy - r, cy + r)
        ax.set_zlim(cz - r, cz + r)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.view_init(elev=20.0, azim=-60.0)
    fig.tight_layout(pad=0.05)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    ours_pkl = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "/home/yxz/MyProjects/MASH-HOI/mash_hoi/Results/Mesh/exp_ddpm_phys_dgn/res/part0/res_mash-hoi.pkl"
    )
    dga_pkl = (
        sys.argv[2]
        if len(sys.argv) > 2
        else "/home/yxz/MyProjects/DexGrasp-Anything/Results/eval/final/2025-12-24_10-00-09/res_diffuser.pkl"
    )
    gt_pt = (
        sys.argv[3]
        if len(sys.argv) > 3
        else "/home/yxz/MyProjects/DATASET/Original-Dataset/Original-DexGraspNet/DexGraspNet/dexgraspnet_shadowhand_downsample.pt"
    )
    out_dir = (
        sys.argv[4]
        if len(sys.argv) > 4
        else "/home/yxz/MyProjects/MASH-HOI/mash_hoi/Results/Mesh/exp_ddpm_phys_dgn/res/part0/global_pose_vis"
    )
    os.makedirs(out_dir, exist_ok=True)

    seed = int(os.environ.get("POSE_VIS_SEED", "0"))
    axis_len = float(os.environ.get("POSE_VIS_AXIS_LEN", "0.02"))

    ours_dict = _load_sample_qpos_from_pkl(ours_pkl)
    dga_dict = _load_sample_qpos_from_pkl(dga_pkl)
    gt_metadata = _load_gt_metadata_from_pt(gt_pt)

    rng = np.random.default_rng(seed=seed)

    ours_keys = sorted(list(ours_dict.keys()))
    dga_keys = sorted(list(dga_dict.keys()))
    if not ours_keys:
        raise ValueError("No objects found in Ours pkl.")
    if not dga_keys:
        raise ValueError("No objects found in DGA pkl.")

    ours_key = str(rng.choice(ours_keys))
    dga_key = str(rng.choice(dga_keys))

    gt_index = int(rng.integers(low=0, high=len(gt_metadata)))
    gt_object_name = gt_metadata[gt_index].get("object_name")
    if not isinstance(gt_object_name, str):
        raise ValueError(f"Unexpected GT object_name at index {gt_index}: {gt_object_name!r}")

    ours_R, ours_T = _decode_qpos_rot_trans(ours_dict[ours_key])
    dga_R, dga_T = _decode_qpos_rot_trans(dga_dict[dga_key])
    gt_R, gt_T = _load_gt_poses_from_metadata(gt_metadata, object_name=gt_object_name)

    print(f"Selected Ours object key: {ours_key}")
    print(f"Selected DGA object key:  {dga_key}")
    print(f"Selected GT object_name:  {gt_object_name}")

    ours_pts, ours_colors = _frames_from_RT(ours_R, ours_T, axis_len=axis_len)
    dga_pts, dga_colors = _frames_from_RT(dga_R, dga_T, axis_len=axis_len)
    gt_pts, gt_colors = _frames_from_RT(gt_R, gt_T, axis_len=axis_len)

    all_pts = np.concatenate([ours_pts, dga_pts, gt_pts], axis=0)
    xmin, ymin, zmin = all_pts.min(axis=0)
    xmax, ymax, zmax = all_pts.max(axis=0)
    margin = 0.05 * max(xmax - xmin, ymax - ymin, zmax - zmin)
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    cz = 0.5 * (zmin + zmax)
    r = 0.5 * max(xmax - xmin, ymax - ymin, zmax - zmin) + margin
    xyz_limits = ((cx - r, cx + r), (cy - r, cy + r), (cz - r, cz + r))

    ours_out = os.path.join(out_dir, f"{ours_key}_ours.png")
    dga_out = os.path.join(out_dir, f"{dga_key}_dga.png")
    gt_safe_name = gt_object_name.replace("/", "_").replace(" ", "_") if isinstance(gt_object_name, str) else "gt_object"
    gt_out = os.path.join(out_dir, f"{gt_safe_name}_gt.png")

    _plot_frames_matplotlib(ours_pts, ours_colors, ours_out, xyz_limits=xyz_limits)
    _plot_frames_matplotlib(dga_pts, dga_colors, dga_out, xyz_limits=xyz_limits)
    _plot_frames_matplotlib(gt_pts, gt_colors, gt_out, xyz_limits=xyz_limits)

    print(f"Ours poses: R={ours_R.shape} T={ours_T.shape} -> {ours_out}")
    print(f"DGA poses:  R={dga_R.shape} T={dga_T.shape} -> {dga_out}")
    print(f"GT poses:   R={gt_R.shape} T={gt_T.shape} -> {gt_out}")


if __name__ == "__main__":
    main()
