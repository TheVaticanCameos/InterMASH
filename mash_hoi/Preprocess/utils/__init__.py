# Preprocess utilities submodule

from .normal import (
    # PyTorch3D methods
    orient_normals_consistent,
    orient_normals_outward,
    estimate_and_orient_normals_pytorch3d,
    # Open3D methods
    estimate_and_orient_normals_open3d,
    # WNNC methods
    estimate_normals_wnnc,
    # Comparison
    compare_normal_estimation_methods,
)

from .mash_utils import (
    split_mash_params,
    compute_sample_points_from_mash_params,
    denormalize_params,
)

from .edge_utils import (
    # Disk fitting
    fit_disk_with_normals,
    fit_disks_batched,
    # IoU computation
    compute_edge_matrix_batched,
    # High-level processing
    process_batch
)

from .visualize_utils import (
    visualize_iou_on_object_pointcloud,
    visualize_edge_detailed,
)
