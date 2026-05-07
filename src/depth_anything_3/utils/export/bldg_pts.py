import os
import numpy as np
import open3d as o3d

from depth_anything_3.specs import Prediction
from depth_anything_3.utils.logger import logger
from depth_anything_3.utils.RTS import _load_binary_masks
from .glb import _depths_to_world_points_with_mask


def export_to_bldg_pts(
    prediction: Prediction,
    export_dir: str,
    conf_thresh_percentile: float = 40.0,
    bldg_mask_paths: list[str] | None = None,
    num_max_points: int = 100000,
):
    return
    # This never worked, Idk why but we have other workaround

    conf_thresh = -np.inf
    if prediction.conf is not None and conf_thresh_percentile is not None:
        conf_thresh = np.percentile(prediction.conf, conf_thresh_percentile)

    N, H, W = prediction.depth.shape
    if bldg_mask_paths is not None:
        bldg_masks = _load_binary_masks(bldg_mask_paths, (N, H, W))
    else:
        raise NotImplementedError("Not implemented")

    bldg_points = _depths_to_world_points_with_mask(
        prediction.depth,
        prediction.intrinsics,
        prediction.extrinsics,
        bldg_masks,
        prediction.conf,
        conf_thresh,
    )

    if num_max_points is not None and bldg_points.shape[0] > num_max_points:
        sample_idx = np.random.choice(bldg_points.shape[0], num_max_points, replace=False)
        bldg_points = bldg_points[sample_idx]

    colmap_dir = os.path.join(export_dir, "DA3_colmap")
    os.makedirs(colmap_dir, exist_ok=True)
    ply_path = os.path.join(colmap_dir, "pointsBLDG.ply")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(bldg_points)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=8))
    o3d.io.write_point_cloud(ply_path, pcd)
    logger.info(f"Saved building point cloud to {ply_path}")
    