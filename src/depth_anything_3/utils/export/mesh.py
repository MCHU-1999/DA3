import os
import numpy as np
import open3d as o3d

from depth_anything_3.specs import Prediction
from depth_anything_3.utils.logger import logger

from .glb import _depths_to_world_points


def export_to_mesh(
    prediction: Prediction,
    export_dir: str,
    conf_thresh_percentile: float = 10.0,
    poisson_depth: int = 8,
    min_density_quantile: float = 0.01,
) -> str:
    # 0. Make folders
    mesh_dir = os.path.join(export_dir, "DA3_mesh")
    os.makedirs(mesh_dir, exist_ok=True)
    mesh_path = os.path.join(mesh_dir, "mesh.ply")
    pc_path = os.path.join(mesh_dir, "points.ply")

    # 1. Data preparation
    conf_thresh = np.percentile(prediction.conf, conf_thresh_percentile)
    points = _depths_to_world_points(
        prediction.depth,
        prediction.intrinsics,
        prediction.extrinsics,  # w2c
        prediction.conf,
        conf_thresh,
    )
    num_points = len(points)
    logger.info(f"Exporting to mesh with {num_points} confidence-filtered points")
    num_frames = len(prediction.processed_images)
    h, w = prediction.processed_images.shape[1:3]
    points_xyf = _create_xyf(num_frames, h, w)
    points_xyf = points_xyf[prediction.conf >= conf_thresh]

    # 2) Build confidence-filtered point cloud and reconstruct mesh with Open3D
    if num_points < 100:
        raise ValueError(
            f"Not enough points ({num_points}) after confidence filtering to build a mesh"
        )

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    # pc.colors = o3d.utility.Vector3dVector((colors.astype(np.float64) / 255.0).clip(0.0, 1.0))

    pc.estimate_normals()
    pc.orient_normals_consistent_tangent_plane(30)

    # mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
    #     pc,
    #     depth=poisson_depth,
    # )
    # densities = np.asarray(densities)
    # if densities.size > 0 and 0.0 < min_density_quantile < 1.0:
    #     remove_mask = densities < np.quantile(densities, min_density_quantile)
    #     mesh.remove_vertices_by_mask(remove_mask)

    # mesh.compute_vertex_normals()

    # 3. Export
    # o3d.io.write_triangle_mesh(mesh_path, mesh, write_ascii=False, compressed=True)
    # logger.info(f"Exported mesh to {mesh_path}")

    o3d.io.write_point_cloud(pc_path, pc, write_ascii=False, compressed=True)
    logger.info(f"Exported point cloud to {pc_path}")
    logger.info(f"Kept {len(points_xyf)} 2D-3D correspondences after confidence thresholding")
    return mesh_path

def _create_xyf(num_frames, height, width):
    """
    Creates a grid of pixel coordinates and frame indices (fidx) for all frames.
    """
    # Create coordinate grids for a single frame
    y_grid, x_grid = np.indices((height, width), dtype=np.int32)
    x_grid = x_grid[np.newaxis, :, :]
    y_grid = y_grid[np.newaxis, :, :]

    # Broadcast to all frames
    x_coords = np.broadcast_to(x_grid, (num_frames, height, width))
    y_coords = np.broadcast_to(y_grid, (num_frames, height, width))

    # Create frame indices and broadcast
    f_idx = np.arange(num_frames, dtype=np.int32)[:, np.newaxis, np.newaxis]
    f_coords = np.broadcast_to(f_idx, (num_frames, height, width))

    # Stack coordinates and frame indices
    points_xyf = np.stack((x_coords, y_coords, f_coords), axis=-1)

    return points_xyf
