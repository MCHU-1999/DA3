import os

import numpy as np
from PIL import Image

from depth_anything_3.specs import Prediction
from depth_anything_3.utils.export.glb import _depths_to_world_points, _depths_to_world_points_with_mask
from depth_anything_3.utils.pose_align import apply_umeyama_alignment_to_ext
from depth_anything_3.utils.logger import logger


def _load_binary_masks(mask_paths: list[str], target_shape: tuple[int, int, int]) -> np.ndarray:
    """Load mask images and convert them to a resized boolean mask volume (N, H, W)."""
    N, H, W = target_shape
    if mask_paths is None or len(mask_paths) != N:
        raise ValueError(f"mask_paths must have length {N}, got {0 if mask_paths is None else len(mask_paths)}")

    masks = np.zeros((N, H, W), dtype=bool)
    for i, mask_path in enumerate(mask_paths):
        if not os.path.exists(mask_path):
            raise ValueError(f"Mask file not found: {mask_path}")
        with Image.open(mask_path) as m:
            m = m.convert("L")
            if m.size != (W, H):
                m = m.resize((W, H), Image.NEAREST)
            masks[i] = np.asarray(m) > 0

    if np.all(masks == 0):
        logger.warn("")
    
    return masks


def _estimate_min_building_dimension(points: np.ndarray, method: str = "pca") -> float | None:
    """Estimate the smallest building dimension from points via PCA or axis-aligned bbox."""
    if points.shape[0] < 3:
        return None

    if method == "bbox":
        extents = np.ptp(points, axis=0)
        min_dim = float(np.min(extents))
    elif method == "pca":
        center = np.mean(points, axis=0)
        X = points - center
        _, _, vt = np.linalg.svd(X, full_matrices=False)
        proj = X @ vt.T
        extents = np.ptp(proj, axis=0)
        min_dim = float(np.min(extents))
    else:
        raise ValueError(f"Unknown dimension method: {method}. Use 'pca' or 'bbox'.")

    if not np.isfinite(min_dim) or min_dim <= 1e-8:
        return None
    return min_dim


def _rotation_matrix_from_vectors(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Compute rotation matrix that rotates unit vector src to unit vector dst."""
    src = src / (np.linalg.norm(src) + 1e-12)
    dst = dst / (np.linalg.norm(dst) + 1e-12)

    c = float(np.clip(np.dot(src, dst), -1.0, 1.0))
    if c > 1.0 - 1e-10:
        return np.eye(3, dtype=np.float64)

    if c < -1.0 + 1e-10:
        # 180-degree flip around any axis orthogonal to src
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(src[0]) > 0.9:
            axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        axis = axis - np.dot(axis, src) * src
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        K = np.array(
            [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]],
            dtype=np.float64,
        )
        return np.eye(3, dtype=np.float64) + 2.0 * (K @ K)

    v = np.cross(src, dst)
    s = np.linalg.norm(v)
    K = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]], dtype=np.float64)
    R = np.eye(3, dtype=np.float64) + K + (K @ K) * ((1.0 - c) / (s * s + 1e-12))
    return R


def project_point_to_plane(p, n, d):
    """
    Projects point p onto a plane defined by normal n and constant d.
    Plane equation: n·x + d = 0
    """
    p = np.array(p)
    n = np.array(n)

    # Scalar factor: (n·p + d) / |n|²
    factor = (np.dot(n, p) + d) / np.dot(n, n)
    
    # Return the projected point
    return p - factor * n


def _orient_plane_normal_toward_cameras(
    normal: np.ndarray,
    d: float,
    camera_positions: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Flip plane normal so cameras lie on the positive side (same side as normal).
    
    Since cameras are always on the positive side of the ground plane, use their positions
    to orient the plane normal correctly.
    """
    n = normal.astype(np.float64)
    if camera_positions is None or camera_positions.shape[0] == 0:
        return n, float(d)

    signed = camera_positions @ n + float(d)
    # If cameras are mostly on the negative side, flip the plane orientation.
    if np.mean(signed) < 0:
        n = -n
        d = -float(d)
    return n, float(d)


def _fit_ground_plane_ransac(
    prediction: Prediction,
    gnd_mask_paths: list[str],
    conf_thresh_percentile: float = 40.0,
    max_iters: int = 2000,
    dist_thresh: float | None = None,
    random_state: int = 42,
) -> dict | None:
    """Fit a ground plane from masked world points using RANSAC.

    Returns:
        Dict with keys: normal, d, center, num_points, num_inliers, inlier_ratio, dist_thresh.
        Returns None if insufficient valid points are available.
    """
    if prediction.depth is None or prediction.intrinsics is None or prediction.extrinsics is None:
        return None

    N, H, W = prediction.depth.shape
    if gnd_mask_paths is None or len(gnd_mask_paths) != N:
        raise ValueError(f"gnd_mask_paths must have length {N}, got {0 if gnd_mask_paths is None else len(gnd_mask_paths)}")

    # Load binary masks and resize to depth resolution if needed.
    gnd_masks = _load_binary_masks(gnd_mask_paths, (N, H, W))

    conf_thresh = -np.inf
    if prediction.conf is not None and conf_thresh_percentile is not None:
        conf_thresh = np.percentile(prediction.conf, conf_thresh_percentile)

    points = _depths_to_world_points_with_mask(
        prediction.depth,
        prediction.intrinsics,
        prediction.extrinsics,
        gnd_masks,
        prediction.conf,
        conf_thresh,
    )

    num_points = int(points.shape[0])
    if num_points < 3:
        return None

    if dist_thresh is None:
        lo = np.percentile(points, 5, axis=0)
        hi = np.percentile(points, 95, axis=0)
        scene_diag = np.linalg.norm(hi - lo)
        dist_thresh = max(float(scene_diag) * 0.005, 1e-4)

    rng = np.random.default_rng(random_state)
    best_count = 0
    best_inliers = None
    best_n = None
    best_d = None

    for _ in range(max_iters):
        idx = rng.choice(num_points, size=3, replace=False)
        p0, p1, p2 = points[idx]
        n = np.cross(p1 - p0, p2 - p0)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-10:
            continue
        n = n / n_norm
        d = -np.dot(n, p0)

        dists = np.abs(points @ n + d)
        inliers = dists <= dist_thresh
        count = int(inliers.sum())

        if count > best_count:
            best_count = count
            best_inliers = inliers
            best_n = n
            best_d = d
            if count >= int(0.95 * num_points):
                break

    if best_inliers is None or best_count < 3:
        return None

    # Refit plane from inliers with SVD.
    inlier_pts = points[best_inliers]
    center = inlier_pts.mean(axis=0)
    _, _, vt = np.linalg.svd(inlier_pts - center, full_matrices=False)
    n_refit = vt[-1]
    n_refit_norm = np.linalg.norm(n_refit)
    if n_refit_norm < 1e-10:
        n_refit = best_n
        d_refit = best_d
    else:
        n_refit = n_refit / n_refit_norm
        d_refit = -np.dot(n_refit, center)

    return {
        "normal": n_refit.astype(np.float32),
        "d": float(d_refit),
        "center": center.astype(np.float32),
        "num_points": num_points,
        "num_inliers": best_count,
        "inlier_ratio": float(best_count / max(1, num_points)),
        "dist_thresh": float(dist_thresh),
    }


def rotate_translate_scale(
    prediction: Prediction,
    conf_thresh_percentile: float = 40.0,
    bldg_mask_paths: list[str] | None = None,
    gnd_mask_paths: list[str] | None = None,
    target_min_bldg_dim_m: float = 12.0,
    rotate_to_ground: bool = True,
    world_up_axis: tuple[float, float, float] = (0.0, 0.0, 1.0),
):
    """Shift world origin so the robust scene center is at (0, 0, 0)."""
    logger.info(f"[Rotate-Translate-Scale]")

    if prediction.extrinsics is None:
        return prediction

    conf_thresh = -np.inf
    if prediction.conf is not None and conf_thresh_percentile is not None:
        conf_thresh = np.percentile(prediction.conf, conf_thresh_percentile)

    # ===== STEP 1: Scale building to target dimension =====
    # Do this first so RANSAC dist_thresh is in final metric units (e.g., meters).
    if bldg_mask_paths is not None:
        N, H, W = prediction.depth.shape
        bldg_masks = _load_binary_masks(bldg_mask_paths, (N, H, W))
        bldg_points = _depths_to_world_points_with_mask(
            prediction.depth,
            prediction.intrinsics,
            prediction.extrinsics,
            bldg_masks,
            prediction.conf,
            conf_thresh,
        )
        min_dim = _estimate_min_building_dimension(bldg_points, method="bbox")
        bldg_center = np.mean(bldg_points, axis=0)
        # logger.info(f"{len(bldg_points)=}")
        # logger.info(f"{min_dim=}")
        # logger.info(f"{bldg_center=}")
        
        if min_dim is not None:
            scale = float(target_min_bldg_dim_m / min_dim)
            if np.isfinite(scale) and scale > 0:
                prediction.extrinsics = apply_umeyama_alignment_to_ext(
                    np.eye(3, dtype=np.float64),
                    np.zeros(3, dtype=np.float64),
                    scale,
                    prediction.extrinsics,
                ).astype(np.float32)
                prediction.depth = (prediction.depth * scale).astype(np.float32)
                bldg_center *= scale
            else:
                logger.warn(f"Invalid scale, no scaling for this scene.")
        else:
            logger.warn(f"Invalid min_dim, no scaling for this scene.")

    # ===== STEP 2: Fit ground plane (RANSAC with properly scaled distances) =====
    if gnd_mask_paths is not None:
        plane = _fit_ground_plane_ransac(
            prediction,
            gnd_mask_paths=gnd_mask_paths,
            conf_thresh_percentile=conf_thresh_percentile,
            dist_thresh=0.05
        )
        logger.info(f"RANSAC returned plane object: {plane}")
        if plane is not None:
            center = project_point_to_plane(bldg_center, plane["normal"], plane["d"])
        else:
            center = None
    else:
        plane = None
        center = None

    if center is None:
        center = bldg_center
    if not np.isfinite(center).all():
        return prediction

    # ===== STEP 3: Orient plane normal toward cameras (in scaled space) =====
    # Camera positions and plane equation are in scaled metric units.
    rotation_matrix = None
    if rotate_to_ground and gnd_mask_paths is not None and plane is not None:
        n = np.asarray(plane["normal"], dtype=np.float64)
        d = float(plane["d"])
        
        # Extract camera centers from scaled extrinsics.
        # Extrinsics are w2c matrices: camera_center = -R^T @ t
        extrinsics_w2c = prediction.extrinsics.reshape(-1, 4, 4) if prediction.extrinsics.shape[-1] == 4 else \
                         np.pad(prediction.extrinsics.reshape(-1, 3, 4), ((0, 0), (0, 1), (0, 0)), constant_values=((0, 0), (0, 1), (0, 0)))
        R_w2c = extrinsics_w2c[:, :3, :3]
        t_w2c = extrinsics_w2c[:, :3, 3]
        R_c2w = R_w2c.transpose(0, 2, 1)  # (N, 3, 3)
        camera_positions = -(R_c2w @ t_w2c[:, :, np.newaxis]).squeeze(-1)  # (N, 3)
        
        # Orient plane normal toward cameras (fixes ambiguous normal direction).
        n, _ = _orient_plane_normal_toward_cameras(n, d, camera_positions)

        # Compute rotation matrix to align ground normal to world up axis.
        up = np.asarray(world_up_axis, dtype=np.float64)
        up_norm = np.linalg.norm(up)
        if up_norm > 1e-6:
            up = up / up_norm
            rotation_matrix = _rotation_matrix_from_vectors(n, up)

    # ===== STEP 4: Translate scene center to origin =====
    prediction.extrinsics = apply_umeyama_alignment_to_ext(
        np.eye(3, dtype=np.float64),
        -center,
        1.0,
        prediction.extrinsics,
    ).astype(np.float32)

    # ===== STEP 5: Rotate to align ground normal to world up =====
    if rotation_matrix is not None:
        prediction.extrinsics = apply_umeyama_alignment_to_ext(
            rotation_matrix,
            np.zeros(3, dtype=np.float64),
            1.0,
            prediction.extrinsics,
        ).astype(np.float32)

    return prediction
