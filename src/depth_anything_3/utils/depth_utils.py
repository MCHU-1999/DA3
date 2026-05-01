#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# The below scripts are free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
# For inquiries contact  george.drettakis@inria.fr
#
import numpy as np

def update_depths(
        depths,
        conf=None,
        conf_percentile=40.0,
        sky=None,
    ):
    """
    Re-render depth maps with masks
    Mask rules:
    - conf below per-frame percentile -> 500
    - sky == True -> 500
    """
    new_depths = []

    for i, depth in enumerate(depths):

        # Start with valid depth mask
        if isinstance(depth, np.ndarray):
            depth_arr = depth
        else:
            depth_arr = depth.cpu().numpy()
        valid = np.isfinite(depth_arr) & (depth_arr > 0)

        # Confidence mask: keep only top (100 - conf_percentile)% by confidence
        if conf is not None:
            conf_i = conf[i]
            if isinstance(conf_i, np.ndarray):
                conf_arr = conf_i
            else:
                conf_arr = conf_i.cpu().numpy()

            finite_conf = np.isfinite(conf_arr)
            if finite_conf.any():
                thr = np.quantile(conf_arr[finite_conf], conf_percentile / 100.0)
                valid = valid & finite_conf & (conf_arr >= thr)
            else:
                valid = np.zeros_like(valid, dtype=bool)

        # Sky mask: sky pixels should be zero normal
        if sky is not None:
            sky_i = sky[i]
            if isinstance(sky_i, np.ndarray):
                sky_arr = sky_i.astype(bool)
            else:
                sky_arr = sky_i.cpu().numpy().astype(bool)
            valid = valid & (~sky_arr)

        depth[~valid] = 500
        new_depths.append(depth)

    return np.stack(new_depths, axis=0)