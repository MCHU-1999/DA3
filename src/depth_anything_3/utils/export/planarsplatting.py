import os
import numpy as np
import cv2

from depth_anything_3.specs import Prediction
from depth_anything_3.utils.parallel_utils import async_call


@async_call
def export_to_planarsplatting(
    prediction: Prediction,
    export_dir: str,
    img_name_list: list = [],
    img_res: list[int] = []     # [h, w] - original image resolution
):
    if len(img_name_list) == 0:
        raise Exception("When calling export_to_planarsplatting(), the parameter `img_name_list` is needed.")
    if len(img_res) != 2:
        raise Exception("When calling export_to_planarsplatting(), the parameter `img_res` should be [height, width].")

    depth_dir = os.path.join(export_dir, "DA3_depth")
    os.makedirs(depth_dir, exist_ok=True)
    normal_dir = os.path.join(export_dir, "DA3_normal") 
    os.makedirs(normal_dir, exist_ok=True)

    # Original resolution
    target_h, target_w = img_res

    # Save each map separately
    for i, img_name in enumerate(img_name_list):
        depth_file = os.path.join(depth_dir, f"{img_name}.npy")
        normal_file = os.path.join(normal_dir, f"{img_name}.npy")

        # Upscale depth to original resolution
        this_depth = prediction.depth[i]  # (H_proc, W_proc)
        depth_upscaled = cv2.resize(
            this_depth, 
            (target_w, target_h), 
            interpolation=cv2.INTER_LINEAR
        )
        np.save(depth_file, depth_upscaled)

        # Upscale normals to original resolution
        this_normal = prediction.normal[i]  # (H_proc, W_proc, 3)
        normal_upscaled = cv2.resize(
            this_normal, 
            (target_w, target_h), 
            interpolation=cv2.INTER_LINEAR
        )
        # Renormalize after interpolation (important for normals!)
        norms = np.linalg.norm(normal_upscaled, axis=-1, keepdims=True)
        norms[norms == 0] = 1
        normal_upscaled = normal_upscaled / norms
        
        np.save(normal_file, normal_upscaled)
