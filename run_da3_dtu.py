import glob, os
import numpy as np
from depth_anything_3.api import DepthAnything3

from typing import NamedTuple, List, Dict


class DtuDataset(NamedTuple):
    extrinsics_list: np.ndarray     # (N, 4, 4)
    intrinsics_list: np.ndarray     # (N, 3, 3)
    img_paths_list: np.ndarray | List[str]
    img_name_list: np.ndarray | List[str]
    width: int = 1554
    height: int = 1162
    N: int = 49

def read_dtu_dataset(data_dir, camera_npz_path):
    ##
    # The important checks
    image_dir = os.path.join(data_dir, "images")
    if not os.path.exists(image_dir):
        raise ValueError(f'The input path {image_dir} does not exist.')
    
    if not os.path.exists(camera_npz_path):
        raise ValueError(f'The input path {camera_npz_path} does not exist.')
    
    ##
    # Read intrinsics and extrinsics
    camera_params = np.load(camera_npz_path)
    intrinsic = camera_params['K']
    extrinsics_list = camera_params['extrinsics']
    
    # Read and sort images from directory based on int(filename)
    img_files = glob.glob(os.path.join(image_dir, "*.png"))
    img_files = sorted(img_files, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

    img_paths_list = img_files
    img_name_list = [os.path.basename(f) for f in img_files]
    
    # Create intrinsics list by repeating intrinsic matrix for each image
    num_images = len(img_files)
    intrinsics_list = [intrinsic for _ in range(num_images)]

    dataset = DtuDataset(
        extrinsics_list=np.stack(extrinsics_list, axis=0),
        intrinsics_list=np.stack(intrinsics_list, axis=0),
        img_paths_list=img_paths_list,
        img_name_list=img_name_list,
        N=num_images
    )
    
    return dataset
    

if __name__ == "__main__":
    raise NotImplementedError("This script is not implemented yet.")