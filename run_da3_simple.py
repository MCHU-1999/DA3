import glob, os, torch, sys
import numpy as np
from depth_anything_3.api import DepthAnything3
from colmap_io import read_intrinsics_binary, read_extrinsics_binary, qvec2rotmat
from colmap_io import read_intrinsics_text, read_extrinsics_text
import argparse

from typing import NamedTuple, List, Dict

def print_err(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

class ColmapDataset(NamedTuple):
    extrinsics_list: np.ndarray     # (N, 4, 4)
    intrinsics_list: np.ndarray     # (N, 3, 3)
    img_paths_list: np.ndarray | List[str]
    img_name_list: np.ndarray | List[str]
    width: int
    height: int
    N: int

def read_colmap_dataset(data_dir):
    ##
    # The important checks
    image_dir = os.path.join(data_dir, "images")
    if not os.path.exists(image_dir):
        raise ValueError(f'The input path {image_dir} does not exist.')

    colmap_cam_file_path = os.path.join(data_dir, 'sparse/0/cameras.bin')
    if not os.path.exists(colmap_cam_file_path):
        colmap_cam_file_path = os.path.join(data_dir, 'sparse/0/cameras.txt')
        if not os.path.exists(colmap_cam_file_path):
            raise ValueError(f'The input path {colmap_cam_file_path} does not exist.')
    
    colmap_image_file_path = os.path.join(data_dir, 'sparse/0/images.bin')
    if not os.path.exists(colmap_image_file_path):
        colmap_image_file_path = os.path.join(data_dir, 'sparse/0/images.txt')
        if not os.path.exists(colmap_image_file_path):
            raise ValueError(f'The input path {colmap_image_file_path} does not exist.')
    
    ##
    # Read intrinsics
    if colmap_cam_file_path.endswith(".bin"):
        cameras = read_intrinsics_binary(colmap_cam_file_path)
    else: 
        cameras = read_intrinsics_text(colmap_cam_file_path)

    camera = next(iter(cameras.values()))   # assume all images were captured with the same camera.
    fx, fy, cx, cy = camera.params[:4]
    intrinsic = np.array([[fx, 0., cx],
                          [0., fy, cy],
                          [0., 0., 1.0]]).astype(np.float32)
    h = camera.height
    w = camera.width
    
    if colmap_image_file_path.endswith(".bin"):
        images_meta = read_extrinsics_binary(colmap_image_file_path)
    else:
        images_meta = read_extrinsics_text(colmap_image_file_path)

    if len(images_meta) == 0:
        raise ValueError(f"No valid images found in {image_dir}")
    
    img_paths_list = []
    extrinsics_list = []
    intrinsics_list = []
    img_name_list = []

    for img_id, img_meta in images_meta.items():
        frame_name: str = img_meta.name
        frame_path = os.path.join(image_dir, frame_name)

        q = img_meta.qvec
        t = img_meta.tvec
        r = qvec2rotmat(q)
        rt = np.eye(4)
        rt[:3,:3] = r
        rt[:3, 3] = t
        
        # Before (wrong):
        # c2w = np.linalg.inv(rt).astype(np.float32)
        # extrinsics_list.append(c2w)
        # After (correct):
        extrinsics_list.append(rt.astype(np.float32))  # rt is already w2c

        intrinsics_list.append(intrinsic)
        img_paths_list.append(frame_path)
        img_name_list.append(frame_name.rsplit('.', 1)[0])

    dataset = ColmapDataset(
        extrinsics_list=np.stack(extrinsics_list, axis=0),
        intrinsics_list=np.stack(intrinsics_list, axis=0),
        img_paths_list=img_paths_list,
        img_name_list=img_name_list,
        width=w,
        height=h,
        N=len(img_paths_list)
    )
    
    return dataset
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-d", "--data_dir", type=str, default='path/to/colmap/data', help='path of input colmap data')
    args = parser.parse_args()

    DATA_DIR = args.data_dir

    # Setup model and device
    device = torch.device("cuda")
    model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE")
    model = model.to(device=device)

    dataset = read_colmap_dataset(DATA_DIR)
    # Export depth data and 3D visualization
    prediction = model.inference(
        image=dataset.img_paths_list,
        extrinsics=dataset.extrinsics_list,
        # extrinsics=None,
        intrinsics=dataset.intrinsics_list,
        # intrinsics=None,
        export_dir=DATA_DIR,
        export_format="planarsplatting-colmap",
        # export_format="planarsplatting-colmap-mesh",
        # export_format="planarsplatting-colmap-glb",
        process_res=420,
        # process_res=840,
        process_res_method="upper_bound_resize",
        export_kwargs={
            "planarsplatting": {
                "img_name_list": dataset.img_name_list,
                "img_res": [dataset.height, dataset.width]
            }
        },
        show_cameras=False,
        # show_cameras=True,
        conf_thresh_percentile=40
    )

    # prediction.processed_images : [N, H, W, 3] uint8   array
    print_err("processed_images.shape: ", prediction.processed_images.shape)
    # prediction.depth            : [N, H, W]    float32 array
    print_err("depth.shape: ", prediction.depth.shape)  
    # prediction.conf             : [N, H, W]    float32 array
    print_err("conf.shape: ", prediction.conf.shape)  
    # prediction.extrinsics       : [N, 3, 4]    float32 array # opencv w2c or colmap format
    print_err("extrinsics.shape: ", prediction.extrinsics.shape)
    # prediction.intrinsics       : [N, 3, 3]    float32 array
    print_err("intrinsics.shape: ", prediction.intrinsics.shape)