#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# The below scripts are free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
# For inquiries contact  george.drettakis@inria.fr
#
import torch
import numpy as np

def ndc_2_cam(ndc_xyz, intrinsic, W, H):
    inv_scale = torch.tensor([[W - 1, H - 1]], device=ndc_xyz.device)
    cam_z = ndc_xyz[..., 2:3]
    cam_xy = ndc_xyz[..., :2] * inv_scale * cam_z
    cam_xyz = torch.cat([cam_xy, cam_z], dim=-1)
    cam_xyz = cam_xyz @ torch.inverse(intrinsic[0, ...].t())
    return cam_xyz

def depth2point_cam(sampled_depth, ref_intrinsic):
    B, N, C, H, W = sampled_depth.shape
    valid_z = sampled_depth
    valid_x = torch.arange(W, dtype=torch.float32, device=sampled_depth.device) / (W - 1)
    valid_y = torch.arange(H, dtype=torch.float32, device=sampled_depth.device) / (H - 1)
    valid_x, valid_y = torch.meshgrid(valid_x, valid_y, indexing='xy')
    
    valid_x = valid_x[None, None, None, ...].expand(B, N, C, -1, -1)
    valid_y = valid_y[None, None, None, ...].expand(B, N, C, -1, -1)
    ndc_xyz = torch.stack([valid_x, valid_y, valid_z], dim=-1).view(B, N, C, H, W, 3) 
    cam_xyz = ndc_2_cam(ndc_xyz, ref_intrinsic, W, H) 
    return ndc_xyz, cam_xyz

def depth2point_world(depth_image, intrinsic_matrix, extrinsic_matrix):
    # extrinsic_matrix here is W2C
    _, xyz_cam = depth2point_cam(depth_image[None,None,None,...], intrinsic_matrix[None,...])
    xyz_cam = xyz_cam.reshape(-1,3)
    
    # Check if extrinsic is 3x4 and pad it to 4x4
    if extrinsic_matrix.shape == (3, 4):
        bottom_row = torch.tensor([[0., 0., 0., 1.]], dtype=extrinsic_matrix.dtype, device=extrinsic_matrix.device)
        extrinsic_matrix = torch.cat([extrinsic_matrix, bottom_row], dim=0)
    elif extrinsic_matrix.shape != (4, 4):
        raise ValueError(f"Expected extrinsic matrix to be 3x4 or 4x4, but got {extrinsic_matrix.shape}")

    # Safely apply the extrinsic matrix now that we guarantee it is 4x4
    xyz_world = torch.cat([xyz_cam, torch.ones_like(xyz_cam[...,0:1])], axis=-1) @ torch.inverse(extrinsic_matrix).transpose(0,1)
    xyz_world = xyz_world[...,:3]

    return xyz_world

def depth_pcd2normal(xyz, offset=None):
    hd, wd, _ = xyz.shape 
    if offset is not None:
        ix, iy = torch.meshgrid(torch.arange(wd), torch.arange(hd), indexing='xy')
        xy = (torch.stack((ix, iy), dim=-1)[1:-1,1:-1]).to(xyz.device)
        p_offset = torch.tensor([[0,1],[0,-1],[1,0],[-1,0]]).float().to(xyz.device)
        new_offset = p_offset[None,None] + offset.reshape(hd, wd, 4, 2)[1:-1,1:-1]
        xys = xy[:,:,None] + new_offset
        xys[..., 0] = 2 * xys[..., 0] / (wd - 1) - 1.0
        xys[..., 1] = 2 * xys[..., 1] / (hd - 1) - 1.0
        sampled_xyzs = torch.nn.functional.grid_sample(xyz.permute(2,0,1)[None], xys.reshape(1, -1, 1, 2))
        sampled_xyzs = sampled_xyzs.permute(0,2,3,1).reshape(hd-2,wd-2,4,3)
        bottom_point = sampled_xyzs[:,:,0]
        top_point = sampled_xyzs[:,:,1]
        right_point = sampled_xyzs[:,:,2]
        left_point = sampled_xyzs[:,:,3]
    else:
        bottom_point = xyz[..., 2:hd,   1:wd-1, :]
        top_point    = xyz[..., 0:hd-2, 1:wd-1, :]
        right_point  = xyz[..., 1:hd-1, 2:wd,   :]
        left_point   = xyz[..., 1:hd-1, 0:wd-2, :]
        
    left_to_right = right_point - left_point
    bottom_to_top = top_point - bottom_point 
    xyz_normal = torch.cross(left_to_right, bottom_to_top, dim=-1)
    xyz_normal = torch.nn.functional.normalize(xyz_normal, p=2, dim=-1)
    xyz_normal = torch.nn.functional.pad(xyz_normal.permute(2,0,1), (1,1,1,1), mode='constant').permute(1,2,0)
    return xyz_normal

def normal_from_depth_image(depth, intrinsic_matrix, extrinsic_matrix, offset=None, device=None):
    """
    Wrapper to handle both NumPy arrays and PyTorch tensors, routing them to the 
    specified device (CPU or GPU) for processing.
    """
    # Auto-detect device if not provided
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    # Convert NumPy arrays to PyTorch tensors
    if isinstance(depth, np.ndarray):
        depth = torch.from_numpy(depth).float()
    if isinstance(intrinsic_matrix, np.ndarray):
        intrinsic_matrix = torch.from_numpy(intrinsic_matrix).float()
    if isinstance(extrinsic_matrix, np.ndarray):
        extrinsic_matrix = torch.from_numpy(extrinsic_matrix).float()
        
    # Move to the target device (GPU/CPU)
    depth = depth.to(device)
    intrinsic_matrix = intrinsic_matrix.to(device)
    extrinsic_matrix = extrinsic_matrix.to(device)

    # The core process
    xyz_world = depth2point_world(depth, intrinsic_matrix, extrinsic_matrix)
    xyz_world = xyz_world.reshape(*depth.shape, 3)
    xyz_normal = depth_pcd2normal(xyz_world, offset)

    return xyz_normal   # (H, W, 3)

def render_normals(
        depths,
        extrinsics,
        intrinsics,
        conf=None,
        conf_percentile=40.0,
        sky=None,
        device=None,
    ):
    """
    Render normal maps from depth maps.
    Mask rules:
    - conf below per-frame percentile -> (0,0,0)
    - sky == True -> (0,0,0)
    """
    normals = []

    for i, (depth, intrinsic, extrinsic) in enumerate(zip(depths, intrinsics, extrinsics)):
        # Compute normal (returns torch tensor)
        normal_torch = normal_from_depth_image(depth, intrinsic, extrinsic, device=device)
        normal = normal_torch.cpu().numpy()  # (H, W, 3)

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

        normal[~valid] = 0.0
        normals.append(normal)

    return np.stack(normals, axis=0)