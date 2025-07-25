import numpy as np
from robosuite.utils.numba import jit_decorator

# from pointnet2_ops import pointnet2_utils


def furthest_point_sampling(data, number):
    '''
    Args:
        data (torch.Tensor): (B, N, 3) tensor
        number: int
    Returns:
        fps_data: (B, number, 3) tensor
    '''
    fps_idx = pointnet2_utils.furthest_point_sample(data, number)
    fps_data = pointnet2_utils.gather_operation(data.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()
    return fps_data


@jit_decorator
def get_point_cloud_in_camera(depth_map, camera_matrix):
    """
    Convert batch of depth maps to point clouds using batched camera parameters.

    Args:
        depth_map (np.ndarray): Batch of depth images (B x H x W)
        camera_matrix (np.ndarray): Batch of camera intrinsic matrices (B x 3 x 3)

    Returns:
        np.ndarray: Point clouds as Bx3xN array where B is batch size, N = H*W is number of points per cloud
    """
    batch_size, height, width = depth_map.shape

    # Create pixel coordinate grid
    xx, yy = np.meshgrid(np.arange(width), np.arange(height))

    # Expand dimensions for broadcasting
    xx = np.tile(xx[np.newaxis, ...], (batch_size, 1, 1))  # B x H x W
    yy = np.tile(yy[np.newaxis, ...], (batch_size, 1, 1))  # B x H x W

    # Apply per-batch camera intrinsics, ensuring correct broadcasting
    fx = camera_matrix[:, 0, 0][:, None, None]  # B x 1 x 1
    fy = camera_matrix[:, 1, 1][:, None, None]  # B x 1 x 1
    cx = camera_matrix[:, 0, 2][:, None, None]  # B x 1 x 1
    cy = camera_matrix[:, 1, 2][:, None, None]  # B x 1 x 1

    xx = (xx - cx) / fx  # B x H x W
    yy = (yy - cy) / fy  # B x H x W

    # Convert to 3D points
    x_3d = xx * depth_map
    y_3d = yy * depth_map
    points = np.stack((x_3d, y_3d, depth_map), axis=-1)  # B x H x W x 3

    # Reshape to BxNx3
    points = points.reshape(batch_size, -1, 3)  # B x (H*W) x 3

    # Transform points to world frame
    points_homogeneous = np.concatenate(
        [points, np.ones((batch_size, points.shape[1], 1))], axis=-1)  # B x (H*W) x 4

    points = points_homogeneous[..., :3]  # B x (H*W) x 3
    points = np.transpose(points, (0, 2, 1))  # B x 3 x (H*W)

    return points  # B x 3 x N
