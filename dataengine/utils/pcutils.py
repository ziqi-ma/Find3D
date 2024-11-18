# rendering and normalization code from https://github.com/zyc00/PartSLIP2/blob/partslip%2B%2B/src/render_pc.py
from pytorch3d.renderer import (
    look_at_view_transform,
    PerspectiveCameras,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    NormWeightedCompositor
)
import torch
import torch.nn as nn


class PointRendererWithDepth(nn.Module):
    def __init__(self, rasterizer, compositor):
        super().__init__()
        self.rasterizer = rasterizer
        self.compositor = compositor

    def forward(self, point_clouds, **kwargs) -> torch.Tensor:
        fragments = self.rasterizer(point_clouds, **kwargs)
        # Construct weights based on the distance of a point to the true point.
        # However, this could be done differently: e.g. predicted as opposed
        # to a function of the weights.
        r = self.rasterizer.raster_settings.radius
        dists2 = fragments.dists.permute(0, 3, 1, 2)
        weights = 1 - dists2 / (r * r)
        images = self.compositor(
            fragments.idx.long().permute(0, 3, 1, 2),
            weights,
            point_clouds.features_packed().permute(1, 0),
            **kwargs,
        )
        # permute so image comes at the end
        images = images.permute(0, 2, 3, 1)
        return images, fragments.zbuf


def render_single_view(pc, view, device, background_color=(1,1,1), resolution=800, camera_distance=2.2, 
                        point_size=0.005, points_per_pixel=5, bin_size=0, znear=0.01):
    R, T = look_at_view_transform(camera_distance, view[0], view[1])
    
    cameras = PerspectiveCameras(device=device, R=R, T=T)#, znear=znear)
    extrinsic_matrix = (cameras.get_world_to_view_transform().get_matrix().squeeze()).T
    intrinsic_matrix = cameras.get_projection_transform().get_matrix().squeeze()
    # change to the correct form of 
    #K = [
    #                [fx,   0,   px,   0],
    #                [0,   fy,   py,   0],
    #                [0,    0,    1,   0],
    #                [0,    0,    0,   1],
    #        ]
    intrinsic_matrix[2,3] = 0
    intrinsic_matrix[3,2] = 0
    intrinsic_matrix[2,2] = 1
    intrinsic_matrix[3,3] = 1

    raster_settings = PointsRasterizationSettings(
        image_size=resolution, 
        radius=point_size,
        points_per_pixel=points_per_pixel,
        bin_size=bin_size,
    )
    compositor=NormWeightedCompositor(background_color=background_color)
    
    rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
    renderer = PointsRenderer(
        rasterizer=rasterizer,
        compositor=compositor
    )
    img = renderer(pc)
    pc_idx = rasterizer(pc).idx
    screen_coords = cameras.transform_points_screen(pc._points_list[0], image_size=(resolution, resolution))
    return img, intrinsic_matrix.cpu(), extrinsic_matrix.cpu(), pc_idx, screen_coords


def render_single_view_with_depth(pc, view, device, background_color=(1,1,1), resolution=800, camera_distance=2.2, 
                        point_size=0.005, points_per_pixel=5, bin_size=0, znear=0.01):
    R, T = look_at_view_transform(camera_distance, view[0], view[1])
    
    cameras = PerspectiveCameras(device=device, R=R, T=T)#, znear=znear)
    extrinsic_matrix = (cameras.get_world_to_view_transform().get_matrix().squeeze()).T
    intrinsic_matrix = cameras.get_projection_transform().get_matrix().squeeze()
    # change to the correct form of 
    #K = [
    #                [fx,   0,   px,   0],
    #                [0,   fy,   py,   0],
    #                [0,    0,    1,   0],
    #                [0,    0,    0,   1],
    #        ]
    intrinsic_matrix[2,3] = 0
    intrinsic_matrix[3,2] = 0
    intrinsic_matrix[2,2] = 1
    intrinsic_matrix[3,3] = 1

    raster_settings = PointsRasterizationSettings(
        image_size=resolution, 
        radius=point_size,
        points_per_pixel=points_per_pixel,
        bin_size=bin_size,
    )
    compositor=NormWeightedCompositor(background_color=background_color)
    rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
    renderer = PointRendererWithDepth(
        rasterizer=rasterizer,
        compositor=compositor
    )
    img, depth = renderer(pc)
    depth = depth[:,:,:,0]
    return img, depth, intrinsic_matrix.cpu(), extrinsic_matrix.cpu()
    

def normalize_pc(pc_file, io, device):
    pc = io.load_pointcloud(pc_file, device = device)
    xyz = pc.points_padded().reshape(-1,3)
    rgb = pc.features_padded().reshape(-1,3)
    xyz = xyz - xyz.mean(axis=0)
    xyz = xyz / torch.norm(xyz, dim=1, p=2).max().item()
    xyz = xyz.cpu().numpy()
    rgb = rgb.cpu().numpy()
    return xyz, rgb