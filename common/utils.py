import torch
import numpy as np
import torch.nn.functional as F
import torch
import open3d as o3d
import matplotlib.pyplot as plt
import plotly.graph_objects as go


def visualize_pts(points, colors, save_path=None, save_rendered_path=None):
    
    if save_path:
        np.save(f"{save_path}xyz.npy", points.cpu().numpy())
        np.save(f"{save_path}rgb.npy", colors.cpu().numpy())
    
    points = points.cpu().numpy()
    fig = go.Figure(data=[go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode='markers',
        marker=dict(
            size=1.7,
            color=colors.cpu().numpy(),#(colors.cpu().numpy()*255).astype(int),  # Use RGB colors
            opacity=0.99
        ))])
    x_min, x_max = -2,2#points[:, 0].min(), points[:, 0].max()
    y_min, y_max = -2,2#points[:, 1].min(), points[:, 1].max()
    z_min, z_max = -2,2#points[:, 2].min(), points[:, 2].max()
    fig.update_layout(
        scene=dict(
            bgcolor='rgb(220, 220, 220)'  # Set the 3D scene background to light grey
        ),
        paper_bgcolor='rgb(220, 220, 220)' # Set the overall figure background to light grey
    )
    fig.update_layout(
        scene=dict(
            xaxis=dict(title='x', range=[x_min, x_max], showgrid=False, zeroline=False, visible=False),
            yaxis=dict(title='y', range=[y_min, y_max], showgrid=False, zeroline=False, visible=False),
            zaxis=dict(title='z', range=[z_min, z_max], showgrid=False, zeroline=False, visible=False),
            aspectmode='manual',
            aspectratio=dict(
                x=(x_max - x_min),
                y=(y_max - y_min),
                z=(z_max - z_min)
            )
        ),
        scene_camera=dict(
            up=dict(x=0, y=1, z=0),  # Adjust these values for your point cloud
            eye=dict(x=0, y=-2, z=0),  # Increase the values to move further away
            center = dict(x=0,y=0,z=0)
        )
    )
    
    if save_rendered_path:
        fig.write_image(save_rendered_path)
    else:
        fig.show()

def visualize_pt_labels(pts, labels, save_path=None, save_rendered_path=None): # pts is n*3, colors is n, 0 - n-1 where 0 is unlabeled
    part_num = labels.max()
    cmap_matrix = torch.tensor([[1,1,1], [1,0,0], [0,1,0], [0,0,1], [1,1,0], [1,0,1],
                [0,1,1], [0.5,0.5,0.5], [0.5,0.5,0], [0.5,0,0.5],[0,0.5,0.5],
                [0.1,0.2,0.3],[0.2,0.5,0.3], [0.6,0.3,0.2], [0.5,0.3,0.5],
                [0.6,0.7,0.2],[0.5,0.8,0.3]])[:part_num+1,:]
    colors = ["white", "red", "green", "blue", "yellow", "magenta", "cyan","grey", "olive",
                "purple", "teal", "navy", "darkgreen", "brown", "pinkpurple", "yellowgreen", "limegreen"]
    caption_list=[f"{i}:{colors[i]}" for i in range(part_num+1)]
    onehot = F.one_hot(labels.long(), num_classes=part_num+1) * 1.0 # n_pts, part_num+1, each row 00.010.0, first place is unlabeled (0 originally)
    pts_rgb = torch.matmul(onehot, cmap_matrix) # n_pts,3
    visualize_pts(pts, pts_rgb, save_path=save_path, save_rendered_path=save_rendered_path)
    print(caption_list)


def visualize_pt_heatmap(pts, scores, save_path=None): # pts is n*3, scores shape (n,) and is a value between 0 and 1
    pts_rgb = torch.tensor(plt.cm.jet(scores.numpy())[:,:3]).squeeze()
    visualize_pts(pts, pts_rgb, save_path=save_path)


def visualize_pts_subsampled(pts, colors, n_samples):
    perm = torch.randperm(n_samples)
    idx = perm[:n_samples]
    subsampled_pts = pts[idx,:]
    subsampled_colors = colors[idx,:]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(subsampled_pts.numpy())
    pcd.colors = o3d.utility.Vector3dVector(subsampled_colors)
    o3d.visualization.draw_plotly([pcd],
                                  front=[0, 0, -1],
                                  lookat=[0, 0, -1],
                                  up=[0, 1, 0])


def rotate_pts(pts, angles, device=None): # list of points as a tensor, N*3

    roll = angles[0].reshape(1)
    yaw = angles[1].reshape(1)
    pitch = angles[2].reshape(1)

    tensor_0 = torch.zeros(1).to(device)
    tensor_1 = torch.ones(1).to(device)

    RX = torch.stack([
                    torch.stack([tensor_1, tensor_0, tensor_0]),
                    torch.stack([tensor_0, torch.cos(roll), -torch.sin(roll)]),
                    torch.stack([tensor_0, torch.sin(roll), torch.cos(roll)])]).reshape(3,3)

    RY = torch.stack([
                    torch.stack([torch.cos(yaw), tensor_0, torch.sin(yaw)]),
                    torch.stack([tensor_0, tensor_1, tensor_0]),
                    torch.stack([-torch.sin(yaw), tensor_0, torch.cos(yaw)])]).reshape(3,3)

    RZ = torch.stack([
                    torch.stack([torch.cos(pitch), -torch.sin(pitch), tensor_0]),
                    torch.stack([torch.sin(pitch), torch.cos(pitch), tensor_0]),
                    torch.stack([tensor_0, tensor_0, tensor_1])]).reshape(3,3)

    R = torch.mm(RZ, RY)
    R = torch.mm(R, RX)
    if device == "cuda":
        R = R.cuda()
    pts_new = torch.mm(pts, R.T)
    return pts_new