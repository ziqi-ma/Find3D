import torch
from types import SimpleNamespace
from model.backbone.pt3.model import PointSemSeg
import numpy as np
import random
from transformers import AutoTokenizer, AutoModel
import open3d as o3d
from common.utils import visualize_pts

def load_model(checkpoint_path):
    args = SimpleNamespace()
    model = PointSemSeg(args=args, dim_output=768)
    model.load_state_dict(torch.load(checkpoint_path)["model_state_dict"])
    model.eval()
    model = model.cuda()
    return model

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def fnv_hash_vec(arr):
    """
    FNV64-1A
    """
    assert arr.ndim == 2
    # Floor first for negative coordinates
    arr = arr.copy()
    arr = arr.astype(np.uint64, copy=False)
    hashed_arr = np.uint64(14695981039346656037) * np.ones(
        arr.shape[0], dtype=np.uint64
    )
    for j in range(arr.shape[1]):
        hashed_arr *= np.uint64(1099511628211)
        hashed_arr = np.bitwise_xor(hashed_arr, arr[:, j])
    return hashed_arr


def grid_sample_numpy(xyz, rgb, normal, grid_size): # this should hopefully be 5000 or close
    xyz = xyz.cpu().numpy()
    rgb = rgb.cpu().numpy()
    normal = normal.cpu().numpy()

    scaled_coord = xyz / np.array(grid_size)
    grid_coord = np.floor(scaled_coord).astype(int)
    min_coord = grid_coord.min(0)
    grid_coord -= min_coord
    scaled_coord -= min_coord
    min_coord = min_coord * np.array(grid_size)
    key = fnv_hash_vec(grid_coord)
    idx_sort = np.argsort(key)
    key_sort = key[idx_sort]
    _, inverse, count = np.unique(key_sort, return_inverse=True, return_counts=True)
    idx_select = (
        np.cumsum(np.insert(count, 0, 0)[0:-1])
        + np.random.randint(0, count.max(), count.size) % count
    )
    idx_unique = idx_sort[idx_select]

    grid_coord = grid_coord[idx_unique]
    
    xyz = torch.tensor(xyz[idx_unique]).cuda()
    rgb = torch.tensor(rgb[idx_unique]).cuda()
    normal = torch.tensor(normal[idx_unique]).cuda()
    grid_coord = torch.tensor(grid_coord).cuda()

    return xyz, rgb, normal, grid_coord
    

def preprocess_pcd(xyz, rgb, normal): # rgb should be 0-1
    assert rgb.max() <=1
    # normalize
    # this is the same preprocessing I do before training
    center = xyz.mean(0)
    scale = max((xyz - center).abs().max(0)[0])
    xyz -= center
    xyz *= (0.75 / float(scale)) # put in 0.75-size box

    # axis swap
    xyz = torch.cat([-xyz[:,0].reshape(-1,1), xyz[:,2].reshape(-1,1), xyz[:,1].reshape(-1,1)], dim=1)

    # center shift
    xyz_min = xyz.min(dim=0)[0]
    xyz_max = xyz.max(dim=0)[0]
    xyz_max[2] = 0
    shift = (xyz_min+xyz_max)/2
    xyz -= shift

    # subsample/upsample to 5000 pts for grid sampling
    if xyz.shape[0] != 5000:
        random_indices = torch.randint(0, xyz.shape[0], (5000,))
        pts_xyz_subsampled = xyz[random_indices]
        pts_rgb_subsampled = rgb[random_indices]
        normal_subsampled = normal[random_indices]
    else:
        pts_xyz_subsampled = xyz
        pts_rgb_subsampled = rgb
        normal_subsampled = normal

    # grid sampling
    pts_xyz_gridsampled, pts_rgb_gridsampled, normal_gridsampled, grid_coord = grid_sample_numpy(pts_xyz_subsampled, pts_rgb_subsampled, normal_subsampled, 0.02)

    # another center shift, z=false
    xyz_min = pts_xyz_gridsampled.min(dim=0)[0]
    xyz_min[2] = 0
    xyz_max = pts_xyz_gridsampled.max(dim=0)[0]
    xyz_max[2] = 0
    shift = (xyz_min+xyz_max)/2
    pts_xyz_gridsampled -= shift
    xyz -= shift

    # normalize color
    pts_rgb_gridsampled = pts_rgb_gridsampled / 0.5 - 1

    # combine color and normal as feat
    feat = torch.cat([pts_rgb_gridsampled, normal_gridsampled], dim=1)

    data_dict = {}
    data_dict["coord"] = pts_xyz_gridsampled
    data_dict["feat"] = feat
    data_dict["grid_coord"] = grid_coord
    data_dict["xyz_full"] = xyz
    data_dict["offset"] = torch.tensor([pts_xyz_gridsampled.shape[0]])
    return data_dict


def encode_text(texts):
    siglip = AutoModel.from_pretrained("google/siglip-base-patch16-224") # dim 768 #"google/siglip-so400m-patch14-384")
    tokenizer = AutoTokenizer.from_pretrained("google/siglip-base-patch16-224")#"google/siglip-so400m-patch14-384")
    inputs = tokenizer(texts, padding="max_length", return_tensors="pt")
    for key in inputs:
        inputs[key] = inputs[key].cuda()
    with torch.no_grad():
        text_feat = siglip.cuda().get_text_features(**inputs)
    text_feat = text_feat / (text_feat.norm(dim=-1, keepdim=True) + 1e-12)
    return text_feat


def read_pcd(obj_path, visualize=True):
    pcd = o3d.io.read_point_cloud(obj_path)
    if visualize:
        visualize_pts(torch.tensor(np.asarray(pcd.points)), torch.tensor(np.asarray(pcd.colors)), save_path="actual")
    xyz = torch.tensor(np.asarray(pcd.points)).float()
    rgb = torch.tensor(np.asarray(pcd.colors)).float()
    normal = torch.tensor(np.asarray(pcd.normals)).float()
    return xyz, rgb, normal