# for usage see
# training: model.training.loss DistillLossContrastive
# evaluate: model.evaluation.core compute_overall_iou_objwise (this is iou per mask)

# this creates under
# e.g. [name]_uid/norotate/masks/merged
# - mask2points.pt (this is n_masks*n_pts binary)
import os
import torch
import os.path as osp
import time
from tqdm import tqdm
from common.utils import visualize_pts
import matplotlib.pyplot as plt
import pandas as pd
from dataengine.configs import DATA_ROOT

def label_mask2pt(obj_dir, uid):
    # root_dir is e.g. [name]_uid/norotate
    if not os.path.exists(f"{obj_dir}/masks/merged"): # if no mask, skip
        return
    if len(os.listdir(f"{obj_dir}/masks/merged")) == 0:
        return
    if os.path.exists(f"{obj_dir}/masks/merged/mask2points.pt"): # if already labeled, skip
        pass
    # get per mask points
    pix2frontface = torch.load(osp.join(obj_dir, "pix2face.pt")).cuda() # pix2frontface is n_view, h, w and the value is face index
    point2face = torch.load(f"{DATA_ROOT}/labeled/points/{uid}/point2face.pt").cuda() # point2face is of size 5000, each a face index for the point
    all_masks = torch.load(f"{obj_dir}/masks/merged/allmasks.pt").cuda()
    all_masks = all_masks.view(all_masks.shape[0],-1).type(torch.cuda.FloatTensor)
    mask2view = torch.load(f"{obj_dir}/masks/merged/mask2view.pt").cuda()
    n_views = 10

    mask2pt_list = []
    # for each view, for now we don't parallelize since io needs to be sequential anyway
    # all the labels are in order from view0 to view10
    # so by concatenating by filtered view idx in order we preserve the original order
    for i in range(n_views):
        # for each pixel, if the face index is in the point, write point_feats[idx, i, :] with the feature
        pix2face = pix2frontface[i,:,:].view(-1) 
        # get flattened point idx->pix matrix
        # this is n_pts * n_pixels like below 
        # [0 0 1 ... 1 1 0 0 ... 1] each 1 is the pixels that show the same face as this point (0'th point)
        # [0 0 0  1 ...0 1 ... 0 0] each 1 is the pixels that show the same face as this point (1st point)
        # ...
        # faces with no corresponding points are ignored
        # points with no corresponding faces seem in this view is all 0
        point2pix = ((pix2face - point2face.unsqueeze(0).T) == 0) * 1.0 # n_pts, (h*w)
        pix2point = point2pix.T # (h*w),n_pts
        # for each mask in this view
        masks_this_view = all_masks[mask2view==i,:] # k,(h*w)
        # [0 0 1 1 .. 1 0 0 0]
        # [0 1 0 0 .. 1 0 0 0] pixels chosen for this mask
        # right now there is no weighting, e.g. if a mask covers 2 faces 1 super large 1 super small
        # we are taking all points on the large face and small face and eventually averaging them
        # later we could also add a weighting, i.e. inverse to the number of points sampled for each pixel
        # this will downweigh the points on the large face and upweigh the points  on the small face
        masks_pts = masks_this_view @ pix2point # k, n_pts, all pixels for this mask' corresponding pts > 0
        masks_pts_binary = (masks_pts>0)*1 # k, n_pts
        mask2pt_list.append(masks_pts_binary)
    
    mask2pt = torch.cat(mask2pt_list, dim=0) # this should be n_masks, n_pts
    torch.save(mask2pt, f"{obj_dir}/masks/merged/mask2points.pt")

# for debugging
def visualize_mask_pts(obj_dir, uid):
    mask2points = torch.load(f"{obj_dir}/masks/merged/mask2points.pt")
    allmasks = torch.load(f"{obj_dir}/masks/merged/allmasks.pt")
    f = open(f"{obj_dir}/masks/merged/mask_labels.txt", "r")
    labels = f.read().splitlines()
    f.close()
    pt_xyz = torch.load(f"{DATA_ROOT}/labeled/points/{uid}/points.pt")

    rand_indices = [10,13,36]

    for idx in rand_indices:
        print(labels[idx])
        mask = allmasks[idx,:,:]
        plt.imshow(mask)
        plt.savefig(f"viz/{labels[idx]}")
        mask2points_idx = mask2points[idx,:] # binary of (n_pts,)
        
        # mark these points purple
        rgb_r = 1-(mask2points_idx * 0.5).view(-1,1)
        rgb_g = 1-mask2points_idx.view(-1,1)
        rgb_b = 1-(mask2points_idx * 0.5).view(-1,1)
        rgb_all = torch.cat([rgb_r,rgb_g, rgb_b], dim=1)
        visualize_pts(pt_xyz, rgb_all, save_path=f"viz/pc{labels[idx]}")



if __name__ == "__main__":
    chunk_id = 0 # change this to process all chunks

    parent_folder = f"{DATA_ROOT}/labeled/rendered"
    cur_df = pd.read_csv(f"{DATA_ROOT}/labeled/chunk_ids/chunk{chunk_id}.csv")
    cur_df["path"] = cur_df["class"] + "_" + cur_df["uid"]
    child_dirs = cur_df["path"].tolist()

    start = time.time()
    for nameuid in tqdm(child_dirs):
        full_dir = parent_folder+"/"+nameuid
        uid = nameuid.split("_")[-1]
        label_mask2pt(f"{full_dir}/oriented", uid)
        # visualize for debugging
        # visualize_mask_pts(f"{full_dir}/oriented", uid)
    end = time.time()