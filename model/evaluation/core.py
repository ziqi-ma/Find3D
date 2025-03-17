# evaluate and visualize
import torch
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F
from common.utils import visualize_pt_labels, visualize_pt_heatmap
import numpy as np
from utils import preprocess_pcd


def batch_iou(mask1, mask2): # both mask1 and mask2 are binary, batched and flattened
    # of shape (BS, H*W)
    union_binary = ((mask1 + mask2)>0)*1 # cur_view_n_masks, (H*W)
    union_area = union_binary.sum(dim=1)
    intersection_binary = mask1 * mask2 # cur_view_n_masks, (H*W)
    intersection_area = intersection_binary.sum(dim=1)
    iou = intersection_area / (union_area+1e-12)
    return iou


def compute_3d_iou_upsample(
        pred, # n_subsampled_pts, feat_dim
        part_text_embeds, # n_parts, feat_dim
        temperature,
        cat,
        xyz_sub,
        xyz_full, # n_pts, 3
        gt_full, # n_pts,
        panoptic = False,
        N_CHUNKS=1,
        visualize_seg=False,
        visualize_all_heatmap=False,
        xyz_visualization=None
        ):
    xyz_full = xyz_full.squeeze()
    # first get each point's logits
    logits = pred @ part_text_embeds.T # n_pts, n_mask

    
    if panoptic:
        pred_softmax = torch.nn.Softmax(dim=1)(logits * temperature)
    else:
        # prepend 0 as no label since if all queries are negative we should report no label
        logits_prepend0 = torch.cat([torch.zeros(logits.shape[0],1).cuda(), logits],axis=1)
        pred_softmax = torch.nn.Softmax(dim=1)(logits_prepend0 * temperature)
    
    # assign to nearest neighbor
    chunk_len = xyz_full.shape[0]//N_CHUNKS+1
    closest_idx_list = []
    for i in range(N_CHUNKS):
        cur_chunk = xyz_full[chunk_len*i:chunk_len*(i+1)]
        dist_all = (xyz_sub.unsqueeze(0) - cur_chunk.cuda().unsqueeze(1))**2 # 300k,5k,3
        cur_dist = (dist_all.sum(dim=-1))**0.5 # 300k,5k
        min_idxs = torch.min(cur_dist, 1)[1]
        del cur_dist
        closest_idx_list.append(min_idxs)
    all_nn_idxs = torch.cat(closest_idx_list,axis=0)
    all_probs = pred_softmax[all_nn_idxs]
    
    # now argmax
    if panoptic:
        pred_full = all_probs.argmax(dim=1).cpu() + 1# here, no unlabeled, 1,...n_part correspond to actual part assignment
    else:
        pred_full = all_probs.argmax(dim=1).cpu()# here, 0 is unlabeled, 1,...n_part correspond to actual part assignment
    
    acc = ((pred_full == gt_full)*1).sum() / pred_full.shape[0]
    pred_np = pred_full.numpy()
    label_np = gt_full.squeeze().numpy()

    if visualize_seg:
        # visualize on original scale xyz
        visualize_pt_labels(xyz_visualization.cpu(), gt_full.squeeze().cpu(), save_path=f"{cat}_gt")
        visualize_pt_labels(xyz_visualization.cpu(), pred_full.cpu(), save_path=f"{cat}_pred")

    if visualize_all_heatmap:
        xyz_full_ori_axis = torch.cat([-xyz_full[:,0].reshape(-1,1), xyz_full[:,2].reshape(-1,1), xyz_full[:,1].reshape(-1,1)], dim=1)
        for i in range(all_probs.shape[1]-1): # all queries
            cur_scores = all_probs[:,i+1] # skip 0 which corresponds to unlabeled
            visualize_pt_heatmap(xyz_full_ori_axis.cpu(), cur_scores.cpu(), save_path=f"{cat}_heatmap{i}")

    # get full iou
    part_ious = []
    for part in range(part_text_embeds.shape[0]):
        I = np.sum(np.logical_and(pred_np == part+1, label_np == part+1))
        U = np.sum(np.logical_or(pred_np == part+1, label_np == part+1))
        if U == 0:
            pass
        else:
            iou = I / float(U)
            part_ious.append(iou)
    full_miou = np.mean(part_ious)
    full_macc = acc.item()
    return full_miou, full_macc


def visualize_3d_upsample(
        pred, # n_subsampled_pts, feat_dim
        part_text_embeds, # n_parts, feat_dim
        temperature,
        xyz_sub,
        xyz_full, # n_pts, 3
        panoptic = False,
        N_CHUNKS=1,
        heatmap = False # heatmap or segmentation
        ):
    xyz_full = xyz_full.squeeze()
    logits = pred @ part_text_embeds.T # n_pts, n_mask

    if panoptic:
        pred_softmax = torch.nn.Softmax(dim=1)(logits * temperature)
    else:
        # prepend 0 as no label
        logits_prepend0 = torch.cat([torch.zeros(logits.shape[0],1).cuda(), logits],axis=1)
        pred_softmax = torch.nn.Softmax(dim=1)(logits_prepend0 * temperature)
    
    chunk_len = xyz_full.shape[0]//N_CHUNKS+1
    closest_idx_list = []
    for i in range(N_CHUNKS):
        cur_chunk = xyz_full[chunk_len*i:chunk_len*(i+1)]
        dist_all = (xyz_sub.unsqueeze(0) - cur_chunk.cuda().unsqueeze(1))**2 # 300k,5k,3
        cur_dist = (dist_all.sum(dim=-1))**0.5 # 300k,5k
        min_idxs = torch.min(cur_dist, 1)[1]
        del cur_dist
        closest_idx_list.append(min_idxs)
    all_nn_idxs = torch.cat(closest_idx_list,axis=0)
    # just inversely weight all points
    all_probs = pred_softmax[all_nn_idxs]
    all_logits = logits[all_nn_idxs]
    
    # now argmax
    if panoptic:
        pred_full = all_probs.argmax(dim=1).cpu() + 1# here, no unlabeled, 1,...n_part correspond to actual part assignment
    else:
        pred_full = all_probs.argmax(dim=1).cpu()# here, 0 is unlabeled, 1,...n_part correspond to actual part assignment
    
    # convert back to the original coordinate system for visualization
        xyz_full_ori_axis = torch.cat([-xyz_full[:,0].reshape(-1,1), xyz_full[:,2].reshape(-1,1), xyz_full[:,1].reshape(-1,1)], dim=1)
    if heatmap:
        for i in range(all_probs.shape[1]-1): # all queries
            cur_scores = all_logits[:,i]
            visualize_pt_heatmap(xyz_full_ori_axis.cpu(), cur_scores.cpu())
    else: # segmentation
        visualize_pt_labels(xyz_full_ori_axis.cpu(), pred_full.cpu())
    return

def visualize_3d_upsample_render(
        pred, # n_subsampled_pts, feat_dim
        part_text_embeds, # n_parts, feat_dim
        temperature,
        xyz_sub,
        xyz_full, # n_pts, 3
        xyz_visualization,
        img_save_path,
        panoptic = False,
        N_CHUNKS=1,
        heatmap = False # heatmap or segmentation
        ):
    xyz_full = xyz_full.squeeze()
    logits = pred @ part_text_embeds.T # n_pts, n_mask

    if panoptic:
        pred_softmax = torch.nn.Softmax(dim=1)(logits * temperature)
    else:
        # prepend 0 as no label
        logits_prepend0 = torch.cat([torch.zeros(logits.shape[0],1).cuda(), logits],axis=1)
        pred_softmax = torch.nn.Softmax(dim=1)(logits_prepend0 * temperature)

    chunk_len = xyz_full.shape[0]//N_CHUNKS+1
    closest_idx_list = []
    for i in range(N_CHUNKS):
        cur_chunk = xyz_full[chunk_len*i:chunk_len*(i+1)]
        dist_all = (xyz_sub.unsqueeze(0) - cur_chunk.cuda().unsqueeze(1))**2 # 300k,5k,3
        cur_dist = (dist_all.sum(dim=-1))**0.5 # 300k,5k
        min_idxs = torch.min(cur_dist, 1)[1]
        del cur_dist
        closest_idx_list.append(min_idxs)
    all_nn_idxs = torch.cat(closest_idx_list,axis=0)
    # just inversely weight all points
    all_probs = pred_softmax[all_nn_idxs]
    
    # now argmax
    if panoptic:
        pred_full = all_probs.argmax(dim=1).cpu() + 1# here, no unlabeled, 1,...n_part correspond to actual part assignment
    else:
        pred_full = all_probs.argmax(dim=1).cpu()# here, 0 is unlabeled, 1,...n_part correspond to actual part assignment
    
    # convert back to the original coordinate system for visualization
    visualize_pt_labels(xyz_visualization.cpu(), pred_full.cpu(), save_rendered_path=img_save_path)
    return

# note the two methods below are heuristic---they are not used in any formal
# evaluation, they are only for visualization/diagnostics during training
def compute_overall_iou_objwise(pred, # n_pts, feat_dim
                                text_embeds, # n_mask, feat_dim
                                masks, # n_masks, h, w - binary in 2d
                                mask_view_idxs, # n_masks, each has a view index, -1 for padding
                                point2face, # n_pts
                                pixel2face, # 10,H,W
                                temperature
                                ):
    # the text embedding is normalized to norm 1
    # the pred is not normalized since it's whatever the model outputs
    # we regard anything > 0 as a match and <= as not a match when obtaining masks
    # we can further adjust this if we decide to normalize the pred here (not during training since then the 
    # contrastive loss will be affected)
    THRESHOLD = 0.6
    # first get each point's logits
    n_views, H, W = pixel2face.shape
    masks_flattened = masks.view(masks.shape[0],-1) # n_masks, (H*W)

    # for each view, get a H,W,n_mask distribution
    all_ious = []
    for i in range(n_views):
        if torch.sum((mask_view_idxs==i)*1) == 0:
            continue
        relevant_masks = masks_flattened[mask_view_idxs==i,:]# cur_view_n_masks,(H*W)
        relevant_text_embeds = text_embeds[mask_view_idxs==i,:]# cur_view_n_masks,feat_dim
        logits = pred @ relevant_text_embeds.T # n_pts, cur_view_n_masks

        # get binary mask of (H*W)*5000
        cur_faces = pixel2face[i,:,:].view(-1,1)
        pixel2point_mask = (cur_faces == point2face.view(1,-1))*1.0 # (H*W),n_pts
        # this is binary where all points contributing to each pixel is 1
        # need to normalize
        n_pts = pixel2point_mask.sum(dim=1)
        normalized_mask = pixel2point_mask / (n_pts+1e-12).view(-1,1) # (H*W),n_pts
        # should be
        # [1/3 0 0 ... 1/3 ... 1/3]
        # [0  1/2 1/2 ...0  .....0]
        view_logits = normalized_mask @ logits # (H*W),cur_view_n_masks
        # append 0 - in case only one mask available
        view_logits_append0 = torch.cat([view_logits, torch.zeros(view_logits.shape[0],1).cuda()],axis=1)
        # view_logits: for each pixel, we get an average of the logits of all points that correspond to this pixel 
        view_softmax = torch.nn.Softmax(dim=1)(view_logits_append0 * temperature)[:,:-1]
        
        # we need a threshold and can't just take max! because the majority of pixels should not correspond to any mask
        # to avoid setting a manual threshold, we use a coefficient * max across all labels
        # the premise is that since we provide ground truth text queries, the max should be informative (large, close to 1)
        # and by taking a coefficient e.g. 0.5 we are taking some relatively high-correspondence regions
        thres_binary_mask = (view_softmax > THRESHOLD)*1 # (H*W),cur_view_n_masks

        # get threshold IoU
        iou = batch_iou(thres_binary_mask.T, relevant_masks)

        all_ious.append(iou)

    all_iou_vec = torch.cat(all_ious)
    mean_iou = all_iou_vec.mean().item()
    return mean_iou

# visualize predicted masks
def viz_pred_mask(pred, # n_pts, feat_dim
                  text_embeds, # n_mask, feat_dim
                  texts, # list of n_mask
                  masks, # n_masks, h, w - binary in 2d
                  mask_view_idxs, # n_masks, each has a view index, -1 for padding
                  point2face, # 5000
                  pixel2face, # 10,H,W
                  n_epoch, # which epoch we are evaluating
                  obj_visualize_idx, # which object we are evaluating
                  prefix, # prefix for saving
                  temperature,
                  threshold=0.6
                  ):
    i_vals = [2,3,6]
    j_vals = [0,1]
    for i in i_vals:
        for j in j_vals:
            # first get relevant masks and texts
            if torch.sum((mask_view_idxs==i)*1) <= j:
                return
            relevant_masks = masks[mask_view_idxs==i,:,:][j,:,:] # h,w
            H,W = relevant_masks.shape
            relevant_text_embeds = text_embeds[mask_view_idxs==i,:] # n_masks, feat_dim
            relevant_text = [i for (i, v) in zip(texts, (mask_view_idxs==i).tolist()) if v][j][0]
            # first get each point's logits
            logits = pred @ relevant_text_embeds.T # n_pts, n_curview_masks

            # first get binary mask of (H*W)*5000
            cur_faces = pixel2face[i,:,:].view(-1,1)
            pixel2point_mask = (cur_faces == point2face.view(1,-1))*1.0 # (H*W),n_pts
            # this is binary where all points contributing to each pixel is 1
            # need to normalize
            n_pts = pixel2point_mask.sum(dim=1)
            normalized_mask = pixel2point_mask / (n_pts+1e-12).view(-1,1) # (H*W),n_pts
            # should be
            # [1/3 0 0 ... 1/3 ... 1/3]
            # [0  1/2 1/2 ...0  .....0]
            view_logits = normalized_mask @ logits # (H*W),cur_n_masks
            # we append a new 0 category just in case in current view there is only one mask, in which case we would have gotten all 1 after softmax otherwise
            view_logits_append0 = torch.cat([view_logits, torch.zeros(view_logits.shape[0],1).cuda()],axis=1)
            # view_logits: for each pixel, we get an average of the logits of all points that correspond to this pixel 
            view_softmax = torch.nn.Softmax(dim=1)(view_logits_append0 * temperature)
            view_softmax_heatmap = view_softmax[:,j].view(H,W)
            os.makedirs(f"training_checkpts/visualization/{prefix}_obj{obj_visualize_idx}/", exist_ok=True)

            # visualize heatmap and gt
            plt.clf()
            plt.imshow(view_softmax_heatmap.cpu())
            plt.colorbar()
            plt.title(relevant_text)
            plt.savefig(f"training_checkpts/visualization/{prefix}_obj{obj_visualize_idx}/view{i}mask{j}_{n_epoch}_pred_heatmap.png")

            plt.clf()
            plt.imshow(((view_softmax_heatmap>threshold)*1).cpu())
            plt.title(relevant_text)
            plt.savefig(f"training_checkpts/visualization/{prefix}_obj{obj_visualize_idx}/view{i}mask{j}_{n_epoch}_pred_mask.png")

            plt.clf()
            plt.imshow(relevant_masks.cpu())
            plt.title(relevant_text)
            plt.savefig(f"training_checkpts/visualization/{prefix}_obj{obj_visualize_idx}/view{i}mask{j}_gt_heatmap.png")
        
    return


def get_feature(model, xyz, rgb, normal): # evaluate loader can only have batch size=1
    data = preprocess_pcd(xyz.cuda(), rgb.cuda(), normal.cuda())
    with torch.no_grad():
        for key in data.keys():
            if isinstance(data[key], torch.Tensor) and "full" not in key:
                data[key] = data[key].cuda(non_blocking=True)
        net_out = model(x=data) # n_pts,dim_feats
        xyz_sub = data["coord"] # n_pts,3
    return xyz_sub, net_out