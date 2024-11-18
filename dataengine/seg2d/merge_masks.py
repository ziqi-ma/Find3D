# this script creates the following:
# under each valid data folder, create a merged directory
# which includes all masks from all views (merged if multiple masks under the same text)
# records which points correspond to each mask
# each mask corresponds to which view
# and the text per mask
# e.g. [name]_uid/norotate/masks/merged
# - allmasks.pt (this is n_masks*h*w binary)
# - mask2view.pt (this is shape (n_masks,) each to a view id 0-9)
# - mask_labels.txt (this is list of text mask names of all masks, size n_masks)

# for usage see
# training: model.training.loss DistillLossContrastive
# evaluate: model.evaluation.core compute_overall_iou_objwise (this is iou per mask)

import torch
import json
import os
from tqdm import tqdm
import shutil
import pandas as pd
from dataengine.configs import DATA_ROOT

def merge_masks(obj_dir, file_e):
    if not os.path.exists(obj_dir): # rendering prob had issues
        return
    if os.path.exists(f"{obj_dir}/masks/merged"): 
        done=True
        for i in range(10):
            if os.path.exists(f"{obj_dir}/masks/{i:02d}"):
                done = False
        if done:
            return
    if not os.path.exists(f"{obj_dir}/masks/partnames.json"): # not labeled
        file_e.write(f"{obj_dir}: no partnames\n")
        return
    metadata_path = f"{obj_dir}/masks/partnames.json"
    with open(metadata_path, "r") as f:
        mask2parts = json.load(f)
    
    if len(mask2parts) == 0:
        return
    
    # merge features per view
    lastview = ""
    view_seg_dict = {}
    all_masks_list = []
    all_labels_list = []
    all_mask_view_idx_list = []
    for labeled_file in mask2parts: # of the format "view00/mask23.png"
        curview = labeled_file.split("/")[0]
        if curview != lastview and lastview != "":
            for label in view_seg_dict:
                all_labels_list.append(label)
                all_masks_list.append(view_seg_dict[label]) # append a h*w binary matrix
                all_mask_view_idx_list.append(int(lastview[-2:]))
            view_seg_dict = {} # re-initialize the segmentation dictionary, this is for merging parts with the same name
        
        partname = mask2parts[labeled_file]
        if "unknown" not in partname:
            maskname = labeled_file.split("/")[1].split(".")[0] # e.g. mask23
            seg = torch.tensor(torch.load(f"{obj_dir}/masks/{curview}/{maskname}.pt")*1.0)
            # aggregate the mask
            if partname in view_seg_dict:
                view_seg_dict[partname] += seg
                view_seg_dict[partname] = (view_seg_dict[partname] > 0)*1.0
            else:
                view_seg_dict[partname] = seg
        lastview = curview
    
    if lastview != "":
        for label in view_seg_dict:
            all_labels_list.append(label)
            all_masks_list.append(view_seg_dict[label]) # append a h*w binary matrix
            all_mask_view_idx_list.append(int(lastview[-2:]))
    
    if len(all_labels_list) == 0:
        return
    # save
    os.makedirs(f"{obj_dir}/masks/merged", exist_ok=True)
    all_masks = torch.stack(all_masks_list) # should be n_masks, h, w
    mask2view = torch.tensor(all_mask_view_idx_list) # should be a tensor of (n_masks,)
    torch.save(all_masks, f"{obj_dir}/masks/merged/allmasks.pt")
    torch.save(mask2view, f"{obj_dir}/masks/merged/mask2view.pt")
    with open(f"{obj_dir}/masks/merged/mask_labels.txt", 'w') as f:
        for label in all_labels_list:
            f.write(f"{label}\n")

    # delete all the non-merged masks
    for i in range(10):
        if os.path.exists(f"{obj_dir}/masks/{i:02d}"):
            shutil.rmtree(f"{obj_dir}/masks/{i:02d}")
    return


if __name__ == "__main__":
    parent_folder = f"{DATA_ROOT}/labeled/rendered"
    cur_df = pd.read_csv(f"{DATA_ROOT}/labeled/chunk_ids/merged3.csv")
    cur_df["path"] = cur_df["class"] + "_" + cur_df["uid"]
    child_dirs = cur_df["path"].tolist()
    full_dirs = [parent_folder+"/"+child_dir for child_dir in child_dirs]
    file_e = open(f"merge_masks_exceptions.txt", "a")
    for obj_dir in tqdm(full_dirs):
        merge_masks(f"{obj_dir}/oriented", file_e)
        

