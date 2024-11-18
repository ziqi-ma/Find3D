import os
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import numpy as np
import cv2
from tqdm.notebook import tqdm
import os
import matplotlib.pyplot as plt
import time
import supervision as sv
import pandas as pd
from dataengine.configs import DATA_ROOT, SAM_CHECKPOINT_PATH


def get_obj_multiview_masks(root_dir, sam):
    # if have masks already, skip
    if not os.path.exists(root_dir): # rendering prob had issues
        return
    if os.path.exists(f"{root_dir}/masks"):
        return
    
    mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=20,
            pred_iou_thresh=0.75,
            stability_score_thresh=0.75,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=350,
    )
    IMAGES_EXTENSIONS = ['jpg', 'jpeg', 'png']
    image_paths = sv.list_files_with_extensions(directory=f"{root_dir}/imgs",extensions=IMAGES_EXTENSIONS)
    for image_path in image_paths:
        image_name = image_path.name
        view_name = image_name.split(".")[0]
        image_path = str(image_path)
        image = cv2.imread(image_path) # this should be 500x500
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h,w,_ = image.shape
        #image = cv2.resize(image, (500,500),interpolation = cv2.INTER_LINEAR)

        masks = mask_generator.generate(image)
        i = 0
        
        os.makedirs(f"{root_dir}/masks/{view_name}", exist_ok = True)
        for mask in masks:
            # 0. filter out if too large or too small i.e. area > 0.2 of full area or < 350
            if mask['area']> 0.2 * h * w or mask['area'] < 350:
                continue
            # 1. save mask binary as pt
            seg = mask["segmentation"]
            torch.save(seg, f"{root_dir}/masks/{view_name}/mask{i}.pt")
            # 2. save overlay purple
            h, w = seg.shape[-2:]
            color = np.array([135/255, 0/255, 255/255, 0.4])
            mask_image = seg.reshape(h, w, 1) * color.reshape(1, 1, -1)
            plt.clf()
            plt.imshow(image)
            plt.gca().imshow(mask_image)
            plt.axis('off')
            plt.savefig(f"{root_dir}/masks/{view_name}/mask{i}.png")
            i += 1


def get_masks_dirs(dir_list):
    # set models
    SAM_ENCODER_VERSION = "vit_h"
    sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH).cuda()
    file_e = open("sam_seg_exceptions.txt", "a")
    start_time = time.time()
    for dir in tqdm(dir_list):
        try:
            get_obj_multiview_masks(f"{dir}/oriented", sam)
        except Exception:
            file_e.write(f"{dir}\n")
    file_e.close()
    end_time = time.time()
    file = open(f"get_masks_time.txt", "a")  # append mode
    file.write(f"rendering time {end_time-start_time} seconds")
    file.close()
    return


if __name__ == "__main__":
    chunk_idx = 0
    # generate masks
    parent_folder = f"{DATA_ROOT}/labeled/rendered"
    cur_df = pd.read_csv(f"{DATA_ROOT}/labeled/chunk_ids/merged{chunk_idx}.csv")
    cur_df["path"] = cur_df["class"] + "_" + cur_df["uid"]
    child_dirs = cur_df["path"].tolist()
    full_dirs = [parent_folder+"/"+child_dir for child_dir in child_dirs]
    get_masks_dirs(full_dirs)