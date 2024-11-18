### this script is for visualization and debugging
import torch
import matplotlib.pyplot as plt
import os
import numpy as np

def visualize_all_masks(dir):
    mask2view = torch.load(f"{dir}/oriented/masks/merged/mask2view.pt")
    allmasks = torch.load(f"{dir}/oriented/masks/merged/allmasks.pt")
    f = open(f"{dir}/oriented/masks/merged/mask_labels.txt", "r")
    os.makedirs(f"{dir}/visualization", exist_ok=True)
    labels = f.read().splitlines()
    f.close()
    cmap_matrix = torch.tensor([[1,0,0], [0,1,0], [0,0,1], [1,1,0], [1,0,1],
                [0,1,1], [0.5,0.5,0.5], [0.5,0.5,0], [0.5,0,0.5],[0,0.5,0.5],
                [0.1,0.2,0.3],[0.2,0.5,0.3], [0.6,0.3,0.2], [0.5,0.3,0.5],
                [0.6,0.7,0.2],[0.5,0.8,0.3],[1,0.5,0.45],[1,0.08,0.58],[1,0.84,0]])
    colors = ["red", "green", "blue", "yellow", "magenta", "cyan","grey", "olive",
                "purple", "teal", "navy", "darkgreen", "brown", "pinkpurple", "yellowgreen", "limegreen",
                "salmon", "deeppink","gold"]
    max_view = mask2view.max().int()+1
    for i in range(max_view):
        all_mask = torch.zeros((500,500,3))
        cur_masks = allmasks[mask2view == i]
        curlabels = np.array(labels)[(mask2view==i).numpy()]
        curlen = np.min((cur_masks.shape[0], 5)) # later masks override previous
        for j in range(curlen):
            all_mask *= (1-cur_masks[j,:,:].view(500,500,1))
            all_mask += cur_masks[j,:,:].view(500,500,1)*cmap_matrix[j,:].view(1,1,3)
        caption_list = [f"{colors[i]}:{curlabels[i]}" for i in range(curlen)]
        plt.clf()
        plt.imshow(all_mask)
        plt.text(-0.1, 0, " ".join(caption_list), fontsize=10, wrap=True)
        plt.axis("off")
        plt.savefig(f"view{i}.png")
    return

if __name__=="__main__":
    dir = "a specific object directory"
    visualize_all_masks(dir)
        