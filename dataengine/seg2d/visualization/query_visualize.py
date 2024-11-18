### this script is just for debugging
### this is query visualization, both segmenting features (for one view)
### into provided labels or getting a heatmap for given text query (for one view)
import torch
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from dataengine.configs import DATA_ROOT

def visualize_segment_feats(labels, filename, save_name):
    part_num = len(labels)
    cmap_matrix = torch.tensor([[1,1,1], [1,0,0], [0,1,0], [0,0,1], [1,1,0], [1,0,1],
                [0,1,1], [0.5,0.5,0.5], [0.5,0.5,0], [0.5,0,0.5],[0,0.5,0.5],
                [0.1,0.2,0.3],[0.2,0.5,0.3], [0.6,0.3,0.2], [0.5,0.3,0.5],
                [0.6,0.7,0.2],[0.5,0.8,0.3]]).cuda()[:part_num+1,:]
    colors = ["red", "green", "blue", "yellow", "magenta", "cyan","grey", "olive",
                "purple", "teal", "navy", "darkgreen", "brown", "pinkpurple", "yellowgreen", "limegreen"]
    caption_list=[f"{labels[i]}:{colors[i]}" for i in range(part_num)]
    try:
        feat_vec = torch.load(filename) # h,w,c
    except Exception as e: 
        return
    h,w,c = feat_vec.shape
    feat_reshaped = feat_vec.reshape(h*w,c)

    model = AutoModel.from_pretrained("google/siglip-base-patch16-224").cuda() # dim 768
    tokenizer = AutoTokenizer.from_pretrained("google/siglip-base-patch16-224")

    inputs = tokenizer(labels, padding="max_length", return_tensors="pt")
    inputs["input_ids"] = inputs["input_ids"].cuda()
    with torch.no_grad():
        text_feat = model.get_text_features(**inputs)
    # normalize
    text_feat = text_feat / (text_feat.norm(dim=-1, keepdim=True) + 1e-12)

    logits = 100. * feat_reshaped @ text_feat.t() # h*w,k
    label_pix = torch.argmax(logits, dim=1)+1
    label_pix[torch.sum(feat_reshaped,1)==0] = 0 # set empty pixels to 0 (white)

    onehot = F.one_hot(label_pix, num_classes=part_num+1) * 1.0 # h*w, part_num+1, each row 00.010.0, first place is unlabeled (0 originally)
    pix_rgb = torch.matmul(onehot, cmap_matrix) # h*w,3
    vis_rgb = pix_rgb.reshape(h,w,3).cpu().numpy()
    plt.clf()
    plt.imshow(vis_rgb)
    plt.text(-0.1, 0, " ".join(caption_list), fontsize=10, wrap=True)
    plt.axis("off")
    plt.savefig(save_name)

def visualize_seg_obj_allviews(objpath, labels, n_views):
    os.makedirs(f"{objpath}/visualization/segment_query", exist_ok=True)
    for i in range(n_views):
        visualize_segment_feats(labels, f"{objpath}/masks/view{i:02d}/fmap.pt", f"{objpath}/visualization/segment_query/view{i:02d}.png")

### heatmap visualization
def visualize_heatmap_feats(query, filename, save_name):
    feat_vec = torch.load(filename) # h,w,c
    h,w,c = feat_vec.shape
    feat_reshaped = feat_vec.reshape(h*w,c)

    model = AutoModel.from_pretrained("google/siglip-base-patch16-224").cuda() # dim 768 #"google/siglip-so400m-patch14-384")
    tokenizer = AutoTokenizer.from_pretrained("google/siglip-base-patch16-224")#"google/siglip-so400m-patch14-384")

    inputs = tokenizer([query], padding="max_length", return_tensors="pt")
    inputs["input_ids"] = inputs["input_ids"].cuda()
    with torch.no_grad():
        text_feat = model.get_text_features(**inputs)
    # normalize
    text_feat = text_feat / (text_feat.norm(dim=-1, keepdim=True) + 1e-12)

    logits = feat_reshaped @ text_feat.t() # h*w,1
    logits = logits.view(h,w)
    
    #visualize
    plt.clf()
    plt.imshow(logits.detach().cpu())
    plt.title(query)
    plt.colorbar()
    plt.axis('off')
    plt.savefig(save_name)
    return


def visualize_heatmap_allviews(objpath, query, n_views):
    os.makedirs(f"{objpath}/visualization/heatmap_query/{query}", exist_ok=True)
    for i in range(n_views):
        visualize_heatmap_feats(query, f"{objpath}/masks/view{i:02d}/fmap.pt", f"{objpath}/visualization/heatmap_query/{query}/view{i:02d}.png")


if __name__ == "__main__":
    parent_folder = f"{DATA_ROOT}/rendered"
    child_dirs = ["snowman_db58509a85104a03bfb98cc35b1d929a"]
    n_objs = len(child_dirs)
    full_dirs = [parent_folder+"/"+child_dir for child_dir in child_dirs]

    for obj_dir in full_dirs:
        f = open(f"{obj_dir}/orientation.txt", "r")
        orientation = f.read()
        visualize_seg_obj_allviews(f"{obj_dir}/{orientation}", ["head","body","bottom","hat","arm"], 10)
        visualize_heatmap_allviews(f"{obj_dir}/{orientation}", "head", 10)