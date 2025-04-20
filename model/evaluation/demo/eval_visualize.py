import torch
import numpy as np
from model.evaluation.core import visualize_3d_upsample
import numpy as np
import argparse
from model.evaluation.utils import set_seed, load_model, preprocess_pcd, read_pcd, encode_text


# data is a dict that has gone through preprocessing as training (normalizing etc.)
def visualize_seg3d(model, data, mode, N_CHUNKS=5): # evaluate loader can only have batch size=1
    if mode == "segmentation":
        heatmap = False
    elif mode == "heatmap":
        heatmap = True
    else:
        print("unsupported mode")
        return
    temperature = np.exp(model.ln_logit_scale.item())
    with torch.no_grad():
        for key in data.keys():
            if isinstance(data[key], torch.Tensor) and "full" not in key:
                data[key] = data[key].cuda(non_blocking=True)
        net_out = model(x=data)
        text_embeds = data['label_embeds']
        xyz_sub = data["coord"]
        xyz_full = data["xyz_full"]
        visualize_3d_upsample(net_out, # n_subsampled_pts, feat_dim
                            text_embeds, # n_parts, feat_dim
                            temperature,
                            xyz_sub,
                            xyz_full, # n_pts, 3
                            panoptic=False,
                            N_CHUNKS=N_CHUNKS,
                            heatmap=heatmap)
    return


def eval_obj_wild(model, obj_path, mode, queries):
    if mode not in ["segmentation", "heatmap"]:
        print("only segmentation or heatmap mode are supported")
        return
    xyz, rgb, normal = read_pcd(obj_path)
    data_dict = preprocess_pcd(xyz.cuda(), rgb.cuda(), normal.cuda())
    data_dict["label_embeds"] = encode_text(queries)
    visualize_seg3d(model, data_dict, mode)
    return


if __name__ == '__main__':
    set_seed(123)
    parser = argparse.ArgumentParser(description="Please specify input point cloud path and model checkpoint path")
    parser.add_argument("--object_path", required=True, type=str, help='The point cloud to evaluate on. Should be a .pcd file')
    parser.add_argument("--mode", required=True, type=str, help='segmentation or heatmap')
    parser.add_argument("--queries", required=True, nargs='+', help='list of queries')
    args = parser.parse_args()
    
    model = load_model()
    eval_obj_wild(model,args.object_path, args.mode, args.queries)

    