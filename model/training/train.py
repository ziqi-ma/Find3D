## Some of the training code builds upon https://github.com/ardianumam/PartDistill/blob/main/train.py
import os
import torch
import argparse
from torch.utils.data import DataLoader
import torch.optim as optim
from model.backbone.pt3.model import PointSemSeg
from model.data.data import TrainingData, EvalData, collate_fn
from tqdm import tqdm
import numpy as np
from model.evaluation.core import viz_pred_mask, compute_overall_iou_objwise
from model.training.loss import DistillLossContrastive
from transformers import AutoTokenizer, AutoModel
import wandb
import random


def create_data_loader(data_root, shuffle_train, shuffle_test, eval_split, drop_last_train=True, drop_last_test=False, is_test_only=False):
    # create data loader
    test_data = EvalData(data_root, split=eval_split)
    test_loader = DataLoader(test_data, 
                             batch_size=1, 
                             shuffle=shuffle_test,
                             collate_fn=collate_fn, 
                             num_workers=0, 
                             drop_last=drop_last_test)
    if is_test_only:    
        return test_loader
    
    train_data = TrainingData(data_root)
    
    BS = len(train_data) if len(train_data) < args.batch_size else args.batch_size # to handle if the dataset has less data than args.batch_size
    train_loader = DataLoader(train_data, 
                              batch_size=BS, 
                              shuffle=shuffle_train, 
                              collate_fn=collate_fn,
                              num_workers=0, # this needs to be 0 since we do text embedding in loader which requires cuda, cuda cannot have multiple workers
                              drop_last=drop_last_train)
    train_iter_per_epoch = (len(train_data) // args.batch_size)+1

    return train_loader, test_loader, train_iter_per_epoch
        

def evaluate(model, dataloader, loss_fn, n_epoch, set_name, temperature, eval_loss = True, visualize_idxs=[20,25,55,80,139]): # evaluate loader can only have batch size=1
    n_visualize_epoch = 5
    prefix = "pt"
    iou_list = []
    loss_list = []
    i = 0
    with torch.no_grad():
        for data in tqdm(dataloader, desc=f"Evaluating {set_name}-set"):
            for key in data.keys():
                if isinstance(data[key], torch.Tensor):
                    data[key] = data[key].cuda(non_blocking=True)

            net_out = model(x=data)
            text_embeds = data['label_embeds']
            masks = data['masks']
            mask_view_idxs = data["mask_view_idxs"]
            point2face = data['point2face']
            pix2face = data['pixel2face']
            labels = data['labels']
            mask_pts = data['mask2pt']
            pt_offset = data['offset']
            m = AutoModel.from_pretrained("google/siglip-base-patch16-224")
            tokenizer = AutoTokenizer.from_pretrained("google/siglip-base-patch16-224")
            inputs = tokenizer(labels[0], padding="max_length", return_tensors="pt")
            with torch.no_grad():
                text_feat = m.get_text_features(**inputs) # n_masks, feat_dim (768)
        
            #normalize
            text_feat = text_feat / (text_feat.norm(dim=-1, keepdim=True) + 1e-12)
            iou = compute_overall_iou_objwise(pred=net_out, # n_pts, feat_dim
                                              text_embeds = text_embeds, # n_masks, feat_dim
                                              masks=masks, #n_masks, h, w - binary in 2d
                                              mask_view_idxs = mask_view_idxs, # n_masks each has a view index, -1 for padding
                                              # metadata below
                                              point2face = point2face, # n_pts
                                              pixel2face = pix2face, # 10,H,W
                                              temperature= temperature
                                              )
            iou_list += [iou]

            # compute loss
            if eval_loss:
                loss = loss_fn(net_out,
                             pt_offset, # offset is tensor of shape B+1, marking starting idx of each obj
                             text_embeds,
                             mask_pts,
                             model.ln_logit_scale)
                loss_list += [loss.item()]

            # visualize if fall on visualization index
            if n_epoch % n_visualize_epoch == 0 and (i in visualize_idxs):
                viz_pred_mask(pred=net_out,
                            text_embeds = text_embeds, # n_masks, feat_dim
                            texts = [[x] for x in labels[0]],
                            masks=masks, #n_masks, h, w - binary in 2d
                            mask_view_idxs = mask_view_idxs, # n_masks each has a view index, -1 for padding
                            point2face = point2face, # n_pts
                            pixel2face = pix2face,# 10,H,W
                            n_epoch = n_epoch, # which epoch we are evaluating
                            obj_visualize_idx = i, # which object we are evaluating
                            prefix = f"{prefix}-{set_name}",
                            temperature=temperature
                            )
            i += 1
    miou = np.mean(iou_list)
    if eval_loss: 
        loss = np.mean(loss_list)
    else:
        loss = 0
    return miou, loss


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wandb.init(project="find3d", config=args)
        
    # define the checkpoint dir
    ckpt_dir = os.path.join(args.ckpt_dir, f"find3d_{args.exp_suffix}")
    os.makedirs(ckpt_dir, exist_ok=True)

    model = PointSemSeg(args=args, dim_output=768)
    
    model = model.to(device)

    train_loader, test_loader, train_iter_per_epoch = create_data_loader(args.data_root,
                                                   shuffle_train=True, 
                                                   shuffle_test=False,
                                                   eval_split = "val",
                                                   drop_last_train=True, 
                                                   drop_last_test=False)
    # also create an evaluation loader on the training set
    train_val_loader = create_data_loader(args.data_root, shuffle_train=True, shuffle_test=False, eval_split="train",is_test_only=True)

    # define the opt
    opt = optim.Adam(model.parameters(), lr=args.lr)
    
    if hasattr(args, "continue_path") and args.continue_path:
        model.load_state_dict(torch.load(args.continue_path)["model_state_dict"])
        print(f"loading from {args.continue_path} to keep training!")
        opt.load_state_dict(torch.load(args.continue_path)["optimizer_state_dict"])
        print("loaded optimizer state")
    
    model.train()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, train_iter_per_epoch * args.n_epoch, eta_min=args.eta_min)
           
    # define the loss
    criterion = DistillLossContrastive()

    iter = 0

    for epoch in range(args.n_epoch):
        
        epoch = epoch + 1
        loss_epoch_current = []

        for data in (tqdm(train_loader, desc=f"Training epoch: {epoch}/{args.n_epoch}")):
            for key in data.keys():
                if isinstance(data[key], torch.Tensor):
                    data[key] = data[key].cuda(non_blocking=True)

            mask_points = data['mask2pt'] 
            mask_embeds = data['label_embeds']
            pt_offset = data['offset']
            net_out = model(data) #input total_pts,3 net_out=[total_pts_batch, dim_feat]
            
            loss = criterion(net_out,
                             pt_offset, # offset is tensor of shape B+1, marking starting idx of each obj
                             mask_embeds,
                             mask_points,
                             model.ln_logit_scale)
            
            loss_epoch_current.append(loss.item())
            cur_lr = scheduler.get_last_lr()[0]

            opt.zero_grad()
            loss.backward()
            opt.step()
            scheduler.step()
            values_to_log = {"iter":iter, "train cumulative loss (batch 64)": loss.item(), "temperature": np.exp(model.ln_logit_scale.item()), "lr": cur_lr}
            wandb.log(values_to_log, step=iter, commit=True)
            iter += 1
        
        epoch_loss_avg = np.around(np.mean(loss_epoch_current), decimals=4)

        log_n_epochs = 5
        if epoch % log_n_epochs == 0:
            # evaluate
            miou_test, loss_test = evaluate(model=model.eval(),
                                    dataloader=test_loader, loss_fn=criterion, set_name = "val", eval_loss=True,
                                    n_epoch=epoch, temperature = torch.exp(model.ln_logit_scale))
            
            miou_train, loss_train_objbatch = evaluate(model=model.eval(),
                                dataloader=train_val_loader, loss_fn=criterion, set_name = "train", eval_loss=True,
                                n_epoch=epoch, temperature = torch.exp(model.ln_logit_scale))
            values_to_log = {"step":iter, "train loss (batch 64)": epoch_loss_avg, "train obj-batch loss":loss_train_objbatch, "val obj-batch loss": loss_test, "train iou": miou_train, "val iou": miou_test, "temperature": np.exp(model.ln_logit_scale.item())}
            wandb.log(values_to_log, step=iter, commit=True)
        
            epoch_text_out = f"Epoch {epoch}/{args.n_epoch} --> loss: batch-64 train {epoch_loss_avg}  batch-1 train {loss_train_objbatch} val {loss_test}, iou: train {miou_train} val {miou_test}"
            print(epoch_text_out)
            model = model.train()
        
        if epoch % 5 == 0:
            ckpt_path_test = os.path.join(ckpt_dir, f"ckpt_{epoch}.pth")
            torch.save({'epoch':epoch,
                    'model_state_dict':model.state_dict(),
                    'optimizer_state_dict':opt.state_dict(),
                    'loss':loss,
                    'lntemperature': model.ln_logit_scale,
                    'scheduler_state_dict':scheduler.state_dict()
                    },ckpt_path_test)


    torch.save({'epoch':epoch,
                'model_state_dict':model.state_dict(),
                'optimizer_state_dict':opt.state_dict(),
                'loss':loss,
                'lntemperature': model.ln_logit_scale,
                'scheduler_state_dict':scheduler.state_dict()
                },ckpt_path_test)

       
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epoch', type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--lr', type=float, default=0.0003, metavar='LR',
                        help='learning rate')
    parser.add_argument('--eta_min', type=float, default=0.00005, metavar='LR',
                        help='learning rate')
    parser.add_argument('--step', type=int, default=40,
                        help='lr decay step')
    parser.add_argument('--use_aug', type=int, default=1, choices=[0, 1]) #1 and 0 to use or not to use data augmentation
    parser.add_argument('--normalize_cloud', type=int, default=1, choices=[0, 1]) #1 and 0 to normalize or not
    parser.add_argument('--ckpt_dir',  type=str, default="checkpoints") #location to store the output 
    parser.add_argument('--continue_path',  type=str)
    parser.add_argument('--n_mov_avg', type=int, default=5) 
    parser.add_argument('--exp_suffix', default='', type=str)
    parser.add_argument('--data_root', required=True, default='', type=str)

    args = parser.parse_args()
    args.seed = 123
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    train(args)