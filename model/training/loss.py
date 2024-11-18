import torch.nn as nn
import torch.nn.functional as F
import torch

# note that batching happens with offsets - all point clouds are concatenated and the offsets
# denote the start/end of each object
    
class DistillLossContrastive(nn.Module):
    def __init__(self):
        super(DistillLossContrastive, self).__init__()

    def forward(self, net_out, pt_offset, mask_embs, mask_pts, logit_scale, is_backward_dist=False, stud_kowledge=None): 
        """
        net_out: [n_total_pts, dim_ft]
        pt_offset: [BS+1,] # start of each object's point idx, the last one is value n_total_pts we can take the [:BS], e.g. [ 3984,   8430,  12707,  16621] if total 16621 pts
        mask_offset: [BS+1,] # start of each object's mask idx, the last one is value n_total_masks we can take the [:BS]
        mask_embs: [n_totak_masks, dim_ft] n_masks_max max number of masks for objects in this batch, padded with 0
        mask_pts: list of [n_cur_masks, n_pts_cur_obj] binary, each entry indicates whether the point is visible for given object's given mask 
        """
        N_MASKS_TOTAL, N_DIM = mask_embs.shape
        mask_npts = [torch.sum(mask_pt, dim=1).view(-1,1) for mask_pt in mask_pts] # list of size(n_cur_mask,0), each is number of points for given mask
        all_masks_npts = torch.cat(mask_npts).cuda()
        mask_nopts = ((all_masks_npts==0)*1).squeeze()
        # since n_pts is different per object, this cannot be vectorized so has to be done sequentially, sadly
        obj_start_idxs = torch.cat((torch.tensor([0]).cuda(), pt_offset[:-1])) # e.g. [0,25,300] whereas pt_offset is [25,300,450]

        sum_feats = [mask_pt.cuda()*1.0 @ net_out[start_idx:end_idx,:] for (mask_pt, start_idx, end_idx) in zip(mask_pts, obj_start_idxs, pt_offset)] # each item is n_cur_masks, n_pts_cur, times n_pts_cur, n_dim, resulting in n_cur_masks, n_dim
        all_sum_feats = torch.cat(sum_feats)
        all_mask_avg_feats = all_sum_feats / (all_masks_npts+1e-12) # n_total_masks, out_dim

        # get dot product with text
        logits = mask_embs @ all_mask_avg_feats.T * torch.exp(logit_scale) # this is (n_total_mask, n_total_mask), row=text col=point
        target = torch.arange(N_MASKS_TOTAL).cuda() # size (BS*N_MASKS_MAX,) this is label for diagonal
        modified_target = mask_nopts * -100 + (1-mask_nopts)*target # if no point, -100, otherwise, embedding
        texts_loss = F.cross_entropy(logits, modified_target, reduction='none') # BS*N_MASKS_MAX - CE across texts
        pts_loss = F.cross_entropy(logits.T, modified_target, reduction='none') # BS*N_MASKS_MAX - CE across images
        
        # disregard zeros
        if texts_loss.sum()>0:
            texts_loss_nonzero_avg = (texts_loss[texts_loss>0]).mean()
        else:
            texts_loss_nonzero_avg = torch.tensor(0)

        if pts_loss.sum()>0:
            pts_loss_nonzero_avg = (pts_loss[pts_loss>0]).mean()
        else:
            pts_loss_nonzero_avg = torch.tensor(0)
        loss =  (texts_loss_nonzero_avg + pts_loss_nonzero_avg) / 2.0

        if is_backward_dist:
            pass

        return loss
