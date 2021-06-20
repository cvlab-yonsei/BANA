import torch
import torch.nn as nn
import torch.nn.functional as F

from .Layers import VGG16
from utils.etc_util import pad_for_grid


    
class Labeler(nn.Module):
    def __init__(self, num_classes, roi_size, grid_size):
        super().__init__()
        self.backbone = VGG16(dilation=1)
        self.classifier = nn.Conv2d(1024, num_classes, 1, bias=False)

        self.OH, self.OW = roi_size
        self.GS = grid_size
        self.from_scratch_layers = [self.classifier]
        
    def get_features(self, x):
        return self.backbone(x)
    
    def weighted_avg_pool_2d(self, input, weight): 
        filtered = input * weight
        _,dims,input_H,input_W = filtered.shape
        stride_H = input_H//self.OH
        stride_W = input_W//self.OW
        if stride_H==0:
            stride_H += 1
            pad_H = self.OH - input_H
            filtered = F.pad(filtered, [0,0,0,pad_H], mode='replicate')
            weight = F.pad(weight, [0,0,0,pad_H], mode='replicate')
        if stride_W==0:
            stride_W += 1
            pad_W = self.OW - input_W
            filtered = F.pad(filtered, [pad_W,0,0,0], mode='replicate')
            weight = F.pad(weight, [pad_W,0,0,0], mode='replicate')
        ks_H = input_H - (self.OH-1)*stride_H
        ks_W = input_W - (self.OW-1)*stride_W
        if ks_H <= 0:
            ks_H = 1
        if ks_W <= 0:
            ks_W = 1
        kernel = torch.ones((dims,1,ks_H,ks_W)).type_as(filtered)
        numer = F.conv2d(filtered, kernel, stride=(stride_H,stride_W), groups=dims)
        denom = F.conv2d(weight, kernel[0][None], stride=(stride_H,stride_W)) + 1e-12
        return numer / denom
    
    def gen_grid(self, box_coord, width, height):
        wmin, hmin, wmax, hmax = box_coord[:4]
        grid_x = torch.linspace(wmin, wmax, width).view(1,1,width,1)
        grid_y = torch.linspace(hmin, hmax, height).view(1,height,1,1)
        grid_x = grid_x.expand(1,height,width,1)
        grid_y = grid_y.expand(1,height,width,1)
        grid = torch.cat((grid_x,grid_y), dim=-1)
        return grid
    
    def BAP(self, features, bboxes, batch_id_per_box, bg_protos, valid_within_batch, indices_valid_bg_mask):
        # bg_protos: (N,GS**2,dims,1,1)
        batch_size, _, fH, fW = features.shape
        norm_H, norm_W = (fH-1)/2, (fW-1)/2
        widths = bboxes[:,[0,2]]*norm_W + norm_W
        heights = bboxes[:,[1,3]]*norm_H + norm_H
        widths = (widths[:,1].ceil() - widths[:,0].floor()).int()
        heights = (heights[:,1].ceil() - heights[:,0].floor()).int()
        fg_protos = []
        loss_regul = 0 
        count = 0 
        for batch_id in range(batch_size):
            feature_map = features[batch_id][None] # (1,dims,fH,fW)
            indices = batch_id_per_box==batch_id
            for coord, width, height in zip(bboxes[indices],widths[indices],heights[indices]):
                grid = self.gen_grid(coord, width, height).type_as(feature_map)
                roi = F.grid_sample(feature_map, grid) # (1,dims,BH,BW)
                GAP_attn = torch.ones(1,1,*roi.shape[-2:]).type_as(roi)
                if indices_valid_bg_mask[batch_id] and len(valid_within_batch[batch_id]):
                    normed_roi = F.normalize(roi, dim=1)
                    #(N,GS**2,dims,1,1)->(M,dims,1,1)
                    valid_bg_p = bg_protos[batch_id, valid_within_batch[batch_id]]
                    normed_bg_p = F.normalize(valid_bg_p, dim=1)
                    bg_attns = F.relu(torch.sum(normed_roi*normed_bg_p, dim=1, keepdim=True)) 
                    bg_attn = torch.mean(bg_attns, dim=0, keepdim=True) # (1,1,BH,BW): Naive Aggregate!
                    fg_attn = 1 - bg_attn
                    BAP_fgs = self.weighted_avg_pool_2d(roi, fg_attn) # BAP 1,256,OH,OW
                    fg_protos.append(BAP_fgs)
                else:
                    GAP_fgs = self.weighted_avg_pool_2d(roi, GAP_attn) # GAP 1,256,OH,OW
                    fg_protos.append(GAP_fgs)
        fg_protos = torch.cat(fg_protos, dim=0)
        return fg_protos
    
    def get_denom_and_indices(self, padded_mask, grid_size):
        batch_size = padded_mask.shape[0]
        padded_H, padded_W = padded_mask.shape[-2:]
        cell_H, cell_W = padded_H//grid_size, padded_W//grid_size
        denom_grids = padded_mask.unfold(2,cell_H,cell_H).unfold(3,cell_W,cell_W) 
        denom_grids = torch.sum(denom_grids, dim=(4,5)) # (N,1,GS,GS,cH,cW) --> (N,1,GS,GS)
        denom_grids = denom_grids.view(-1,1,1,1) # (N * GS**2,1,1,1)
        valid_grids = [idx for idx, cell in enumerate(denom_grids) if cell > 0]
        denom_grids = denom_grids.view(batch_size,-1,1,1,1) # (N,GS**2,1,1,1)
        return denom_grids, valid_grids
    
    def get_bg_prototypes(self, padded_features, padded_mask, denom_grids, grid_size):
        batch_size,dims = padded_features.shape[:2]
        padded_H, padded_W = padded_mask.shape[-2:]
        cell_H, cell_W = padded_H//grid_size, padded_W//grid_size
        bg_features = (padded_mask * padded_features).unfold(2,cell_H,cell_H).unfold(3,cell_W,cell_W)
        bg_protos = torch.sum(bg_features, dim=(4,5)) # (N,dims,GS,GS,cH,cW) --> (N,dims,GS,GS)
        bg_protos = bg_protos.view(batch_size,dims,-1).permute(0,2,1)
        bg_protos = bg_protos.contiguous().view(batch_size,-1,dims,1,1)
        bg_protos = bg_protos / (denom_grids + 1e-12) # (N,GS**2,dims,1,1)
        return bg_protos
    
    def forward(self, img, bboxes, batch_id_per_box, bg_mask, indices_valid_bg_mask):
        '''
        img     : (N,3,H,W)
        bboxes  : (K,5)
        batch_id_per_box : (K,)
        bg_mask : (N,1,H,W)
        '''
        features = self.get_features(img) # (N,256,105,105)
        batch_size,dims,fH,fW = features.shape
##########################################################
        padded_mask = pad_for_grid(F.interpolate(bg_mask, (fH,fW)), self.GS)
        denom_grids, valid_grids = self.get_denom_and_indices(padded_mask, self.GS)
        valid_within_batch = []
        for grids in denom_grids:
            valid_within_batch.append([idx for idx, cell in enumerate(grids) if cell > 0])
##########################################################
        padded_features = pad_for_grid(features, self.GS)
        bg_protos = self.get_bg_prototypes(padded_features, padded_mask, denom_grids, self.GS)
########################################################## MODIFIED !
        fg_protos = self.BAP(features, bboxes, batch_id_per_box, bg_protos, valid_within_batch, indices_valid_bg_mask)
        num_fgs = fg_protos.shape[0]
        fg_protos = fg_protos.view(num_fgs,dims,-1).permute(0,2,1).contiguous().view(-1,dims,1,1)
        bg_protos = bg_protos.contiguous().view(-1,dims,1,1)[valid_grids] # (M,dims,1,1)
        protos = torch.cat((fg_protos,bg_protos),dim=0)
        out = self.classifier(protos)
        return out
    
    def get_params(self, do_init=True):
        # pret_weight, pret_bias, scratch_weight, scratch_bias
        params = ([], [], [], [])
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m in self.from_scratch_layers:
                    if do_init:
                        nn.init.normal_(m.weight, std=0.01)
                    params[2].append(m.weight)
                else:
                    params[0].append(m.weight)
                if m.bias is not None:
                    if m in self.from_scratch_layers:
                        if do_init:
                            nn.init.constant_(m.bias, 0)
                        params[3].append(m.bias)
                    else:
                        params[1].append(m.bias)
            if isinstance(m, nn.BatchNorm2d):
                if m in self.from_scratch_layers:
                    if do_init:
                        nn.init.constant_(m.weight, 1)
                    params[2].append(m.weight)
                else:
                    params[0].append(m.weight)
                if m.bias is not None:
                    if m in self.from_scratch_layers:
                        if do_init:
                            nn.init.constant_(m.bias, 0)
                        params[3].append(m.bias)
                    else:
                        params[1].append(m.bias)
        return params
    
    def segment(self, x):
        feature = self.get_features(x)
        return self.classifier(feature)
