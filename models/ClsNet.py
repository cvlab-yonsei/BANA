import torch
import torch.nn as nn
import torch.nn.functional as F

from .Layers import VGG16


def pad_for_grid(mask, grid_size):
    Pad_H = grid_size - mask.shape[2] % grid_size
    Pad_W = grid_size - mask.shape[3] % grid_size
    if Pad_H == grid_size:
        Pad_H = 0
    if Pad_W == grid_size:
        Pad_W = 0
    if Pad_H % 2 == 0:
        if Pad_W % 2 == 0:
            out = F.pad(mask, [Pad_W//2, Pad_W//2, Pad_H//2, Pad_H//2], value=0)
        else:
            out = F.pad(mask, [ 0, Pad_W, Pad_H//2, Pad_H//2], value=0)
    else:
        if Pad_W % 2 == 0:
            out = F.pad(mask, [Pad_W//2, Pad_W//2, 0, Pad_H], value=0)
        else:
            out = F.pad(mask, [0, Pad_W, 0, Pad_H], value=0)
    return out


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
            weight   = F.pad(weight,   [0,0,0,pad_H], mode='replicate')
        if stride_W==0:
            stride_W += 1
            pad_W = self.OW - input_W
            filtered = F.pad(filtered, [pad_W,0,0,0], mode='replicate')
            weight   = F.pad(weight,   [pad_W,0,0,0], mode='replicate')
        ks_H = input_H - (self.OH-1)*stride_H
        ks_W = input_W - (self.OW-1)*stride_W
        if ks_H <= 0:
            ks_H = 1
        if ks_W <= 0:
            ks_W = 1
        kernel = torch.ones((dims,1,ks_H,ks_W)).type_as(filtered)
        numer  = F.conv2d(filtered, kernel,          stride=(stride_H,stride_W), groups=dims)
        denom  = F.conv2d(weight,   kernel[0][None], stride=(stride_H,stride_W)) + 1e-12
        return numer / denom
    
    def gen_grid(self, box_coord, width, height):
        wmin, hmin, wmax, hmax = box_coord[:4]
        grid_x = torch.linspace(wmin, wmax, width).view(1,1,width,1)
        grid_y = torch.linspace(hmin, hmax, height).view(1,height,1,1)
        grid_x = grid_x.expand(1,height,width,1)
        grid_y = grid_y.expand(1,height,width,1)
        grid = torch.cat((grid_x,grid_y), dim=-1)
        return grid
    
    def BAP(self, features, bboxes, batchID_of_box, bg_protos, valid_cellIDs, ind_valid_bg_mask):
        batch_size, _, fH, fW = features.shape
        norm_H, norm_W = (fH-1)/2, (fW-1)/2
        widths  = bboxes[:,[0,2]]*norm_W + norm_W
        heights = bboxes[:,[1,3]]*norm_H + norm_H
        widths  = (widths[:,1].ceil() - widths[:,0].floor()).int()
        heights = (heights[:,1].ceil() - heights[:,0].floor()).int()
        fg_protos = []
        for batch_id in range(batch_size):
            feature_map = features[batch_id][None] # (1,dims,fH,fW)
            indices = batchID_of_box==batch_id
            for coord, width, height in zip(bboxes[indices],widths[indices],heights[indices]):
                grid = self.gen_grid(coord, width, height).type_as(feature_map)
                roi = F.grid_sample(feature_map, grid) # (1,dims,BH,BW)
                GAP_attn = torch.ones(1,1,*roi.shape[-2:]).type_as(roi)
                ID_list = valid_cellIDs[batch_id]
                if ind_valid_bg_mask[batch_id] and len(ID_list):
                    normed_roi = F.normalize(roi, dim=1)
                    valid_bg_p = bg_protos[batch_id, ID_list] #(N,GS**2,dims,1,1)->(len(ID_list),dims,1,1)
                    normed_bg_p = F.normalize(valid_bg_p, dim=1)
                    bg_attns = F.relu(torch.sum(normed_roi*normed_bg_p, dim=1, keepdim=True)) 
                    bg_attn = torch.mean(bg_attns, dim=0, keepdim=True)
                    fg_attn = 1 - bg_attn
                    fg_by_BAP = self.weighted_avg_pool_2d(roi, fg_attn) # (1,256,OH,OW)
                    fg_protos.append(fg_by_BAP)
                else:
                    fg_by_GAP = self.weighted_avg_pool_2d(roi, GAP_attn) # (1,256,OH,OW)
                    fg_protos.append(fg_by_GAP)
        fg_protos = torch.cat(fg_protos, dim=0)
        return fg_protos
    
    def get_grid_bg_and_IDs(self, padded_mask, grid_size):
        batch_size, _, padded_H, padded_W = padded_mask.shape
        cell_H, cell_W = padded_H//grid_size, padded_W//grid_size
        grid_bg = padded_mask.unfold(2,cell_H,cell_H).unfold(3,cell_W,cell_W) 
        grid_bg = torch.sum(grid_bg, dim=(4,5)) # (N,1,GS,GS,cH,cW) --> (N,1,GS,GS)
        grid_bg = grid_bg.view(-1,1,1,1) # (N * GS**2,1,1,1)
        valid_gridIDs = [idx for idx, cell in enumerate(grid_bg) if cell > 0]
        grid_bg = grid_bg.view(batch_size,-1,1,1,1) # (N,GS**2,1,1,1)
        return grid_bg, valid_gridIDs
    
    def get_bg_prototypes(self, padded_features, padded_mask, denom_grids, grid_size):
        batch_size, dims, padded_H, padded_W = padded_features.shape
        cell_H, cell_W = padded_H//grid_size, padded_W//grid_size
        bg_features = (padded_mask * padded_features).unfold(2,cell_H,cell_H).unfold(3,cell_W,cell_W)
        bg_protos = torch.sum(bg_features, dim=(4,5)) # (N,dims,GS,GS,cH,cW) --> (N,dims,GS,GS)
        bg_protos = bg_protos.view(batch_size,dims,-1).permute(0,2,1)
        bg_protos = bg_protos.contiguous().view(batch_size,-1,dims,1,1)
        bg_protos = bg_protos / (denom_grids + 1e-12) # (N,GS**2,dims,1,1)
        return bg_protos
    
    def forward(self, img, bboxes, batchID_of_box, bg_mask, ind_valid_bg_mask):
        '''
        img               : (N,3,H,W) float32
        bboxes            : (K,5) float32
        batchID_of_box    : (K,) int64
        bg_mask           : (N,1,H,W) float32
        ind_valid_bg_mask : (N,) uint8
        '''
        features = self.get_features(img) # (N,256,105,105)
        batch_size,dims,fH,fW = features.shape
        ##########################################################
        padded_mask = pad_for_grid(F.interpolate(bg_mask, (fH,fW)), self.GS)
        grid_bg, valid_gridIDs = self.get_grid_bg_and_IDs(padded_mask, self.GS)
        valid_cellIDs = []
        for grids in grid_bg:
            valid_cellIDs.append([idx for idx, cell in enumerate(grids) if cell > 0])
        ##########################################################
        padded_features = pad_for_grid(features, self.GS)
        bg_protos = self.get_bg_prototypes(padded_features, padded_mask, grid_bg, self.GS)
        fg_protos = self.BAP(features, bboxes, batchID_of_box, bg_protos, valid_cellIDs, ind_valid_bg_mask)
        ##########################################################
        num_fgs = fg_protos.shape[0]
        fg_protos = fg_protos.view(num_fgs,dims,-1).permute(0,2,1).contiguous().view(-1,dims,1,1) # (num_fgs,dims,OH,OW) --> (num_fgs*OH*OW,dims,1,1)
        bg_protos = bg_protos.contiguous().view(-1,dims,1,1)[valid_gridIDs] # (len(valid_gridIDs),dims,1,1)
        protos = torch.cat((fg_protos,bg_protos),dim=0)
        out = self.classifier(protos)
        return out
    
    def get_params(self, do_init=True):
        '''
        This function is borrowed from AffinitNet. It returns (pret_weight, pret_bias, scratch_weight, scratch_bias).
        Please, also see the paper (Learning Pixel-level Semantic Affinity with Image-level Supervision, CVPR 2018), and codes (https://github.com/jiwoon-ahn/psa/tree/master/network).
        '''
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