import cv2
import torch
import random
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF

def clip_bbox(bbox, clip_coord):
    wmin, hmin, wmax, hmax = clip_coord
    # get rid of out of cropped region
    invalid = (bbox[:,0] >= wmax) | (bbox[:,1] >= hmax) | (bbox[:,2] <= wmin) | (bbox[:,3] <= hmin)
    bbox = bbox[~invalid]
    new_bbox = np.copy(bbox)
    # wmin, hmin
    new_bbox[bbox[:,0]>=wmin, 0] -= wmin
    new_bbox[bbox[:,0]<wmin, 0] = 0
    new_bbox[bbox[:,1]>=hmin, 1] -= hmin
    new_bbox[bbox[:,1]<hmin, 1] = 0
    # wmax, hmax
    new_bbox[bbox[:,2]>=wmax, 2] = wmax - wmin + 1
    new_bbox[bbox[:,2]<wmax, 2] -= wmin
    new_bbox[bbox[:,3]>=hmax, 3] = hmax - hmin + 1
    new_bbox[bbox[:,3]<hmax, 3] -= hmin
    return new_bbox


class Compose():
    '''
    transforms : List of transforms for Image and Bboxes
    '''
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes, bg_mask=None): 
        for tr in self.transforms:
            img, bboxes, bg_mask = tr(img, bboxes, bg_mask) 
        return img, bboxes, bg_mask


class RandomScale():
    '''
    img     : (H, W, 3) numpy float32
    bboxes  : (wmin, hmin, wmax, hmax, cls) N x 5 numpy float32
    bg_mask : (H, W) numpy int64
    '''
    def __init__(self, scale_min, scale_max):
        self.s_min = scale_min
        self.s_max = scale_max

    def __call__(self, img, bboxes, bg_mask=None):
        scale = random.uniform(self.s_min, self.s_max)
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        if bboxes.shape != (0,):
            bboxes[:,:4] *= scale
        if bg_mask is not None:
            bg_mask = cv2.resize(bg_mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        return img, bboxes, bg_mask
    

class RandomHFlip():
    '''
    img    : (H, W, 3) numpy float32
    bboxes : (wmin, hmin, wmax, hmax, cls) N x 5 numpy float32
    bg_mask : (H, W) numpy int32
    '''
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, bboxes, bg_mask=None):
        cen_w, cen_h = np.array(img.shape[:2])[::-1]/2
        center = np.hstack((cen_w, cen_h, cen_w, cen_h))
        if random.random() < self.p and bboxes.shape != (0,):
            img = np.copy(img[:, ::-1, :])
            bboxes[:, [0, 2]] += 2*(center[[0, 2]] - bboxes[:, [0, 2]])
            box_w = abs(bboxes[:, 0] - bboxes[:, 2])
            bboxes[:, 0] -= box_w
            bboxes[:, 2] += box_w
            if bg_mask is not None:
                bg_mask = np.copy(bg_mask[:, ::-1])
        return img, bboxes, bg_mask
        

class ResizeRandomCrop():
    '''
    img    : (H, W, 3) numpy float32
    bboxes : (wmin, hmin, wmax, hmax, cls) N x 5 numpy float32
    bg_mask : (H, W) numpy int32
    '''
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, img, bboxes, bg_mask=None):
        crop_H, crop_W = self.crop_size
        img_H, img_W   = img.shape[:2]
        diff_H = crop_H-img_H
        diff_W = crop_W-img_W
        padH, padW = 0, 0
        if diff_H > 0:
            padH = diff_H // 2
            if diff_H % 2: 
                padH += 1
            if bboxes.shape != (0,):
                bboxes[:,[1,3]] += padH
        if diff_W > 0:
            padW = diff_W // 2
            if diff_W % 2: 
                padW += 1
            if bboxes.shape != (0,):
                bboxes[:,[0,2]] += padW
        img = cv2.copyMakeBorder(img, padH,padH,padW,padW, cv2.BORDER_CONSTANT, value=(123,117,104))
        resize_H, resize_W = img.shape[:2]
        new_wmin = int( np.floor(random.random() * (resize_W-crop_W)) )
        new_hmin = int( np.floor(random.random() * (resize_H-crop_H)) )
        img = img[new_hmin:new_hmin+crop_H, new_wmin:new_wmin+crop_W]
        if bboxes.shape != (0,):
            bboxes = clip_bbox(bboxes, [new_wmin,new_hmin,new_wmin+crop_W,new_hmin+crop_H])
        if bg_mask is not None:
            bg_mask = cv2.copyMakeBorder(bg_mask, padH,padH,padW,padW, cv2.BORDER_CONSTANT, value=0)
            bg_mask = bg_mask[new_hmin:new_hmin+crop_H, new_wmin:new_wmin+crop_W]
        return img, bboxes, bg_mask


class ColorJitter():
    '''
    img    : (H,W,3) numpy float32
    bboxes : (K,5) numpy float32
    bg_mask : (H, W) numpy int32
    '''
    def __init__(self, brightness, contrast, saturation, hue):
        self.brit = brightness
        self.cont = contrast
        self.sat = saturation
        self.hue = hue

    def __call__(self, img, bboxes, bg_mask=None): 
        pil_img = TF.to_pil_image(img.astype("uint8"))
        if self.brit:
            factor  = random.uniform(max(0,1-self.brit), 1+self.brit)
            pil_img = TF.adjust_brightness(pil_img, factor)
        if self.cont:
            factor  = random.uniform(max(0,1-self.cont), 1+self.cont)
            pil_img = TF.adjust_contrast(pil_img, factor)
        if self.sat:
            factor  = random.uniform(max(0,1-self.sat), 1+self.sat)
            pil_img = TF.adjust_saturation(pil_img, factor)
        if self.hue:
            factor  = random.uniform(-self.hue, self.hue)
            pil_img = TF.adjust_hue(pil_img, factor)
        img = np.array(pil_img, dtype=np.float32)
        return img, bboxes, bg_mask


class Normalize_Caffe():
    '''
    img    : (H,W,3) numpy float32
    bboxes : (K,5) numpy float32
    bg_mask : (H, W) numpy int32
    -----
    return (new)     : (3,H,W) tensor float32 
    return (bboxes)  : (K,5) tensor float32
    return (bg_mask) : (H,W) tensor float32
    '''
    def __init__(self, mean=(122.675, 116.669, 104.008)):
        self.mean = mean

    def __call__(self, img_RGB, bboxes, bg_mask=None):
        imgH, imgW  = img_RGB.shape[:2]
        img_BGR = np.empty_like(img_RGB, np.float32)
        img_BGR[..., 0] = img_RGB[..., 2] - self.mean[2]
        img_BGR[..., 1] = img_RGB[..., 1] - self.mean[1]
        img_BGR[..., 2] = img_RGB[..., 0] - self.mean[0]
        norm_H, norm_W = (imgH-1)/2, (imgW-1)/2
        if bboxes.shape != (0,):
            bboxes[:,[0,2]] = bboxes[:,[0,2]]/norm_W - 1
            bboxes[:,[1,3]] = bboxes[:,[1,3]]/norm_H - 1
        img = torch.from_numpy(img_BGR).permute(2,0,1)
        bboxes = torch.from_numpy(bboxes)
        if bg_mask is not None:
            bg_mask = torch.from_numpy(bg_mask).float()
        return img, bboxes, bg_mask