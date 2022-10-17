import cv2
import torch
import random
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF


class Compose():
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask, aux_mask=None): 
        for tr in self.transforms:
            img, mask, aux_mask = tr(img, mask, aux_mask) 
        return img, mask, aux_mask


class RandomScale():
    def __init__(self, scale_min, scale_max):
        self.s_min = scale_min
        self.s_max = scale_max

    def __call__(self, img, mask, aux_mask=None):
        scale = random.uniform(self.s_min, self.s_max)
        img   = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        if mask is not None:
            mask = cv2.resize(mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        if aux_mask is not None:
            aux_mask = cv2.resize(aux_mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        return img, mask, aux_mask
    

class RandomHFlip():
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, mask, aux_mask=None):
        if random.random() < self.p:
            img  = np.copy(img[:, ::-1, :])
            if mask is not None:
                mask = np.copy(mask[:, ::-1])
            if aux_mask is not None:
                aux_mask = np.copy(aux_mask[:, ::-1])
        return img, mask, aux_mask
        

class ResizeRandomCrop():
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, img, mask, aux_mask=None):
        crop_H, crop_W = self.crop_size
        img_H, img_W   = img.shape[:2]
        diff_H = crop_H-img_H
        diff_W = crop_W-img_W
        padH, padW = 0, 0
        if diff_H > 0:
            padH = diff_H // 2
            if diff_H % 2: 
                padH += 1
        if diff_W > 0:
            padW = diff_W // 2
            if diff_W % 2: 
                padW += 1
        img  = cv2.copyMakeBorder(img, padH,padH,padW,padW, cv2.BORDER_CONSTANT, value=(123,117,104))

        resize_H, resize_W = img.shape[:2]
        new_wmin = int( np.floor(random.random() * (resize_W-crop_W)) )
        new_hmin = int( np.floor(random.random() * (resize_H-crop_H)) )
        img  = img[new_hmin:new_hmin+crop_H, new_wmin:new_wmin+crop_W]
        if mask is not None:
            mask = cv2.copyMakeBorder(mask, padH,padH,padW,padW, cv2.BORDER_CONSTANT, value=255)
            mask = mask[new_hmin:new_hmin+crop_H, new_wmin:new_wmin+crop_W]
        if aux_mask is not None:
            aux_mask = cv2.copyMakeBorder(aux_mask, padH,padH,padW,padW, cv2.BORDER_CONSTANT, value=255)
            aux_mask = aux_mask[new_hmin:new_hmin+crop_H, new_wmin:new_wmin+crop_W]
        return img, mask, aux_mask


class ColorJitter():
    def __init__(self, brightness, contrast, saturation, hue):
        self.brit = brightness
        self.cont = contrast
        self.sat = saturation
        self.hue = hue

    def __call__(self, img, mask, aux_mask=None): 
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
        return img, mask, aux_mask


class Normalize_Caffe():
    def __init__(self, mean=(122.675, 116.669, 104.008)):
        self.mean = mean

    def __call__(self, img_RGB, mask, aux_mask=None):
        img_BGR = np.empty_like(img_RGB, np.float32)
        img_BGR[..., 0] = img_RGB[..., 2] - self.mean[2]
        img_BGR[..., 1] = img_RGB[..., 1] - self.mean[1]
        img_BGR[..., 2] = img_RGB[..., 0] - self.mean[0]

        img  = torch.from_numpy(img_BGR).permute(2,0,1)
        if mask is not None:
            mask = torch.from_numpy(mask).long()
        if aux_mask is not None:
            aux_mask = torch.from_numpy(aux_mask).long()
        return img, mask, aux_mask