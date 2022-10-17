import os
import sys
import random
import logging
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import datasets.transforms_mask as Tr
from datasets.voc import VOC_mask, CLASSES
from configs.defaults import _C
from models.SegNet import DeepLab_LargeFOV 
from utils.densecrf import DENSE_CRF
from utils.sem_seg_evaluation import SemSegEvaluator

logger = logging.getLogger("stage3_vgg")

def main(cfg):
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", datefmt="%m/%d %H:%M:%S")
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    fh = logging.FileHandler(f"./logs/{cfg.NAME}.txt")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.info(" ".join(['\n{}: {}'.format(k, v) for k,v in cfg.items()]))

    model = DeepLab_LargeFOV(cfg.DATA.NUM_CLASSES).cuda()
    model.load_state_dict(torch.load(f"./weights/SegNet_VGG.pt"))
    
    scores = validation(cfg, model, True, True)
    logger.info("--- Vaidation Results ---")
    logger.info(f"# of samples: {scores['Total samples']}")
    logger.info(f" pixel Acc. : {scores['pACC']:.2f}")
    logger.info(f"  mean Acc. : {scores['mACC']:.2f}")
    logger.info(f"  mean IoU  : {scores['mIoU']:.2f}")
    logger.info("per-class IoU")
    for ind, cls_iou in enumerate(scores['per IoU']):
        logger.info(f"{ind:2d}-{CLASSES[ind]:<11}: {cls_iou:.2f}")


def validation(cfg, model, do_aug=True, do_crf=True):
    def get_logit(model, img, img_size):
        features = model.get_features(img.cuda()) # (bs, dims, fH, fW)
        normed_f = F.normalize(features, dim=1)
        normed_w = F.normalize(model.classifier.weight, dim=1) # (num_cls, dims, 1, 1)
        logit = F.conv2d(normed_f, normed_w) # (bs, num_cls, fH, fW)
        logit = F.interpolate(logit, img_size, mode='bilinear', align_corners=True)
        return logit

    cfg.DATA.MODE = 'val'
    validset = VOC_mask(cfg, Tr.Normalize_Caffe())
    valid_loader = DataLoader(validset, batch_size=1, num_workers=4)

    evaluator = SemSegEvaluator(cfg.DATA.NUM_CLASSES)
    evaluator.reset()

    if do_aug:
        scales = [0.5, 0.75, 1.5]
        tot_num_infer = 2 * ( 1 + len(scales) ) 

    if do_crf:
        bi_w, bi_xy_std, bi_rgb_std, pos_w, pos_xy_std = cfg.MODEL.DCRF
        dCRF = DENSE_CRF(bi_w, bi_xy_std, bi_rgb_std, pos_w, pos_xy_std)

    with torch.no_grad():
        model.eval()
        for img, mask in valid_loader:
            img_size = img.shape[-2:]

            logit = get_logit(model, img, img_size)

            if do_aug:
                logit += get_logit(model, torch.clone(img).flip(dims=[-1]), img_size).flip(dims=[-1])
                for scale in scales:
                    r_img = F.interpolate(img, scale_factor=scale, mode='bilinear', align_corners=True)
                    logit += get_logit(model, r_img, img_size)
                    rf_img = torch.clone(r_img).flip(dims=[-1])
                    logit += get_logit(model, rf_img, img_size).flip(dims=[-1])
                logit = logit / tot_num_infer 
            
            if do_crf:
                prob = torch.softmax(logit * cfg.MODEL.TEMP, dim=1)
                pred = dCRF.inference(
                    img.squeeze().permute(1,2,0).numpy().astype("uint8"),
                    prob.squeeze().detach().cpu().numpy()
                ).argmax(axis=0)
            else:
                pred = logit.argmax(dim=1).squeeze().detach().cpu().numpy()

            evaluator.process(pred, mask.squeeze().numpy())

        results = evaluator.evaluate()
        return results


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file")
    parser.add_argument("--gpu-id", type=str, default="0", help="select a GPU index")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    cfg = _C.clone()
    cfg.merge_from_file(args.config_file)
    # cfg.freeze()
    main(cfg)
