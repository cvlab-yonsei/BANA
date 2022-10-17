import os
import torch
import collections
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image
from torch.utils.data import Dataset


CLASSES = (
    "background", 
    "aeroplane", 
    "bicycle", 
    "bird", 
    "boat", 
    "bottle", 
    "bus", 
    "car", 
    "cat", 
    "chair", 
    "cow", 
    "diningtable", 
    "dog", 
    "horse", 
    "motorbike",
    "person",
    "pottedplant", 
    "sheep",
    "sofa", 
    "train",
    "tvmonitor"
)


class VOC_box(Dataset):
    def __init__(self, cfg, transforms=None):
        if cfg.DATA.MODE == "train":
            txt_name = "train_aug.txt"
        if cfg.DATA.MODE == "val":
            txt_name = "val.txt"
        
        f_path = os.path.join(cfg.DATA.ROOT, "ImageSets/Segmentation", txt_name)
        self.filenames  = [x.split('\n')[0] for x in open(f_path)]
        self.transforms = transforms
        
        self.img_path  = os.path.join(cfg.DATA.ROOT, "JPEGImages/{}.jpg")
        self.xml_path  = os.path.join(cfg.DATA.ROOT, "Annotations/{}.xml")
        self.mask_path = os.path.join(cfg.DATA.ROOT, "BgMaskfromBoxes/{}.png")
        self.len = len(self.filenames)
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        fn  = self.filenames[index]
        img = np.array(Image.open(self.img_path.format(fn)), dtype=np.float32) 
        bboxes  = self.load_bboxes(self.xml_path.format(fn))
        bg_mask = np.array(Image.open(self.mask_path.format(fn)), dtype=np.int64)
        if self.transforms is not None:
            img, bboxes, bg_mask = self.transforms(img, bboxes, bg_mask)
        return img, bboxes, bg_mask

    def parse_voc_xml(self, node):
        voc_dict = {}
        children = list(node)
        if children:
            def_dic = collections.defaultdict(list)
            for dc in map(self.parse_voc_xml, children):
                for ind, v in dc.items():
                    def_dic[ind].append(v)
            voc_dict = {
                node.tag:
                    {ind: v[0] if len(v) == 1 else v
                     for ind, v in def_dic.items()}
            }
        if node.text:
            text = node.text.strip()
            if not children:
                voc_dict[node.tag] = text
        return voc_dict

    def load_bboxes(self, xml_path):
        XML = self.parse_voc_xml(ET.parse(xml_path).getroot())["annotation"]["object"]
        if not isinstance(XML, list):
            XML = [XML]
        bboxes = []
        for xml in XML:
            bb_wmin = float(xml["bndbox"]["xmin"])
            bb_wmax = float(xml["bndbox"]["xmax"])
            bb_hmin = float(xml["bndbox"]["ymin"])
            bb_hmax = float(xml["bndbox"]["ymax"])
            cls_num = CLASSES.index(xml["name"])
            bboxes.append([bb_wmin, bb_hmin, bb_wmax, bb_hmax, cls_num])
        return np.array(bboxes).astype("float32")


class VOC_mask(Dataset):
    def __init__(self, cfg, transforms=None):
        if cfg.DATA.MODE == "train":
            txt_name      = "train_aug.txt"
            annot_folders = cfg.DATA.PSEUDO_LABEL_FOLDER
        elif cfg.DATA.MODE == "val":
            txt_name      = "val.txt"
            annot_folders = "SegmentationClassAug"
        elif cfg.DATA.MODE == "test":
            txt_name      = "test.txt"
            annot_folders = None
        else:
            raise ValueError("Invalid cfg.DATA.MODE")

        f_path = os.path.join(cfg.DATA.ROOT, "ImageSets/Segmentation", txt_name)
        self.filenames  = [x.split('\n')[0] for x in open(f_path)]
        self.transforms = transforms
        self.img_path  = os.path.join(cfg.DATA.ROOT, "JPEGImages", "{}.jpg")
        if isinstance(annot_folders, list):
            self.mask_path = [os.path.join(cfg.DATA.ROOT, f"Generation/{folder}", "{}.png") for folder in annot_folders]
        elif annot_folders is not None:
            self.mask_path = os.path.join(cfg.DATA.ROOT, annot_folders, "{}.png")
        else:
            self.mask_path = None
        self.len = len(self.filenames)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        fn = self.filenames[index]
        img = np.array(Image.open(self.img_path.format(fn)), dtype=np.float32)

        if self.mask_path is not None:            
            if isinstance(self.mask_path, list):
                mask = [np.array(Image.open(mp.format(fn)), dtype=np.int64) for mp in self.mask_path]
                mask_crf = mask[0]
                mask_ret = mask[1]
                if self.transforms != None:
                    img, mask_crf, mask_ret = self.transforms(img, mask_crf, mask_ret)

                mask_fin = torch.full(mask_crf.shape, 255, dtype=torch.long)
                valid_regions = mask_crf==mask_ret
                mask_fin[valid_regions] = torch.clone(mask_crf[valid_regions])
                return img, mask_fin, mask_crf

            else:
                mask = np.array(Image.open(self.mask_path.format(fn)), dtype=np.int64)
                if self.transforms != None:
                    img, mask, _ = self.transforms(img, mask)
                return img, mask
            
        else:
            if self.transforms != None:
                 img,_,_ =self.transforms(img, None, None)
            return img
