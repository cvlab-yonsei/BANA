import os
from PIL import Image
from torch.utils.data import Dataset

class VOC(Dataset):
    def __init__(self, cfg, transforms=None):
        self.train = False
        if cfg.DATA.MODE == "train_weak":
            txt_name = "train_aug.txt"
            self.train = True
        if cfg.DATA.MODE == "val":
            txt_name = "val.txt"
        if cfg.DATA.MODE == "test":
            txt_name = "test.txt"
            
        f_path = os.path.join(cfg.DATA.ROOT, "ImageSets/Segmentation", txt_name)
        self.filenames = [x.split('\n')[0] for x in open(f_path)]
        self.transforms = transforms
        
        self.annot_folders = ["SegmentationClassAug"]
        if cfg.DATA.PSEUDO_LABEL_PATH:
            self.annot_folders = cfg.DATA.PSEUDO_LABEL_PATH
        if cfg.DATA.MODE == "test":
            self.annot_folders = None
        
        self.img_path  = os.path.join(cfg.DATA.ROOT, "JPEGImages", "{}.jpg")
        if self.annot_folder is not None:
            self.mask_paths = [os.path.join(cfg.DATA.ROOT, folder, "{}.png") for folder in self.annot_folders]
        self.len = len(self.filenames)
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        fn  = self.filenames[index]
        img = Image.open(self.img_path.format(fn))
        if self.annot_folder is not None:
            masks = [Image.open(mp.format(fn)) for mp in self.mask_paths]
        else:
            masks = None
            
        if self.transforms != None:
            img, masks = self.transforms(img, masks)
        
        return img, masks
