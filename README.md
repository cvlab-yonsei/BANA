# BANA

This is the implementation of the paper "Background-Aware Pooling and Noise-Aware Loss for Weakly-Supervised Semantic Segmentation".

For more information, please checkout the project site [[website](https://cvlab.yonsei.ac.kr/projects/BANA/)] and the paper [[PDF](https://arxiv.org/pdf/2104.00905.pdf)].

## Requirements
* Python >= 3.6
* PyTorch >= 1.3.0
* yacs (https://github.com/rbgirshick/yacs)

## Getting started

The folder ```data``` should be like this
```
    data   
    └── VOCdevkit
        └── VOC2012
            ├── JPEGImages
            ├── SegmentationClassAug
            ├── Annotations
            ├── ImageSets
            ├── BgMaskfromBoxes
            └── Generation
```


```bash
git clone https://github.com/cvlab-yonsei/BANA.git
cd BANA
python stage1.py --config-file configs/stage1.yml --gpu-id 0 # For training a classification network
python stage2.py --config-file configs/stage2.yml --gpu-id 0 # For generating pseudo labels
```

## Bibtex
```
@inproceedings{oh2021background,
  title     = {Background-Aware Pooling and Noise-Aware Loss for Weakly-Supervised Semantic Segmentation},
  author    = {Oh, Youngmin and Kim, Beomjun and Ham, Bumsub},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2021},
}
```
