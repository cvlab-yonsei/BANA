#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2017-11-19

from __future__ import absolute_import, print_function

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
# try:
#     from encoding.nn import SyncBatchNorm
#     _BATCH_NORM = SyncBatchNorm
# except:
#     _BATCH_NORM = nn.BatchNorm2d

_BOTTLENECK_EXPANSION = 4

class _ConvBnReLU(nn.Sequential):
    """
    Cascade of 2D convolution, batch norm, and ReLU.
    """
#     BATCH_NORM = _BATCH_NORM
    def __init__(
        self, in_ch, norm_type, out_ch, kernel_size, stride, padding, dilation, relu=True
    ):
        super(_ConvBnReLU, self).__init__()
        self.add_module(
            "conv",
            nn.Conv2d(
                in_ch, out_ch, kernel_size, stride, padding, dilation, bias=False
            ),
        )
        self.add_module("bn", norm_type(out_ch, eps=1e-5, momentum=0.999))
        if relu:
            self.add_module("relu", nn.ReLU())


class _Bottleneck(nn.Module):
    """
    Bottleneck block of MSRA ResNet.
    """
    def __init__(self, in_ch, norm_type, out_ch, stride, dilation, downsample):
        super(_Bottleneck, self).__init__()
        mid_ch = out_ch // _BOTTLENECK_EXPANSION
        self.reduce = _ConvBnReLU(in_ch, norm_type, mid_ch, 1, stride, 0, 1, True)
        self.conv3x3 = _ConvBnReLU(mid_ch, norm_type, mid_ch, 3, 1, dilation, dilation, True)
        self.increase = _ConvBnReLU(mid_ch, norm_type, out_ch, 1, 1, 0, 1, False)
        self.shortcut = (
            _ConvBnReLU(in_ch, norm_type, out_ch, 1, stride, 0, 1, False)
            if downsample
            else lambda x: x  # identity
        )
    def forward(self, x):
        h = self.reduce(x)
        h = self.conv3x3(h)
        h = self.increase(h)
        h += self.shortcut(x)
        return F.relu(h)

class _ResLayer(nn.Sequential):
    """
    Residual layer with multi grids
    """
    def __init__(self, n_layers, in_ch, norm_type, out_ch, stride, dilation, multi_grids=None):
        super(_ResLayer, self).__init__()
        if multi_grids is None:
            multi_grids = [1 for _ in range(n_layers)]
        else:
            assert n_layers == len(multi_grids)

        # Downsampling is only in the first block
        for i in range(n_layers):
            self.add_module(
                "block{}".format(i + 1),
                _Bottleneck(
                    in_ch=(in_ch if i == 0 else out_ch),
                    norm_type=norm_type,
                    out_ch=out_ch,
                    stride=(stride if i == 0 else 1),
                    dilation=dilation * multi_grids[i],
                    downsample=(True if i == 0 else False),
                ),
            )

class _Stem(nn.Sequential):
    """
    The 1st conv layer.
    Note that the max pooling is different from both MSRA and FAIR ResNet.
    """
    def __init__(self, norm_type, out_ch):
        super(_Stem, self).__init__()
        self.add_module("conv1", _ConvBnReLU(3, norm_type, out_ch, 7, 2, 3, 1))
        self.add_module("pool", nn.MaxPool2d(3, 2, 1, ceil_mode=True))

class _Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class RES101(nn.Sequential):
    def __init__(self, sync_bn):
        super().__init__()
        if sync_bn:
            norm_type = SynchronizedBatchNorm2d
        else:
            norm_type = nn.BatchNorm2d
        n_blocks = [3,4,23,3]
        ch = [64 * 2 ** p for p in range(6)]
        self.add_module("layer1", _Stem(norm_type, ch[0]))
        self.add_module("layer2", _ResLayer(n_blocks[0], ch[0], norm_type, ch[2], 1, 1))
        self.add_module("layer3", _ResLayer(n_blocks[1], ch[2], norm_type, ch[3], 2, 1))
        self.add_module("layer4", _ResLayer(n_blocks[2], ch[3], norm_type, ch[4], 1, 2))
        self.add_module("layer5", _ResLayer(n_blocks[3], ch[4], norm_type, ch[5], 1, 4))
#         self.add_module("pool5", nn.AdaptiveAvgPool2d(1))
#         self.add_module("flatten", _Flatten())
#         self.add_module("fc", nn.Linear(ch[5], n_classes))
        
    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, SynchronizedBatchNorm2d):
                m.eval()

    def forward(self, x, return_feature_maps=False):
        conv_out = []
        x = self.layer1(x)
        x = self.layer2(x); conv_out.append(x);
        x = self.layer3(x); conv_out.append(x);
        x = self.layer4(x); conv_out.append(x);
        x = self.layer5(x); conv_out.append(x);
        if return_feature_maps:
            return conv_out
        return x

    

class RES101_V3plus(nn.Sequential):
    def __init__(self, output_stride, sync_bn):
        super().__init__()
        if sync_bn:
            norm_type = SynchronizedBatchNorm2d
        else:
            norm_type = nn.BatchNorm2d
        n_blocks = [3,4,23,3]
        ch = [64 * 2 ** p for p in range(6)]
        if output_stride == 16:
            strides = [1, 2, 2, 1]
            dilations = [1, 1, 1, 2]
        if output_stride == 8:
            strides = [1, 2, 1, 1]
            dilations = [1, 1, 2, 4]
        self.add_module("layer1", _Stem(norm_type, ch[0]))
        self.add_module("layer2", _ResLayer(n_blocks[0], ch[0], norm_type, ch[2],
                                            strides[0], dilations[0]))
        self.add_module("layer3", _ResLayer(n_blocks[1], ch[2], norm_type, ch[3],
                                            strides[1], dilations[1]))
        self.add_module("layer4", _ResLayer(n_blocks[2], ch[3], norm_type, ch[4],
                                            strides[2], dilations[2]))
        self.add_module("layer5", _ResLayer(n_blocks[3], ch[4], norm_type, ch[5],
                                            strides[3], dilations[3]))
        
    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        low_level_feat = x
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x, low_level_feat

    

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation, BatchNorm):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            BatchNorm(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels, BatchNorm, global_avg_pool_bn=False):
        if global_avg_pool_bn: #If Batchsize is 1, error occur.
            super(ASPPPooling, self).__init__(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                BatchNorm(out_channels), 
                nn.ReLU())
        else:
            super(ASPPPooling, self).__init__(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)
    
class ASPP(nn.Module):
    def __init__(self, output_stride, sync_bn, global_avg_pool_bn=False, in_channels=2048, out_channels=256):
        super(ASPP, self).__init__()
        if sync_bn:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            BatchNorm(out_channels),
            nn.ReLU()))
        if output_stride == 16:
            atrous_rates = [6, 12, 18]
        if output_stride == 8:
            atrous_rates = [12, 24, 36]
        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate, BatchNorm))
        modules.append(ASPPPooling(in_channels, out_channels, BatchNorm, global_avg_pool_bn))
        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            BatchNorm(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d) or isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class VGG16(nn.Module):
    def __init__(self, dilation):
        self.conv1_1 = nn.Conv2d(3, 64, 3, 1, 1)
        self.conv1_2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.mp1 = nn.MaxPool2d(3, 2, 1)
        self.conv2_1 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.mp2 = nn.MaxPool2d(3, 2, 1)
        self.conv3_1 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv3_3 = nn.Conv2d(256, 256, 3, 1, 1)
        self.mp3 = nn.MaxPool2d(3, 2, 1)
        self.conv4_1 = nn.Conv2d(256, 512, 3, 1, 1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv4_3 = nn.Conv2d(512, 512, 3, 1, 1)
        self.mp4 = nn.MaxPool2d(3, 1, 1)
        self.conv5_1 = nn.Conv2d(512, 512, 3, 1, padding=2, dilation=2)
        self.conv5_2 = nn.Conv2d(512, 512, 3, 1, padding=2, dilation=2)
        self.conv5_3 = nn.Conv2d(512, 512, 3, 1, padding=2, dilation=2)
        self.mp5 = nn.MaxPool2d(3, 1, 1)
        self.pool5a = nn.AvgPool2d(3,1,1)
        self.fc6 = nn.Conv2d(512, 1024, 3, 1, padding=dilation, dilation=dilation)
        self.dropout1 = nn.Dropout2d(0.5)
        self.fc7 = nn.Conv2d(1024, 1024, 1)
        self.dropout2 = nn.Dropout2d(0.5)

    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.mp1(x)
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.mp2(x)
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        x = self.mp3(x)
        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.conv4_3(x))
        x = self.mp4(x)
        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = F.relu(self.conv5_3(x))
        x = self.mp5(x)
        x = self.pool5a(x)
        x = F.relu(self.fc6(x))
        x = self.dropout1(x)
        x = F.relu(self.fc7(x))
        x = self.dropout2(x)
        return x
