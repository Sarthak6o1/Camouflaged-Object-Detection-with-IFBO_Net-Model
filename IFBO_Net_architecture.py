import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.swin_transformer import swin_tiny_patch4_window7_224
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import os

try:
    import timm
except ImportError:
    !pip install timm
    import timm

class SwinEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = swin_tiny_patch4_window7_224(pretrained=True)

    def forward(self, x):
        features = []
        x = self.encoder.patch_embed(x)

        x = self.encoder.layers[0](x)
        features.append(x.permute(0, 3, 1, 2))

        x = self.encoder.layers[1](x)
        features.append(x.permute(0, 3, 1, 2))

        x = self.encoder.layers[2](x)
        features.append(x.permute(0, 3, 1, 2))

        x = self.encoder.layers[3](x)
        features.append(x.permute(0, 3, 1, 2))

        return features

import torch.nn as nn
class FOM(nn.Module):
    def __init__(self, in_channels, out_channels=32):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.2)
        self.drop = nn.Dropout2d(0.2)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.drop(x)
        return x

import torch.nn as nn

class FID(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv3x3_1_s1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1_s1 = nn.BatchNorm2d(out_channels)
        self.act1_s1 = nn.LeakyReLU(0.2)
        self.drop1_s1 = nn.Dropout2d(0.2)

        self.conv3x3_1_s2 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1_s2 = nn.BatchNorm2d(out_channels)
        self.act1_s2 = nn.LeakyReLU(0.2)
        self.drop1_s2 = nn.Dropout2d(0.2)

        self.fusion_conv = nn.Conv2d(out_channels * 2, out_channels, 1)
        self.fusion_bn = nn.BatchNorm2d(out_channels)
        self.fusion_act = nn.LeakyReLU(0.2)
        self.fusion_drop = nn.Dropout2d(0.2)

    def forward(self, S1, S2):
        x1 = self.conv3x3_1_s1(S1)
        x1 = self.bn1_s1(x1)
        x1 = self.act1_s1(x1)
        x1 = self.drop1_s1(x1)

        x2 = self.conv3x3_1_s2(S2)
        x2 = self.bn1_s2(x2)
        x2 = self.act1_s2(x2)
        x2 = self.drop1_s2(x2)

        x2_upsampled = F.interpolate(x2, size=x1.shape[2:], mode='bilinear', align_corners=False)

        fused_features = torch.cat([x1, x2_upsampled], dim=1)

        output = self.fusion_conv(fused_features)
        output = self.fusion_bn(output)
        output = self.fusion_act(output)
        output = self.fusion_drop(output)

        return output

import torch.nn as nn

class FHIM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.2)
        self.drop = nn.Dropout2d(0.2)

    def forward(self, *features):
        target_size = features[0].shape[2:]
        upsampled_features = []

        for feature in features:
            if feature.shape[2:] != target_size:
                upsampled_feature = F.interpolate(feature, size=target_size, mode='bilinear', align_corners=False)
                upsampled_features.append(upsampled_feature)
            else:
                upsampled_features.append(feature)

        x = torch.cat(upsampled_features, dim=1)

        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.drop(x)
        return x

import torch.nn as nn
import torch.nn.functional as F

class BRM(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(in_channels)
        self.act = nn.LeakyReLU(0.2)
        self.drop = nn.Dropout2d(0.2)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.drop(x)
        dilated = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        eroded = -F.max_pool2d(-x, kernel_size=3, stride=1, padding=1)
        feature = dilated - eroded
        return feature

import torch.nn as nn
import torch.nn.functional as F
class IFBONet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = SwinEncoder()
        self.fom = nn.ModuleList([FOM(96), FOM(192), FOM(384), FOM(768)])
        self.fid = FID(32, 32)
        self.fhim = FHIM(32 * 3, 32)
        self.brm = BRM(32)
        self.final = nn.Conv2d(32, 1, 1)
        self.edge_final = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        feats = self.encoder(x)

        c = [self.fom[i](feats[i]) for i in range(4)]

        f_fid = self.fid(c[2], c[3])

        f_fhim = self.fhim(c[0], c[1], f_fid)

        f_brm = self.brm(f_fhim)

        mask_raw = self.final(f_brm)
        edge_raw = self.edge_final(f_brm)

        mask_pred = F.interpolate(mask_raw, size=(224, 224), mode='bilinear', align_corners=False)
        edge_pred = F.interpolate(edge_raw, size=(224, 224), mode='bilinear', align_corners=False)

        mask_output = torch.sigmoid(mask_pred)
        edge_output = torch.sigmoid(edge_pred)

        return mask_output, edge_output
