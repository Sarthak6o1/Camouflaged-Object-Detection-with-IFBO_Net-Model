import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def compute_mae(pred, gt):
    return torch.abs(pred - gt).mean().item()    

def compute_smeasure(pred, mask):
    pred = (pred > 0.5).float()
    mask = (mask > 0.5).float()
    alpha = 0.5
    y = mask.mean()
    if y == 0:
        return 1 - pred.mean().item()
    elif y == 1:
        return pred.mean().item()
    else:
        fg = pred * mask
        bg = (1 - pred) * (1 - mask)
        o_fg = fg.sum() / (mask.sum() + 1e-6)
        o_bg = bg.sum() / ((1 - mask).sum() + 1e-6)
        return alpha * o_fg + (1 - alpha) * o_bg
def compute_ephi(pred, mask):
    pred = (pred > 0.5).float()
    mask = (mask > 0.5).float()
    pred_mean = pred.mean()
    mask_mean = mask.mean()
    align_matrix = 2 * (pred - pred_mean) * (mask - mask_mean) / (
        (pred - pred_mean) ** 2 + (mask - mask_mean) ** 2 + 1e-8
    )
    enhanced = ((align_matrix + 1) ** 2) / 4
    return enhanced.mean().item()
def compute_fbw(pred, mask, beta2=1.0):
    pred = (pred > 0.5).float()
    mask = (mask > 0.5).float()
    tp = (pred * mask).sum()
    precision = tp / (pred.sum() + 1e-8)
    recall = tp / (mask.sum() + 1e-8)
    fbw = (1 + beta2) * precision * recall / (beta2 * precision + recall + 1e-8)
    return fbw.item()
def compute_accuracy(pred, mask):
    pred_bin = (pred > 0.5).float()
    mask_bin = (mask > 0.5).float()
    correct = (pred_bin == mask_bin).float().sum()
    total = torch.numel(pred)
    return (correct / total).item()

import torch.nn.functional as F

def loss_fn(pred, mask, edge_pred, edge_gt):
    bce_main = F.binary_cross_entropy(pred, mask)
    bce_edge = F.binary_cross_entropy(edge_pred, edge_gt)
    intersection = (pred * mask).sum(dim=(1, 2, 3))
    union = (pred + mask).sum(dim=(1, 2, 3)) - intersection
    iou_loss = 1 - (intersection + 1e-6) / (union + 1e-6)
    iou_loss = iou_loss.mean()

    return 1.0 * bce_main + 0.5 * bce_edge + 1.0 * iou_loss
