import torch
import numpy as np
from scipy.ndimage import distance_transform_edt

def compute_mae(pred, mask):
    pred = torch.sigmoid(pred)
    return torch.abs(pred - mask).mean().item()

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

def compute_ephi(pred, mask, threshold=0.5):
    pred = torch.sigmoid(pred)
    pred = (pred > threshold).float()
    mask = (mask > threshold).float()
    intersection = ((pred == mask) & (mask == 1)).float().sum()
    union = ((pred == 1) | (mask == 1)).float().sum()
    return (intersection / union).item() if union > 0 else 0

def compute_fbw(pred, mask, beta2=0.3**2, threshold=0.5):
    pred = torch.sigmoid(pred)
    pred = (pred > threshold).float()
    mask = (mask > threshold).float()
    fbw_scores = []
    pred_np = pred.cpu().numpy()
    mask_np = mask.cpu().numpy()
    for i in range(pred_np.shape[0]):
        fg = mask_np[i,0]
        bg = 1 - fg
        d_fg = distance_transform_edt(bg)
        d_bg = distance_transform_edt(fg)
        weight = np.ones_like(fg)
        weight[fg == 0] = 1 + 5 * np.exp(-d_fg[fg == 0]**2 / 2)
        weight[fg == 1] = 1 + 5 * np.exp(-d_bg[fg == 1]**2 / 2)
        tp = (pred_np[i,0] * fg * weight).sum()
        fp = (pred_np[i,0] * (1 - fg) * weight).sum()
        fn = ((1 - pred_np[i,0]) * fg * weight).sum()
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        fbw = (1 + beta2) * precision * recall / (beta2 * precision + recall + 1e-8)
        fbw_scores.append(fbw)
    return np.mean(fbw_scores)

def compute_accuracy(pred, mask, threshold=0.5):
    pred = torch.sigmoid(pred)
    pred = (pred > threshold).float()
    mask = (mask > threshold).float()
    correct = (pred == mask).float().sum()
    total = torch.numel(pred)
    return (correct / total).item()

def dice_loss(pred, mask, threshold=0.5, smooth=1e-5):
    pred = torch.sigmoid(pred)
    pred = (pred > threshold).float()
    mask = (mask > threshold).float()
    intersection = (pred * mask).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + mask.sum(dim=(2, 3))
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()

def iou_score(pred, mask, threshold=0.5, smooth=1e-5):
    pred = torch.sigmoid(pred)
    pred = (pred > threshold).float()
    mask = (mask > threshold).float()
    intersection = (pred * mask).sum(dim=(2, 3))
    union = ((pred + mask) >= 1).float().sum(dim=(2, 3))
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean().item()
