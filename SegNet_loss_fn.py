import torch
import torch.nn as nn
class BCEDiceLoss(nn.Module):
    def __init__(self, weight_bce=0.5, weight_dice=0.5, edge_weight=0.2):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.weight_bce = weight_bce
        self.weight_dice = weight_dice
        self.edge_weight = edge_weight

    def forward(self, pred, mask, edge_pred=None, edge_mask=None):
        bce = self.bce(pred, mask)
        smooth = 1e-5
        probs = torch.sigmoid(pred)
        intersection = (probs * mask).sum(dim=(2,3))
        union = probs.sum(dim=(2,3)) + mask.sum(dim=(2,3))
        dice = 1 - ((2. * intersection + smooth) / (union + smooth)).mean()
        loss = self.weight_bce * bce + self.weight_dice * dice
        if edge_pred is not None and edge_mask is not None:
            edge_bce = self.bce(edge_pred, edge_mask)
            loss = loss + self.edge_weight * edge_bce
        return loss
