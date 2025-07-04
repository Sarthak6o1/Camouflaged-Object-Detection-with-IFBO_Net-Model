import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SegNet().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
criterion = BCEDiceLoss()

num_epochs = 50

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    train_mae = 0
    train_salpha = 0
    train_ephi = 0
    train_fbw = 0
    train_acc = 0
    train_dice = 0
    train_iou = 0

    for img, mask in tqdm(train_loader, desc=f"Train Epoch {epoch+1}/{num_epochs}"):
        img, mask = img.to(device), mask.to(device)
        pred = model(img)
        loss = criterion(pred, mask)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        train_loss += loss.item()
        train_mae += compute_mae(pred, mask)
        train_salpha += compute_smeasure(pred, mask)
        train_ephi += compute_ephi(pred, mask)
        train_fbw += compute_fbw(pred, mask)
        train_acc += compute_accuracy(pred, mask)
        train_dice += 1 - dice_loss(pred, mask).item()
        train_iou += iou_score(pred, mask)

    n_train = len(train_loader)
    print(f"Epoch {epoch+1} Train: "
          f"Loss: {train_loss/n_train:.4f}, "
          f"Acc: {train_acc/n_train:.4f}, "
          f"MAE: {train_mae/n_train:.4f}, "
          f"Sα: {train_salpha/n_train:.4f}, "
          f"Eϕ: {train_ephi/n_train:.4f}, "
          f"Fβw: {train_fbw/n_train:.4f}, "
          f"Dice: {train_dice/n_train:.4f}, "
          f"IoU: {train_iou/n_train:.4f}")

    scheduler.step()
