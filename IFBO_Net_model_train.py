import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = IFBONet().to(device)
xavier_init = torch.nn.init.xavier_uniform_
for module in model.modules():
    if isinstance(module, torch.nn.Conv2d):
        xavier_init(module.weight)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

num_epochs = 20 ## use here more epochs

train_losses = []
train_maes = []
train_salphas = []
train_ephis = []
train_fbps = []
train_accs = []

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    epoch_mae = 0
    epoch_salpha = 0
    epoch_ephi = 0
    epoch_fbw = 0
    epoch_acc = 0

    for img, mask in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        img, mask = img.to(device), mask.to(device)
        pred, edge_pred = model(img)
        loss = loss_fn(pred, mask, edge_pred, mask)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        epoch_loss += loss.item()
        epoch_mae += compute_mae(pred, mask)
        epoch_salpha += compute_smeasure(pred, mask)
        epoch_ephi += compute_ephi(pred, mask)
        epoch_fbw += compute_fbw(pred, mask)
        epoch_acc += compute_accuracy(pred, mask)

    n_train = len(train_loader)
    train_losses.append(epoch_loss / n_train)
    train_maes.append(epoch_mae / n_train)
    train_salphas.append(epoch_salpha / n_train)
    train_ephis.append(epoch_ephi / n_train)
    train_fbps.append(epoch_fbw / n_train)
    train_accs.append(epoch_acc / n_train)

    print(f"Epoch {epoch+1}:")
    print(f"  Train Loss: {train_losses[-1]:.4f}, Acc: {train_accs[-1]:.4f}, MAE: {train_maes[-1]:.4f}, Sα: {train_salphas[-1]:.4f}, Eϕ: {train_ephis[-1]:.4f}, Fβw: {train_fbps[-1]:.4f}")

    scheduler.step()
