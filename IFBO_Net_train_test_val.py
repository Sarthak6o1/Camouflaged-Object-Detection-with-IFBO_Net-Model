# -*- coding: utf-8 -*-
"""IFBO_NET (1).ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1xXLH_hsA6WQxPAPYLPFxzxaIvt7Rekmv
"""

import torch

print(torch.__version__)
print(torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
else:
    print("CUDA not available. Running on CPU.")

from google.colab import drive
drive.mount('/content/drive')

from google.colab import drive
drive.mount('/content/drive')
!unzip "/content/drive/MyDrive/archive.zip" -d /content/

import os
print(os.listdir('/content/COD10K-v3'))

train_path = '/content/COD10K-v3/Train'
test_path = '/content/COD10K-v3/Test'

def get_subfolders(base_path):
    return {
        'image': f'{base_path}/Image',
        'gt_object': f'{base_path}/GT_Object',
        'gt_edge': f'{base_path}/GT_Edge',
        'gt_instance': f'{base_path}/GT_Instance'
    }

train_folders = get_subfolders(train_path)
test_folders = get_subfolders(test_path)

import os

def count_files(folder_dict):
    for key, folder in folder_dict.items():
        print(f"{key}: {len(os.listdir(folder))} files")

print("Train set:")
count_files(train_folders)
print("\nTest set:")
count_files(test_folders)

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from torch.utils.data import random_split

class COD10KDataset(Dataset):
    def __init__(self, folders, transform=None, mask_transform=None):
        self.image_paths = sorted(os.listdir(folders['image']))
        self.image_dir = folders['image']
        self.mask_dir = folders['gt_object']
        self.transform = transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_name = self.image_paths[idx]
        img_path = os.path.join(self.image_dir, image_name)
        mask_path = os.path.join(self.mask_dir, image_name.replace('.jpg', '.png'))

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

mask_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

full_dataset = COD10KDataset(train_folders, transform=transform, mask_transform=mask_transform)

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

batch_size = 48
num_workers = 2

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of validation samples: {len(val_dataset)}")
print(f"Number of training batches: {len(train_loader)}")
print(f"Number of validation batches: {len(val_loader)}")

# Create test dataset and test loader using the same transforms
test_dataset = COD10KDataset(test_folders, transform=transform, mask_transform=mask_transform)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers
)

print(f"Number of test samples: {len(test_dataset)}")
print(f"Number of test batches: {len(test_loader)}")

import matplotlib.pyplot as plt

def show_all_types(folders, indices):
    n = len(indices)
    fig, axes = plt.subplots(n, 4, figsize=(12, 5))
    type_names = ['Image', 'GT_Object', 'GT_Edge', 'GT_Instance']
    image_files = sorted(os.listdir(folders['image']))
    for row, idx in enumerate(indices):
        img_name = image_files[idx]
        paths = [
            os.path.join(folders['image'], img_name),
            os.path.join(folders['gt_object'], img_name.replace('.jpg', '.png')),
            os.path.join(folders['gt_edge'], img_name.replace('.jpg', '.png')),
            os.path.join(folders['gt_instance'], img_name.replace('.jpg', '.png'))
        ]
        for col, (path, tname) in enumerate(zip(paths, type_names)):
            img = Image.open(path)
            axes[row, col].imshow(img if col == 0 else img, cmap=None if col == 0 else 'gray')
            axes[row, col].set_title(tname)
            axes[row, col].axis('off')
    plt.tight_layout()
    plt.show()

show_all_types(train_folders, indices=range(5))

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

class EarlyStopping:
    def __init__(self, patience=3, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    def __call__(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def compute_mae(pred, mask):
    return torch.abs(pred.sigmoid() - mask).mean().item()

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

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = IFBONet().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
early_stopper = EarlyStopping(patience=10, min_delta=1e-4)

num_epochs = 60

train_losses, val_losses = [], []
train_maes, val_maes = [], []
train_salphas, val_salphas = [], []
train_ephis, val_ephis = [], []
train_fbps, val_fbps = [], []
train_accs, val_accs = [], []

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

    model.eval()
    val_loss = 0
    val_mae = 0
    val_salpha = 0
    val_ephi = 0
    val_fbw = 0
    val_acc = 0

    with torch.no_grad():
        for img, mask in val_loader:
            img, mask = img.to(device), mask.to(device)
            pred, edge_pred = model(img)
            loss = loss_fn(pred, mask, edge_pred, mask)
            val_loss += loss.item()
            val_mae += compute_mae(pred, mask)
            val_salpha += compute_smeasure(pred, mask)
            val_ephi += compute_ephi(pred, mask)
            val_fbw += compute_fbw(pred, mask)
            val_acc += compute_accuracy(pred, mask)

    n_val = len(val_loader)
    val_losses.append(val_loss / n_val)
    val_maes.append(val_mae / n_val)
    val_salphas.append(val_salpha / n_val)
    val_ephis.append(val_ephi / n_val)
    val_fbps.append(val_fbw / n_val)
    val_accs.append(val_acc / n_val)

    print(f"Epoch {epoch+1}:")
    print(f"  Train Loss: {train_losses[-1]:.4f}, Acc: {train_accs[-1]:.4f}, MAE: {train_maes[-1]:.4f}, Sα: {train_salphas[-1]:.4f}, Eϕ: {train_ephis[-1]:.4f}, Fβw: {train_fbps[-1]:.4f}")
    print(f"  Val   Loss: {val_losses[-1]:.4f}, Acc: {val_accs[-1]:.4f}, MAE: {val_maes[-1]:.4f}, Sα: {val_salphas[-1]:.4f}, Eϕ: {val_ephis[-1]:.4f}, Fβw: {val_fbps[-1]:.4f}")

    scheduler.step()

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

def apply_jet_colormap(mask_tensor):
    mask_np = mask_tensor.squeeze().detach().cpu().numpy()
    mask_norm = (mask_np - mask_np.min()) / (mask_np.max() - mask_np.min() + 1e-8)
    jet_mask = plt.get_cmap('jet')(mask_norm)[:, :, :3]
    jet_mask = (jet_mask * 255).astype(np.uint8)
    return Image.fromarray(jet_mask)

def visualize_val_grid(model, val_loader, device, images_per_row=8):
    model.eval()
    with torch.no_grad():
        batch_images, batch_preds, batch_masks = [], [], []

        for images, masks in val_loader:
            images = images.to(device)
            preds, _ = model(images)

            for i in range(images.size(0)):
                if masks[i].sum().item() == 0:
                    continue

                batch_images.append(images[i].cpu())
                batch_preds.append(preds[i].cpu())
                batch_masks.append(masks[i].cpu())

                # Display when we have a full row
                if len(batch_images) == images_per_row:
                    # Create figure with 3 rows and 8 columns
                    fig, axs = plt.subplots(3, images_per_row, figsize=(12, 8))

                    for j in range(images_per_row):
                        # Input image
                        img_np = batch_images[j].permute(1, 2, 0).numpy()
                        axs[0, j].imshow(img_np)
                        axs[0, j].axis('off')

                        # Prediction
                        pred_img = apply_jet_colormap(batch_preds[j])
                        axs[1, j].imshow(pred_img)
                        axs[1, j].axis('off')

                        # Ground truth
                        mask_np = batch_masks[j].squeeze().numpy()
                        axs[2, j].imshow(mask_np, cmap='gray')
                        axs[2, j].axis('off')

                    # Add row labels
                    fig.text(0.05, 0.75, 'Input', ha='center', va='center', rotation='vertical', fontsize=12)
                    fig.text(0.05, 0.5, 'Prediction', ha='center', va='center', rotation='vertical', fontsize=12)
                    fig.text(0.05, 0.25, 'Ground Truth', ha='center', va='center', rotation='vertical', fontsize=12)

                    plt.tight_layout(rect=[0.05, 0, 1, 1])  # Make space for labels
                    plt.show()
                    batch_images, batch_preds, batch_masks = [], [], []

        # Display remaining images
        if batch_images:
            n = len(batch_images)
            fig, axs = plt.subplots(3, n, figsize=(3*n, 8))
            if n == 1:  # Handle single image case
                axs = np.array([axs[0]], [axs[1]], [[axs[2]]])

            for j in range(n):
                # Input image
                img_np = batch_images[j].permute(1, 2, 0).numpy()
                axs[0, j].imshow(img_np)
                axs[0, j].axis('off')

                # Prediction
                pred_img = apply_jet_colormap(batch_preds[j])
                axs[1, j].imshow(pred_img)
                axs[1, j].axis('off')

                # Ground truth
                mask_np = batch_masks[j].squeeze().numpy()
                axs[2, j].imshow(mask_np, cmap='gray')
                axs[2, j].axis('off')

            # Add row labels
            fig.text(0.05, 0.75, 'Input', ha='center', va='center', rotation='vertical', fontsize=12)
            fig.text(0.05, 0.5, 'Prediction', ha='center', va='center', rotation='vertical', fontsize=12)
            fig.text(0.05, 0.25, 'Ground Truth', ha='center', va='center', rotation='vertical', fontsize=12)

            plt.tight_layout(rect=[0.05, 0, 1, 1])
            plt.show()

# Usage:
visualize_val_grid(model, test_loader, device)

!pip install gradio
import gradio as gr

def predict(image):
    input_image = transform(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        mask_pred, _ = model(input_image)

    mask_output = mask_pred.squeeze().cpu()
    binary_mask = (mask_output > 0.5).numpy().astype(np.uint8) * 255
    predicted_mask_image = Image.fromarray(binary_mask, 'L')

    return predicted_mask_image

interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"), # Input is a PIL Image
    outputs=gr.Image(type="pil", label="Predicted Mask"), # Output is a PIL Image
    title="COD10K Saliency Prediction with IFBONet",
    description="Upload an image to get the predicted saliency mask using the trained IFBONet model."
)

interface.launch(share=True)
