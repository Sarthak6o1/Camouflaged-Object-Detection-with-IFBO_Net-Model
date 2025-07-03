import matplotlib.pyplot as plt
from PIL import Image
import os

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
import matplotlib.pyplot as plt

def show_images_and_heatmaps(images, masks, num_images=8):
    batch_size = images.shape[0]
    num_images = min(num_images, batch_size)
    fig, axes = plt.subplots(num_images, 5, figsize=(10, 10))
    if num_images == 1:
        axes = axes.reshape(1, 5)
    for i in range(num_images):
        axes[i, 0].imshow(images[i].permute(1, 2, 0).cpu())
        axes[i, 0].set_title(f'Image {i+1}')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(masks[i].squeeze().cpu(), cmap='gray')
        axes[i, 1].set_title(f'Mask {i+1}')
        axes[i, 1].axis('off')

        for c in range(3):
            heatmap = images[i, c].cpu()
            axes[i, c+2].imshow(heatmap, cmap='hot')
            axes[i, c+2].set_title(f'Channel {c+1} Heatmap')
            axes[i, c+2].axis('off')
    plt.tight_layout()
    plt.show()

show_images_and_heatmaps(images, masks, num_images=8)


