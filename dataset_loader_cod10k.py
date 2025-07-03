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

train_dataset = COD10KDataset(train_folders, transform=transform, mask_transform=mask_transform)
test_dataset = COD10KDataset(test_folders, transform=transform, mask_transform=mask_transform)

batch_size = 48
num_workers = 2

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of test samples: {len(test_dataset)}")
print(f"Number of training batches: {len(train_loader)}")
print(f"Number of test batches: {len(test_loader)}")

def count_files(folder_dict):
    for key, folder in folder_dict.items():
        print(f"{key}: {len(os.listdir(folder))} files")

print("Train set:")
count_files(train_folders)

print("\nTest set:")
count_files(test_folders)
