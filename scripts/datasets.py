import os
import sys
import numpy as np
import pandas as pd
import torch
import cv2 as cv
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import v2

class ImageDepthLidarDataset(Dataset):
    """
    Dataset class that combines images, depth data, and lidar scans.
    """
    def __init__(self, annotations_file, img_dir, depth_dir, lidar_dir):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.depth_dir = depth_dir
        self.lidar_dir = lidar_dir
        self.transform = v2.ToTensor()

    def __len__(self):
        return len(self.img_labels)

    def load_image(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = cv.imread(img_path, cv.IMREAD_COLOR)
        return self.transform(image)

    def load_depth(self, idx):
        depth_path = os.path.join(self.depth_dir, self.img_labels.iloc[idx, 0].replace(".jpg", ".npy")) # type: ignore
        depth_image = np.load(depth_path)  # Assuming depth is saved as .npy
        return torch.tensor(depth_image).unsqueeze(0).float()  # Add channel dimension

    def load_lidar(self, idx):
        lidar_path = os.path.join(self.lidar_dir, self.img_labels.iloc[idx, 0].replace(".jpg", ".npy")) # type: ignore
        lidar_scan = np.load(lidar_path)
        return torch.tensor(lidar_scan).float()

    def __getitem__(self, idx):
        image_tensor = self.load_image(idx)
        depth_tensor = self.load_depth(idx)
        lidar_tensor = self.load_lidar(idx)
        steering = self.img_labels.iloc[idx, 1].astype(np.float32) # type: ignore
        throttle = self.img_labels.iloc[idx, 2].astype(np.float32) # type: ignore
        return {
            'image': image_tensor,
            'depth': depth_tensor,
            'lidar': lidar_tensor,
            'steering': torch.tensor(steering),
            'throttle': torch.tensor(throttle)
        }

def prepare_dataloaders(annotations_file, img_dir, depth_dir, lidar_dir, batch_size=125):
    dataset = ImageDepthLidarDataset(annotations_file, img_dir, depth_dir, lidar_dir)
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train_data, test_data = random_split(dataset, [train_size, test_size])

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)
    return train_dataloader, test_dataloader
