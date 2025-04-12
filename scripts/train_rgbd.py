import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import v2
import matplotlib.pyplot as plt
from torchvision.models import efficientnet_b2
import cv2 as cv
import torch.onnx

# Pass in command line arguments for data directory name
if len(sys.argv) != 2:
    print('Training script needs data!!!')
    sys.exit(1)  # Exit with an error code
else:
    data_datetime = sys.argv[1]

# Designate processing unit for CNN training
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {DEVICE} device")

class BearCartDataset(Dataset):
    """
    Customized dataset for RGB-D data.
    """
    def __init__(self, annotations_file, rgb_dir, depth_dir):
        self.img_labels = pd.read_csv(annotations_file)
        self.rgb_dir = rgb_dir
        self.depth_dir = depth_dir
        self.transform = v2.ToTensor()

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        # Load RGB image from column 0 in labels.csv
        rgb_name = self.img_labels.iloc[idx, 0]
        rgb_path = os.path.join(self.rgb_dir, rgb_name)
        rgb_image = cv.imread(rgb_path, cv.IMREAD_COLOR)
        if rgb_image is None:
            raise FileNotFoundError(f"Error: Could not read RGB image at {rgb_path}")
        rgb_image = cv.resize(rgb_image, (260, 260))

        # Load Depth image from column 1 in labels.csv
        depth_name = self.img_labels.iloc[idx, 1]
        depth_path = os.path.join(self.depth_dir, depth_name)
        depth_image = cv.imread(depth_path, cv.IMREAD_GRAYSCALE)
        if depth_image is None:
            raise FileNotFoundError(f"Error: Could not read Depth image at {depth_path}")
        depth_image = cv.resize(depth_image, (260, 260))

        # Stack RGB and Depth into a 4-channel tensor
        rgb_tensor = self.transform(rgb_image)
        depth_tensor = self.transform(depth_image).squeeze(0)
        image_tensor = torch.cat((rgb_tensor, depth_tensor.unsqueeze(0)), dim=0)

        # Steering and throttle values
        steering = self.img_labels.iloc[idx, 2].astype(np.float32)
        throttle = self.img_labels.iloc[idx, 3].astype(np.float32)

        return image_tensor.float(), steering, throttle

def train(dataloader, model, loss_fn, optimizer):
    model.train()
    num_used_samples = 0
    ep_loss = 0.
    for b, (im, st, th) in enumerate(dataloader):
        target = torch.stack((st, th), dim=-1)
        feature, target = im.to(DEVICE), target.to(DEVICE)
        pred = model(feature)
        batch_loss = loss_fn(pred, target)
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        num_used_samples += target.shape[0]
        print(f"batch loss: {batch_loss.item()} [{num_used_samples}/{len(dataloader.dataset)}]")
        ep_loss = (ep_loss * b + batch_loss.item()) / (b + 1)
    return ep_loss

def test(dataloader, model, loss_fn):
    model.eval()
    ep_loss = 0.
    with torch.no_grad():
        for b, (im, st, th) in enumerate(dataloader):
            target = torch.stack((st, th), dim=-1)
            feature, target = im.to(DEVICE), target.to(DEVICE)
            pred = model(feature)
            batch_loss = loss_fn(pred, target)
            ep_loss = (ep_loss * b + batch_loss.item()) / (b + 1)
    return ep_loss

def standard_loss(output, target):
    return ((output - target) ** 2).mean()

# MAIN
# Create a dataset
data_dir = os.path.join(os.path.dirname(sys.path[0]), 'data', data_datetime)
annotations_file = os.path.join(data_dir, 'labels.csv')
rgb_dir = os.path.join(data_dir, 'rgb_images')  # RGB images directory
depth_dir = os.path.join(data_dir, 'depth_images')  # Depth images directory
bearcart_dataset = BearCartDataset(annotations_file, rgb_dir, depth_dir)
print(f"data length: {len(bearcart_dataset)}")

# Create training and test dataloaders
train_size = round(len(bearcart_dataset) * 0.9)
test_size = len(bearcart_dataset) - train_size
print(f"train size: {train_size}, test size: {test_size}")
train_data, test_data = random_split(bearcart_dataset, [train_size, test_size])
train_dataloader = DataLoader(train_data, batch_size=125)
test_dataloader = DataLoader(test_data, batch_size=125)

# Create RGB and Depth models
rgb_model = efficientnet_b2(weights="IMAGENET1K_V1").to(DEVICE)
depth_model = efficientnet_b2(weights=None).to(DEVICE)

# Remove classifiers to extract feature maps
rgb_model.classifier = nn.Identity()
depth_model.classifier = nn.Identity()

# Define fusion model
class FeatureFusionModel(nn.Module):
    def __init__(self, rgb_model, depth_model):
        super().__init__()
        self.rgb_model = rgb_model
        self.depth_model = depth_model
        self.fusion = nn.Sequential(
            nn.Linear(1792 * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        # Split 4-channel input into RGB and Depth
        rgb_input = x[:, :3, :, :]
        depth_input = x[:, 3:, :, :]
        rgb_feat = self.rgb_model(rgb_input)
        depth_feat = self.depth_model(depth_input)
        fused = torch.cat((rgb_feat, depth_feat), dim=1)
        return self.fusion(fused)

model = FeatureFusionModel(rgb_model, depth_model).to(DEVICE)

# Hyper-parameters
lr = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_fn = standard_loss
epochs = 12
best_loss = float('inf')  # Best loss on test data
best_counter = 0
train_losses = []
test_losses = []

for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    ep_train_loss = train(train_dataloader, model, loss_fn, optimizer)
    ep_test_loss = test(test_dataloader, model, loss_fn)
    print(f"epoch {t + 1} training loss: {ep_train_loss}, testing loss: {ep_test_loss}")
    train_losses.append(ep_train_loss)
    test_losses.append(ep_test_loss)
    # Save the best model
    if ep_test_loss < best_loss:
        best_loss = ep_test_loss
        model_name = f'efficientnet_b2_rgbd-{t+1}ep-{lr}lr-{ep_test_loss:.4f}mse'
        torch.save(model.state_dict(), os.path.join(data_dir, f'{model_name}.pth'))
        print(f"Best model saved as '{model_name}.pth'")

print("Optimization Done!")

# Graph training process
plt.plot(range(epochs), train_losses, 'b--', label='Training')
plt.plot(range(epochs), test_losses, 'orange', label='Test')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.grid(True)
plt.legend()
plt.title('EfficientNet-B2 Training with RGB-D')
plt.savefig(os.path.join(data_dir, 'efficientnet_b2_rgbd_training.png'))

# Save the model (weights only)
torch.save(model.state_dict(), os.path.join(data_dir, 'efficientnet_b2_rgbd_final.pth'))
print("Model weights saved")

# ONNX export
dummy_input = torch.randn(1, 4, 240, 240).to(DEVICE)  # Adjust shape for 240x240 RGB-D
onnx_model_path = os.path.join(data_dir, 'efficientnet_b2_rgbd.onnx')
torch.onnx.export(model, dummy_input, onnx_model_path, opset_version=11)
print(f"Model exported to ONNX format at: {onnx_model_path}")