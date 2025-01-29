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
    Customized dataset for RGB and IMU data.
    """
    def __init__(self, annotations_file, rgb_dir):
        self.img_labels = pd.read_csv(annotations_file)
        self.rgb_dir = rgb_dir
        self.transform = v2.ToTensor()

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        # Load RGB image from column 0 in labels.csv
        rgb_name = self.img_labels.iloc[idx, 0]  # RGB image name from labels.csv
        rgb_path = os.path.join(self.rgb_dir, rgb_name)
        rgb_image = cv.imread(rgb_path, cv.IMREAD_COLOR)
        if rgb_image is None:
            raise FileNotFoundError(f"Error: Could not read RGB image at {rgb_path}")
        rgb_image = cv.resize(rgb_image, (240, 240))  # Ensure consistent resolution

        # Convert RGB image to tensor
        rgb_tensor = self.transform(rgb_image)

        # IMU data (columns 1-6 in labels.csv)
        imu_data = self.img_labels.iloc[idx, 1:7].to_numpy(dtype=np.float32)
        imu_tensor = torch.tensor(imu_data, dtype=torch.float32)

        # Steering and throttle values
        steering = self.img_labels.iloc[idx, 7].astype(np.float32)
        throttle = self.img_labels.iloc[idx, 8].astype(np.float32)

        return rgb_tensor.float(), imu_tensor, steering, throttle

def train(dataloader, model, loss_fn, optimizer):
    model.train()
    num_used_samples = 0
    ep_loss = 0.
    for b, (rgb, imu, st, th) in enumerate(dataloader):
        target = torch.stack((st, th), dim=-1)
        rgb, imu, target = rgb.to(DEVICE), imu.to(DEVICE), target.to(DEVICE)
        pred = model(rgb, imu)
        batch_loss = loss_fn(pred, target)  # Loss function
        optimizer.zero_grad()  # Zero previous gradient
        batch_loss.backward()  # Back propagation
        optimizer.step()  # Update params
        num_used_samples += target.shape[0]
        print(f"batch loss: {batch_loss.item()} [{num_used_samples}/{len(dataloader.dataset)}]")
        ep_loss = (ep_loss * b + batch_loss.item()) / (b + 1)
    return ep_loss

def test(dataloader, model, loss_fn):
    model.eval()
    ep_loss = 0.
    with torch.no_grad():
        for b, (rgb, imu, st, th) in enumerate(dataloader):
            target = torch.stack((st, th), dim=-1)
            rgb, imu, target = rgb.to(DEVICE), imu.to(DEVICE), target.to(DEVICE)
            pred = model(rgb, imu)
            batch_loss = loss_fn(pred, target)  # Loss function
            ep_loss = (ep_loss * b + batch_loss.item()) / (b + 1)
    return ep_loss

# Custom loss function (standard MSE)
def standard_loss(output, target):
    loss = ((output - target) ** 2).mean()  # Mean Squared Error loss
    return loss

class EfficientNetB2RGBIMU(nn.Module):
    """
    EfficientNet-B2 modified to handle RGB and IMU data.
    """
    def __init__(self):
        super(EfficientNetB2RGBIMU, self).__init__()
        self.rgb_model = efficientnet_b2(weights=None)
        self.rgb_model.classifier = nn.Identity()  # Remove default classifier
        self.imu_fc = nn.Sequential(
            nn.Linear(6, 64),  # 6 IMU features
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(self.rgb_model.classifier[1].in_features + 32, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # Output steering and throttle
        )

    def forward(self, rgb, imu):
        rgb_features = self.rgb_model(rgb)
        imu_features = self.imu_fc(imu)
        combined = torch.cat((rgb_features, imu_features), dim=1)
        return self.fc(combined)

# MAIN
# Create a dataset
data_dir = os.path.join(os.path.dirname(sys.path[0]), 'data', data_datetime)
annotations_file = os.path.join(data_dir, 'labels.csv')
rgb_dir = os.path.join(data_dir, 'rgb_images')  # RGB images directory
bearcart_dataset = BearCartDataset(annotations_file, rgb_dir)
print(f"data length: {len(bearcart_dataset)}")

# Create training and test dataloaders
train_size = round(len(bearcart_dataset) * 0.9)
test_size = len(bearcart_dataset) - train_size
print(f"train size: {train_size}, test size: {test_size}")
train_data, test_data = random_split(bearcart_dataset, [train_size, test_size])
train_dataloader = DataLoader(train_data, batch_size=125)
test_dataloader = DataLoader(test_data, batch_size=125)

# Create model
model = EfficientNetB2RGBIMU().to(DEVICE)

# Hyper-parameters
lr = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_fn = standard_loss
epochs = 20
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
        model_name = f'efficientnet_b2_rgb_imu-{t+1}ep-{lr}lr-{ep_test_loss:.4f}mse'
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
plt.title('EfficientNet-B2 Training with RGB and IMU')
plt.savefig(os.path.join(data_dir, 'efficientnet_b2_rgb_imu_training.png'))

# Save the model (weights only)
torch.save(model.state_dict(), os.path.join(data_dir, 'efficientnet_b2_rgb_imu_final.pth'))
print("Model weights saved")

# ONNX export
dummy_input_rgb = torch.randn(1, 3, 240, 240).to(DEVICE)  # Adjust shape for 240x240 RGB
dummy_input_imu = torch.randn(1, 6).to(DEVICE)  # IMU data with 6 features
onnx_model_path = os.path.join(data_dir, 'efficientnet_b2_rgb_imu.onnx')
torch.onnx.export(model, (dummy_input_rgb, dummy_input_imu), onnx_model_path, opset_version=11)
print(f"Model exported to ONNX format at: {onnx_model_path}")
