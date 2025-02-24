import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import v2
import cv2 as cv
import torch.onnx
import matplotlib.pyplot as plt
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights

# Ensure correct CLI input
if len(sys.argv) != 2:
    print('❌ Error: Training script needs data folder name as argument!')
    sys.exit(1)
else:
    data_datetime = sys.argv[1]

# Select device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cpu":
    print("⚠️ CUDA is not available! Training will be significantly slower.")

class BearCartDataset(Dataset):
    """Dataset loader for RGB images and steering/throttle labels."""
    def __init__(self, annotations_file, img_dir):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = v2.Compose([
            v2.ToTensor(),
            v2.ToDtype(torch.float32)  # Normalize [0, 1]
        ])

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_name = self.img_labels.iloc[idx, 0]
        img_path = os.path.join(self.img_dir, img_name)

        image = cv.imread(img_path, cv.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"❌ Error: Could not read image {img_path}")

        image = cv.resize(image, (260, 260), interpolation=cv.INTER_AREA)

        labels = self.img_labels.iloc[idx, 1:].values.astype(np.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.float32)

        return self.transform(image), labels_tensor

# Create dataset paths
data_dir = os.path.join(os.path.dirname(sys.path[0]), 'data', data_datetime)
annotations_file = os.path.join(data_dir, 'labels.csv')
img_dir = os.path.join(data_dir, 'rgb_images')

# Load dataset
bearcart_dataset = BearCartDataset(annotations_file, img_dir)
train_size = round(len(bearcart_dataset) * 0.9)
test_size = len(bearcart_dataset) - train_size
train_data, test_data = random_split(bearcart_dataset, [train_size, test_size])

# Optimized DataLoaders
train_dataloader = DataLoader(train_data, batch_size=16, pin_memory=True, num_workers=4, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=16, pin_memory=True, num_workers=4)

print(f"✅ Dataset loaded. Training size: {train_size}, Testing size: {test_size}")

# Load EfficientNet-B2 with pretrained weights
print("🔄 Loading EfficientNet-B2 model with pretrained weights...")
model = efficientnet_b2(weights=EfficientNet_B2_Weights.IMAGENET1K_V1).to(DEVICE)

# Modify classifier for steering & throttle
classifier_input_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Linear(classifier_input_features, 256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, 2)
).to(DEVICE)

# Mixed Precision Training
scaler = torch.cuda.amp.GradScaler()

# Loss & Optimizer
loss_fn = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# Training Function (ETA in mm:ss format)
def train(dataloader, model, loss_fn, optimizer, accumulation_steps=4):
    model.train()
    ep_loss = 0.
    optimizer.zero_grad()
    
    num_samples = len(dataloader.dataset)
    processed_samples = 0
    start_time = time.time()

    print("\n📊 Training Progress:")
    print(f"{'Batch':<8}{'Loss':<15}{'Processed':<20}{'Completion %':<15}{'ETA (mm:ss)'}")
    print("-" * 80)

    for b, (im, labels) in enumerate(dataloader):
        batch_size = im.size(0)
        feature, target = im.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)

        with torch.amp.autocast("cuda"):
            pred = model(feature)
            batch_loss = loss_fn(pred, target)

        scaler.scale(batch_loss).backward()

        if (b + 1) % accumulation_steps == 0 or (b + 1) == len(dataloader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        ep_loss += batch_loss.item()
        processed_samples += batch_size
    
    return ep_loss / len(dataloader)

# Training Loop
epochs = 12
best_loss = float('inf')
train_losses, test_losses = [], []

for t in range(epochs):
    print(f"\n📢 Epoch {t + 1} -------------------------------")

    ep_train_loss = train(train_dataloader, model, loss_fn, optimizer, accumulation_steps=4)
    ep_test_loss = train(test_dataloader, model, loss_fn, optimizer, accumulation_steps=4)

    scheduler.step()
    
    train_losses.append(ep_train_loss)
    test_losses.append(ep_test_loss)
    
    print(f"✅ Epoch {t + 1}: Train Loss = {ep_train_loss:.5f}, Test Loss = {ep_test_loss:.5f}")

    model_path = os.path.join(data_dir, f'efficientnet_b2-{t+1}ep-{ep_test_loss:.4f}mse.pth')
    torch.save(model.state_dict(), model_path)
    print(f"✅ Best model saved: {model_path}")

# ✅ Export ONNX Model
torch.onnx.export(model, torch.randn(1, 3, 260, 260).to(DEVICE), os.path.join(data_dir, 'efficientnet_b2.onnx'))
