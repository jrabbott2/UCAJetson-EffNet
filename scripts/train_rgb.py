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
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True)  # Normalize [0, 1]
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

        # ✅ Ensure labels are a PyTorch tensor instead of a pandas Series
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
    nn.Linear(128, 2)  # Output: Steering & Throttle
).to(DEVICE)

# Mixed Precision Training for Jetson Performance
scaler = torch.amp.GradScaler(device="cuda")

# Loss & Optimizer
loss_fn = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# Training Function (ETA in Minutes & Seconds)
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

        # ✅ Ensure ETA is displayed in mm:ss format
        elapsed_time = time.time() - start_time
        percent_complete = (processed_samples / num_samples) * 100
        eta_seconds = ((elapsed_time / (b + 1)) * (len(dataloader) - (b + 1))) if b > 0 else 0
        eta_minutes, eta_seconds = divmod(int(eta_seconds), 60)
        eta_formatted = f"{eta_minutes:02d}:{eta_seconds:02d}" if b > 0 else "Calculating..."

        # Print structured progress
        print(f"{b+1:<8}{batch_loss.item():<15.6f}{processed_samples:<20}{percent_complete:<15.2f}{eta_formatted}")

    return ep_loss / len(dataloader)

# Training Loop
epochs = 12
best_loss = float('inf')
train_losses = []

for t in range(epochs):
    print(f"\n📢 Epoch {t + 1} -------------------------------")

    ep_train_loss = train(train_dataloader, model, loss_fn, optimizer, accumulation_steps=4)
    scheduler.step()
    
    train_losses.append(ep_train_loss)
    
    print(f"✅ Epoch {t + 1}: Train Loss = {ep_train_loss:.5f}")

    # ✅ Save the best model
    model_path = os.path.join(data_dir, f'efficientnet_b2-{t+1}ep-{ep_train_loss:.4f}mse.pth')
    torch.save(model.state_dict(), model_path)
    print(f"✅ Best model saved: {model_path}")

# ✅ Save Final Model
final_model_path = os.path.join(data_dir, 'efficientnet_b2_final.pth')
torch.save(model.state_dict(), final_model_path)
print(f"✅ Final model saved at: {final_model_path}")

# ✅ Generate MSE Loss vs Epoch Graph
plt.plot(range(1, epochs + 1), train_losses, 'b--', label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.title('EfficientNet-B2 Training Progress')
plt.grid(True)
graph_path = os.path.join(data_dir, 'efficientnet_b2_training.png')
plt.savefig(graph_path)
print(f"📊 Loss graph saved at: {graph_path}")

# ✅ Export ONNX Model
onnx_model_path = os.path.join(data_dir, 'efficientnet_b2.onnx')
dummy_input = torch.randn(1, 3, 260, 260).to(DEVICE)
torch.onnx.export(model, dummy_input, onnx_model_path, opset_version=11)
print(f"✅ Model exported to ONNX at: {onnx_model_path}")
