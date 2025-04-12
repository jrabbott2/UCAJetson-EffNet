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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights
import warnings
import multiprocessing

warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")

if len(sys.argv) != 2:
    print('❌ Error: Training script needs data folder name as argument!')
    sys.exit(1)
else:
    data_datetime = sys.argv[1]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cpu":
    print("⚠️ CUDA is not available! Training will be significantly slower.")
else:
    torch.cuda.set_per_process_memory_fraction(0.7, 0)
    torch.backends.cudnn.benchmark = True

class BearCartDataset(Dataset):
    def __init__(self, annotations_file, rgb_dir, depth_dir):
        self.img_labels = pd.read_csv(annotations_file)
        self.rgb_dir = rgb_dir
        self.depth_dir = depth_dir
        self.transform = v2.Compose([
            v2.ToImageTensor(),
            v2.ConvertImageDtype(torch.float32)
        ])

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        rgb_name = self.img_labels.iloc[idx, 0].strip()
        depth_name = self.img_labels.iloc[idx, 1].strip()
        rgb_path = os.path.join(self.rgb_dir, rgb_name)
        depth_path = os.path.join(self.depth_dir, depth_name)

        rgb_image = cv.imread(rgb_path, cv.IMREAD_COLOR)
        depth_image = np.load(depth_path)  # depth is loaded as .npy

        rgb_image = cv.resize(rgb_image, (260, 260), interpolation=cv.INTER_AREA)
        depth_image = cv.resize(depth_image, (260, 260), interpolation=cv.INTER_AREA)

        rgb_tensor = self.transform(rgb_image)         # [3, 260, 260]
        depth_tensor = torch.tensor(depth_image).unsqueeze(0).float()  # [1, 260, 260]
        input_tensor = torch.cat((rgb_tensor, depth_tensor), dim=0)    # [4, 260, 260]

        labels = self.img_labels.iloc[idx, 2:].values.astype(np.float32)
        return input_tensor, torch.tensor(labels)

# Setup paths
data_dir = os.path.join(os.path.dirname(sys.path[0]), 'data', data_datetime)
annotations_file = os.path.join(data_dir, 'labels.csv')
rgb_dir = os.path.join(data_dir, 'rgb_images')
depth_dir = os.path.join(data_dir, 'depth_images')

# Load dataset
bearcart_dataset = BearCartDataset(annotations_file, rgb_dir, depth_dir)
train_size = round(len(bearcart_dataset) * 0.9)
test_size = len(bearcart_dataset) - train_size
train_data, test_data = random_split(bearcart_dataset, [train_size, test_size])

num_workers = min(4, multiprocessing.cpu_count() - 1)
train_dataloader = DataLoader(train_data, batch_size=16, pin_memory=True, num_workers=num_workers, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=16, pin_memory=True, num_workers=num_workers)

print(f"✅ Dataset loaded. Training size: {train_size}, Testing size: {test_size}")

print("🔄 Loading EfficientNet-B2 model with pretrained weights...")
model = efficientnet_b2(weights=EfficientNet_B2_Weights.IMAGENET1K_V1)
model.features[0][0] = nn.Conv2d(4, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)  # for RGB + Depth
model.classifier = nn.Sequential(
    nn.Linear(model.classifier[1].in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, 2)
)
model = model.to(DEVICE)

scaler = torch.cuda.amp.GradScaler()

# Loss & Optimizer
loss_fn = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# Training Function
def train(dataloader, model, loss_fn, optimizer, accumulation_steps=4):
    model.train()
    ep_loss = 0.
    optimizer.zero_grad()

    log_file = os.path.join(data_dir, "training_log.txt")
    with open(log_file, "a") as log:
        log.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] 📊 Training Progress Started\n")

    print("\n📊 Training Progress:")
    print(f"{'Batch':<8}{'Loss':<12}{'Processed':<18}{'Completion %':<12}{'ETA (mm:ss)'}")
    print("-" * 75)
    start_time = time.time()
    num_samples = len(dataloader.dataset)
    processed_samples = 0

    for b, (im, labels) in enumerate(dataloader):
        batch_size = im.size(0)
        feature, target = im.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)

        pred = model(feature)
        batch_loss = loss_fn(pred, target)

        scaler.scale(batch_loss).backward()

        if (b + 1) % accumulation_steps == 0 or (b + 1) == len(dataloader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        ep_loss += batch_loss.item()
        processed_samples += batch_size
        elapsed_time = time.time() - start_time
        completion_pct = (processed_samples / num_samples) * 100
        eta = elapsed_time / completion_pct * (100 - completion_pct) if completion_pct > 0 else 0

        print(f"{b:<8}{batch_loss.item():<12.5f}{processed_samples}/{num_samples:<15}{completion_pct:.2f}%{' ' * 3}{time.strftime('%M:%S', time.gmtime(eta))}")

        with open(log_file, "a") as log:
            log.write(f"Batch {b}: Loss {batch_loss.item():.5f}, Processed {processed_samples}/{num_samples}, Completion: {completion_pct:.2f}%\n")

    return ep_loss / len(dataloader)

def test(dataloader, model, loss_fn):
    model.eval()  # Set to evaluation mode
    ep_loss = 0.

    with torch.no_grad():  # Disable gradient updates for efficiency
        for im, labels in dataloader:
            feature, target = im.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
            pred = model(feature)
            batch_loss = loss_fn(pred, target)
            ep_loss += batch_loss.item()

    return ep_loss / len(dataloader)  # Return average test loss

# Training Loop
epochs = 12
train_losses, test_losses = [], []
best_loss = float('inf')  # Initialize best loss

for t in range(epochs):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 📢 Epoch {t + 1} -------------------------------")
    ep_train_loss = train(train_dataloader, model, loss_fn, optimizer, accumulation_steps=4)
    ep_test_loss = test(test_dataloader, model, loss_fn)
    scheduler.step()
    
    train_losses.append(ep_train_loss)
    test_losses.append(ep_test_loss)

    # ✅ Display MSE Loss after each epoch
    print(f"✅ Epoch {t + 1}: Train Loss = {ep_train_loss:.5f}, Test Loss = {ep_test_loss:.5f}")

    # ✅ Log epoch losses
    log_file = os.path.join(data_dir, "training_log.txt")
    with open(log_file, "a") as log:
        log.write(f"Epoch {t + 1}: Train Loss = {ep_train_loss:.5f}, Test Loss = {ep_test_loss:.5f}\n")


 # ✅ Save Final Model (Outside the Loop)
final_model_path = os.path.join(data_dir, 'efficientnet_b2_final.pth')
torch.save(model.state_dict(), final_model_path)
print(f"✅ Final model saved at: {final_model_path}")


# Graph training process
plt.plot(range(1, epochs + 1), train_losses, 'b--o', label='Training')
plt.plot(range(1, epochs + 1), test_losses, 'orange', marker='s', label='Test')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.title('EfficientNet-B2 Training Progress')
plt.grid(True)
plt.tight_layout()
graph_path = os.path.join(data_dir, 'efficientnet_b2_training.png')
plt.savefig(graph_path)
print(f"📊 Loss graph saved at: {graph_path}")

# ✅ Export ONNX Model
onnx_model_path = os.path.join(data_dir, 'efficientnet_b2.onnx')
dummy_input = torch.randn(1, 4, 260, 260).to(DEVICE)
try:
    torch.onnx.export(model, dummy_input, onnx_model_path, opset_version=11)
    print(f"✅ Model exported to ONNX at: {onnx_model_path}")
except Exception as e:
    print(f"⚠️ ONNX export failed: {e}")