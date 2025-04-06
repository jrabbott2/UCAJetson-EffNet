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
matplotlib.use('Agg')  # Set non-GUI backend for matplotlib
import matplotlib.pyplot as plt
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights
import warnings
import multiprocessing

# Suppress torchvision beta warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")

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
else:
    torch.cuda.set_per_process_memory_fraction(0.7, 0)  # Use 70% of GPU memory
    torch.backends.cudnn.benchmark = True  # Optimize GPU performance

class BearCartDataset(Dataset):
    """Dataset loader for RGB and Depth images along with steering/throttle labels."""
    def __init__(self, annotations_file, rgb_dir, depth_dir):
        self.img_labels = pd.read_csv(annotations_file)
        self.rgb_dir = rgb_dir
        self.depth_dir = depth_dir
        self.transform_rgb = v2.Compose([
            v2.ToImageTensor(),
            v2.ConvertImageDtype(torch.float32)  # Normalize [0, 1]
        ])
        self.transform_depth = v2.Compose([
            v2.ToImageTensor(),
            v2.ConvertImageDtype(torch.float32)  # Normalize [0, 1]
        ])
    
    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        img_name = self.img_labels.iloc[idx, 0]
        img_path = os.path.join(self.img_dir, img_name)

        rgb_image = cv.imread(img_path, cv.IMREAD_COLOR)
        depth_path = os.path.join(self.depth_dir, self.img_labels.iloc[idx, 1].replace('.png', '.npy'))
        depth_image = np.load(depth_path)

        if rgb_image is None or depth_image is None:
            raise FileNotFoundError(f"❌ Error: Could not read image {img_path} or {depth_path}")

        depth_image = cv.resize(depth_image, (260, 260), interpolation=cv.INTER_AREA)
        depth_image = np.expand_dims(depth_image, axis=2)  # Add channel dimension
        if image is None:
            raise FileNotFoundError(f"❌ Error: Could not read image {img_path}")

        image = cv.resize(image, (260, 260), interpolation=cv.INTER_AREA)

        labels = self.img_labels.iloc[idx, 1:].values.astype(np.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.float32)

        return (self.transform_rgb(rgb_image), self.transform_depth(depth_image)), labels_tensor

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
num_workers = min(4, multiprocessing.cpu_count() - 1)
train_dataloader = DataLoader(train_data, batch_size=16, pin_memory=True, num_workers=num_workers, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=16, pin_memory=True, num_workers=num_workers)

print(f"✅ Dataset loaded. Training size: {train_size}, Testing size: {test_size}")

# Load EfficientNet-B2 with pre-trained weights
print("🔄 Loading EfficientNet-B2 model with pretrained weights...")
rgb_model = efficientnet_b2(weights=EfficientNet_B2_Weights.IMAGENET1K_V1).to(DEVICE)
depth_model = efficientnet_b2(weights=None).to(DEVICE)

# Modify classifier for steering & throttle
rgb_features = rgb_model.classifier[1].in_features
depth_features = depth_model.classifier[1].in_features
rgb_model.classifier = nn.Identity()
depth_model.classifier = nn.Identity()

# Feature fusion module
class FeatureFusionModel(nn.Module):
    def __init__(self, rgb_model, depth_model):
        super().__init__()
        self.rgb_model = rgb_model
        self.depth_model = depth_model
        self.fusion_layer = nn.Sequential(
            nn.Linear(rgb_features + depth_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)
        )
    
    def forward(self, rgb_input, depth_input):
        rgb_features = self.rgb_model(rgb_input)
        depth_features = self.depth_model(depth_input)
        fused_features = torch.cat((rgb_features, depth_features), dim=1)
        return self.fusion_layer(fused_features)

# Initialize fusion model
model = FeatureFusionModel(rgb_model, depth_model).to(DEVICE)

# Mixed Precision Training
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

    # ✅ Save Best Model Inside Training Loop
    best_model_path = os.path.join(data_dir, 'best_model.pth')
    if ep_test_loss < best_loss:  # Use test loss for best model tracking
        best_loss = ep_test_loss
        try:
            if os.path.exists(best_model_path):
                os.remove(best_model_path)
                print(f"🗑️ Previous best model deleted: {best_model_path}")
        except Exception as e:
            print(f"⚠️ Error deleting previous best model: {e}")

        torch.save(model.state_dict(), best_model_path)
        print(f"✅ Best model saved: {best_model_path}")

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
dummy_input = torch.randn(1, 3, 260, 260).to(DEVICE)
try:
    torch.onnx.export(model, dummy_input, onnx_model_path, opset_version=11)
    print(f"✅ Model exported to ONNX at: {onnx_model_path}")
except Exception as e:
    print(f"⚠️ ONNX export failed: {e}")

