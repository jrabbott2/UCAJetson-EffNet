import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
import cv2
import matplotlib.pyplot as plt
import torch.onnx

# Suppress matplotlib warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# Custom model import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from convnets import EfficientNetB2

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32 if DEVICE.type == "cuda" else 8
NUM_WORKERS = min(4, os.cpu_count() - 1)
IMG_SIZE = (260, 260)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
EPOCHS = 12

class BearCartDataset(Dataset):
    def __init__(self, annotations_file, img_dir):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])

    def __len__(self): return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        try:
            image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, IMG_SIZE, interpolation=cv2.INTER_AREA)
            return self.transform(image), torch.tensor(
                self.img_labels.iloc[idx, 1:].values.astype(np.float32),
                dtype=torch.float32
            )
        except Exception as e:
            print(f"Error loading {img_path}: {str(e)}")
            return None, None

def collate_fn(batch):
    batch = [b for b in batch if b[0] is not None]
    return torch.utils.data.default_collate(batch)

def train_epoch(dataloader, model, optimizer, scaler):
    model.train()
    total_loss, start_time = 0.0, time.time()
    
    print(f"\n{'Batch':<8}{'Loss':<15}{'Processed':<20}{'ETA (mm:ss)'}")
    for b, (x, y) in enumerate(dataloader):
        x, y = x.to(DEVICE), y.to(DEVICE)
        
        with torch.amp.autocast(device_type='cuda'):  # Updated autocast
            loss = nn.MSELoss()(model(x), y)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        # Progress reporting
        elapsed = time.time() - start_time
        eta = (len(dataloader)-b-1) * (elapsed/(b+1)) if b > 0 else 0
        print(f"{b+1:<8}{loss.item():<15.6f}"
              f"{(b+1)*BATCH_SIZE}/{len(dataloader.dataset):<19}"
              f"{int(eta//60):02d}:{int(eta%60):02d}")

        total_loss += loss.item()
    return total_loss / len(dataloader)

def validate(dataloader, model):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            total_loss += nn.MSELoss()(model(x), y).item()
    return total_loss / len(dataloader)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python3 scripts/train_test.py <experiment_timestamp>")
    
    # Create data directory paths
    data_dir = os.path.join('data', sys.argv[1])
    annotations_file = os.path.join(data_dir, 'labels.csv')
    img_dir = os.path.join(data_dir, 'rgb_images')
    os.makedirs(data_dir, exist_ok=True)

    # Initialize dataset
    dataset = BearCartDataset(annotations_file, img_dir)
    
    # Dataset split
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_data, val_data, test_data = random_split(
        dataset, [train_size, val_size, test_size])

    # DataLoaders
    train_loader = DataLoader(
        train_data, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_data, batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_fn
    )

    # Model setup
    model = EfficientNetB2(pretrained=True).to(DEVICE)
    optimizer = torch.optim.AdamW([
        {'params': model.base_model.features.parameters(), 'lr': 1e-5},
        {'params': model.base_model.classifier.parameters(), 'lr': 1e-3}
    ], weight_decay=1e-4)
    
    # Updated scheduler without verbose parameter
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=2, factor=0.5
    )
    
    # Updated GradScaler initialization
    scaler = torch.amp.GradScaler(device_type='cuda')

    # Training loop
    best_loss = float('inf')
    train_losses = []
    val_losses = []
    previous_lr = None
    
    for epoch in range(1, EPOCHS+1):
        print(f"\nEpoch {epoch}/{EPOCHS}")
        
        # Training phase
        train_loss = train_epoch(train_loader, model, optimizer, scaler)
        train_losses.append(train_loss)
        
        # Validation phase
        val_loss = validate(val_loader, model)
        val_losses.append(val_loss)
        scheduler.step(val_loss)
        
        # Print learning rate changes
        current_lr = optimizer.param_groups[0]['lr']
        if previous_lr is not None and current_lr != previous_lr:
            print(f"Learning rate updated from {previous_lr:.2e} to {current_lr:.2e}")
        previous_lr = current_lr
        
        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_path = os.path.join(data_dir, 'best_model.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f"🏆 New best model saved: {best_model_path}")
        
        # Save checkpoints every 2 epochs
        if epoch % 2 == 0:
            checkpoint_path = os.path.join(data_dir, f'checkpoint_epoch{epoch}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"💾 Checkpoint saved: {checkpoint_path}")
            
            # Update training plot
            plt.figure(figsize=(10, 6))
            plt.plot(train_losses, label='Training Loss', marker='o')
            plt.plot(val_losses, label='Validation Loss', marker='x')
            plt.title('Training Progress')
            plt.xlabel('Epoch')
            plt.ylabel('MSE Loss')
            plt.legend()
            plt.grid(True)
            plot_path = os.path.join(data_dir, 'training_progress.png')
            plt.savefig(plot_path, dpi=300)
            plt.close()
            print(f"📈 Updated training plot saved")

    # Final model export
    onnx_path = os.path.join(data_dir, 'autopilot.onnx')
    dummy_input = torch.randn(1, 3, 260, 260).to(DEVICE)
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_path,
        opset_version=13,
        input_names=['input'],
        output_names=['steering', 'throttle']
    )
    print(f"✅ ONNX model exported to: {onnx_path}")