import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
import cv2
from torch.cuda.amp import GradScaler, autocast
import matplotlib.pyplot as plt
import torch.onnx

# Custom model import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
        
        with autocast():
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
    # Initialization
    dataset = BearCartDataset(sys.argv[1]+'/labels.csv', sys.argv[1]+'/rgb_images')
    train_data, val_data, _ = random_split(dataset, [0.8, 0.1, 0.1])
    
    train_loader = DataLoader(train_data, BATCH_SIZE, True, 
                            num_workers=NUM_WORKERS, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, BATCH_SIZE, 
                          num_workers=NUM_WORKERS, collate_fn=collate_fn)

    # Model setup
    model = EfficientNetB2(pretrained=True).to(DEVICE)
    optimizer = torch.optim.AdamW([
        {'params': model.base_model.features.parameters(), 'lr': 1e-5},
        {'params': model.base_model.classifier.parameters(), 'lr': 1e-3}
    ])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=2, factor=0.5, verbose=True)
    scaler = GradScaler()

    # Training loop
    best_loss = float('inf')
    train_losses, val_losses = [], []
    
    for epoch in range(1, EPOCHS+1):
        print(f"\nEpoch {epoch}/{EPOCHS}")
        train_loss = train_epoch(train_loader, model, optimizer, scaler)
        val_loss = validate(val_loader, model)
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), f"{sys.argv[1]}/best_model.pth")
        
        # Save checkpoint and plot
        if epoch % 2 == 0:
            torch.save(model.state_dict(), f"{sys.argv[1]}/checkpoint_epoch{epoch}.pth")
            plt.figure()
            plt.plot(train_losses, label='Train')
            plt.plot(val_losses, label='Validation')
            plt.savefig(f"{sys.argv[1]}/training_plot.png")
            plt.close()

    # Final export
    torch.onnx.export(model, torch.randn(1,3,260,260).to(DEVICE), 
                     f"{sys.argv[1]}/model.onnx", opset_version=13)