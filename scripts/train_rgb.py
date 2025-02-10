import os
import torchvision
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
import time  # For timing epochs

# Pass in command line arguments for data directory name
if len(sys.argv) != 2:
    print('Training script needs data!!!')
    sys.exit(1)  # Exit with an error code
else:
    data_datetime = sys.argv[1]

# Check if CUDA is available before setting memory limit
if torch.cuda.is_available():
        DEVICE = "cuda"
else:
    print("⚠️ CUDA is not available! Falling back to CPU.")
    DEVICE = "cpu"

class BearCartDataset(Dataset):
    """
    Customized dataset for RGB data only.
    """
    def __init__(self, annotations_file, img_dir):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = v2.Compose([
          v2.ToImage(), 
          v2.ToDtype(torch.float32, scale=True)  # Equivalent to `ToTensor()`
        ])

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        # Load RGB image from column 0 in labels.csv
        img_name = self.img_labels.iloc[idx, 0]  # Image name from the labels.csv
        img_path = os.path.join(self.img_dir, img_name)
        image = cv.imread(img_path, cv.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Error: Could not read RGB image at {img_path}")
        image = cv.resize(image, (260, 260), interpolation=cv.INTER_AREA)  # Ensure consistent resolution

        # Convert RGB image to tensor
        image_tensor = self.transform(image)

        # Steering and throttle values
        steering = self.img_labels.iloc[idx, 1].astype(np.float32)
        throttle = self.img_labels.iloc[idx, 2].astype(np.float32)

        return image_tensor.float(), steering, throttle

import torch

scaler = torch.cuda.amp.GradScaler()  # Initialize gradient scaler

def train(dataloader, model, loss_fn, optimizer, accumulation_steps=4):
    model.train()
    num_used_samples = 0
    ep_loss = 0.
    optimizer.zero_grad()  # Zero previous gradient at the start

    for b, (im, st, th) in enumerate(dataloader):
        target = torch.stack((st, th), dim=-1)
        feature, target = im.to("cuda", non_blocking=True), target.to("cuda", non_blocking=True)

        # ✅ Enable Mixed Precision Training with Gradient Scaling
        with torch.amp.autocast("cuda"):
            pred = model(feature)
            batch_loss = loss_fn(pred, target)

        # Scale and accumulate gradients
        scaler.scale(batch_loss).backward()

        # Perform optimizer step only every `accumulation_steps`
        if (b + 1) % accumulation_steps == 0 or (b + 1) == len(dataloader):
            scaler.step(optimizer)               # Step optimizer with scaled gradients
            scaler.update()                      # Update the scaler
            optimizer.zero_grad()                # Clear gradients after step

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

# Custom loss function (standard MSE)
def standard_loss(output, target):
    loss = ((output - target) ** 2).mean()
    return loss

# MAIN
# Create a dataset
data_dir = os.path.join(os.path.dirname(sys.path[0]), 'data', data_datetime)
annotations_file = os.path.join(data_dir, 'labels.csv')
img_dir = os.path.join(data_dir, 'rgb_images')
bearcart_dataset = BearCartDataset(annotations_file, img_dir)
print(f"Data length: {len(bearcart_dataset)}")

# Create training and test dataloaders
train_size = round(len(bearcart_dataset) * 0.9)
test_size = len(bearcart_dataset) - train_size
print(f"Train size: {train_size}, Test size: {test_size}")
train_data, test_data = random_split(bearcart_dataset, [train_size, test_size])
train_dataloader = DataLoader(train_data, batch_size=16, pin_memory=True, num_workers=4)
test_dataloader = DataLoader(test_data, batch_size=16, pin_memory=True, num_workers=4)

# Uncomment ONE of the following model setups at a time to train and compare weights

# Model with ImageNet pretrained weights
print("Training model with ImageNet weights...")
model = efficientnet_b2(weights=torchvision.models.EfficientNet_B2_Weights.IMAGENET1K_V1).to(DEVICE)

# Model with no pretrained weights (training from scratch)
#print("Training model from scratch (no pretrained weights)...")
#model = efficientnet_b2(weights=None).to(DEVICE)

# Print the classifier structure before modifying
print("Original Classifier Structure:")
print(model.classifier)

# Modify model classifier (Ensure correct index)
if isinstance(model.classifier, nn.Sequential):
    if isinstance(model.classifier[0], nn.Linear):  # If first layer is Linear
        classifier_input_features = model.classifier[0].in_features
    elif len(model.classifier) > 1 and isinstance(model.classifier[1], nn.Linear):  # If second layer is Linear
        classifier_input_features = model.classifier[1].in_features
    else:
        raise ValueError("Could not determine classifier input features.")

    model.classifier = nn.Sequential(
        nn.Linear(classifier_input_features, 128),
        nn.ReLU(),
        nn.Linear(128, 2)
    ).to(DEVICE)
else:
    raise ValueError("Unexpected classifier structure!")


# Hyperparameters
lr = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)  # Reduce LR by 10x every 3 epochs
loss_fn = standard_loss
epochs = 10
best_loss = float('inf')
train_losses = []
test_losses = []

accumulation_steps = 4  # Accumulate gradients over 4 batches

total_start_time = time.time()  # Start total timer

for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    epoch_start_time = time.time()  # Start epoch timer

    ep_train_loss = train(train_dataloader, model, loss_fn, optimizer, accumulation_steps=accumulation_steps)
    ep_test_loss = test(test_dataloader, model, loss_fn)

    scheduler.step()  # Step the learning rate scheduler

    epoch_end_time = time.time()  # End epoch timer
    epoch_time = epoch_end_time - epoch_start_time
    epoch_mins, epoch_secs = divmod(epoch_time, 60)
    print(f"Epoch {t + 1} Training loss: {ep_train_loss}, Testing loss: {ep_test_loss}")
    print(f"Epoch {t + 1} took {int(epoch_mins)}m {epoch_secs:.2f}s.")

    train_losses.append(ep_train_loss)
    test_losses.append(ep_test_loss)

    # Save best model
    previous_model_path = None
    if ep_test_loss < best_loss:
        # Delete previous best model if it exists
        if previous_model_path and os.path.exists(previous_model_path):
            os.remove(previous_model_path)
            print(f"Deleted previous best model: {previous_model_path}")
        best_loss = ep_test_loss
        model_name = f'efficientnet_b2-{t+1}ep-{lr}lr-{ep_test_loss:.4f}mse'
        model_path = os.path.join(data_dir, f'{model_name}.pth')
        torch.save(model.state_dict(), model_path)
        previous_model_path = model_path
        print(f"Best model saved as: {model_path}")

total_end_time = time.time()  # End total timer
total_time = total_end_time - total_start_time
total_mins, total_secs = divmod(total_time, 60)
print(f"Total training time: {int(total_mins)}m {total_secs:.2f}s.")

print("Optimization Done!")

# Graph training process
plt.plot(range(epochs), train_losses, 'b--', label='Training')
plt.plot(range(epochs), test_losses, 'orange', label='Test')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.grid(True)
plt.legend()
plt.title('EfficientNet-B2 Training')
graph_path = os.path.join(data_dir, 'efficientnet_b2_training.png')
plt.savefig(graph_path)
print(f"Training graph saved at: {graph_path}")

# Save final model
final_model_path = os.path.join(data_dir, 'efficientnet_b2_final.pth')
torch.save(model.state_dict(), final_model_path)
print(f"Final model weights saved at: {final_model_path}")

# ONNX export
dummy_input = torch.randn(1, 3, 260, 260).to(DEVICE)  # Corrected input size
onnx_model_path = os.path.join(data_dir, 'efficientnet_b2.onnx')
torch.onnx.export(model, dummy_input, onnx_model_path, opset_version=11)
print(f"Model exported to ONNX format at: {onnx_model_path}")
