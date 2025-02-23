import torch
import torch.nn as nn
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights

class EfficientNetB2(nn.Module):
    def __init__(self, pretrained=True, fine_tune=True, max_throttle=1.0):
        super().__init__()
        self.max_throttle = max_throttle  # For scaling throttle output
        
        # 1. Input normalization layer (critical for pretrained models)
        self.normalize = nn.Sequential(
            nn.BatchNorm2d(3),  # Optional: Helps with input stability
            nn.LayerNorm([260, 260]),  # Optional: Adjust based on your data
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet stats
                                std=[0.229, 0.224, 0.225]),
        )

        # 2. Load base model with pretrained weights
        if pretrained:
            self.base_model = efficientnet_b2(weights=EfficientNet_B2_Weights.IMAGENET1K_V1)
        else:
            self.base_model = efficientnet_b2(weights=None)

        # 3. Freeze feature extractor if not fine-tuning
        if pretrained and not fine_tune:
            for param in self.base_model.features.parameters():
                param.requires_grad = False

        # 4. Adaptive pooling (already good in your original code)
        self.base_model.avgpool = nn.AdaptiveAvgPool2d(1)

        # 5. Enhanced classifier with batch normalization
        in_features = self.base_model.classifier[1].in_features
        self.base_model.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),  # Stabilizes training
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),  # Second batch norm layer
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)  # Keep 2 outputs (steering/throttle)
        )

    def forward(self, x):
        # Verify input dimensions first
        if x.shape[-2:] != (260, 260):
            raise ValueError(f"Input must be 260x260. Got {x.shape[-2:]}")
        
        # Apply normalization
        x = self.normalize(x)
        
        # Base model forward pass
        output = self.base_model(x)
        
        # Final activations with scaling
        steering = torch.tanh(output[:, 0])  # [-1, 1]
        throttle = torch.sigmoid(output[:, 1]) * self.max_throttle  # [0, max_throttle]
        
        return torch.stack((steering, throttle), dim=1)

# Example usage with training setup
if __name__ == "__main__":
    from torchsummary import summary
    import torch.optim as optim
    from torch.cuda.amp import GradScaler  # For mixed precision
    
    # Initialize model
    model = EfficientNetB2(pretrained=True, fine_tune=False, max_throttle=1.0)
    
    # Print model summary
    summary(model, (3, 260, 260))
    
    # Example training setup
    criterion = nn.MSELoss()
    
    # Differential learning rates (lower for pretrained features)
    optimizer = optim.Adam([
        {'params': model.base_model.features.parameters(), 'lr': 1e-5},
        {'params': model.base_model.classifier.parameters(), 'lr': 1e-3}
    ])
    
    # Mixed precision scaler (boosts speed on modern GPUs)
    scaler = GradScaler()
    
    # Dummy training loop example
    for epoch in range(5):
        dummy_input = torch.randn(8, 3, 260, 260)  # Batch of 8
        dummy_targets = torch.randn(8, 2)  # Random steering/throttle
        
        with torch.cuda.amp.autocast():  # FP16 training
            outputs = model(dummy_input)
            loss = criterion(outputs, dummy_targets)
        
        # Backprop with scaler
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")