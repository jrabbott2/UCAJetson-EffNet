import torch
import torch.nn as nn
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights

class EfficientNetB2(nn.Module):
    def __init__(self, pretrained=True, fine_tune=True):
        super().__init__()
        
        # Load EfficientNet-B2 model with pre-trained weights
        if pretrained:
            self.base_model = efficientnet_b2(weights=EfficientNet_B2_Weights.IMAGENET1K_V1)
        else:
            self.base_model = efficientnet_b2(weights=None)  # Train from scratch
        
        # Freeze feature extractor layers if fine-tuning is disabled
        if pretrained and not fine_tune:
            for param in self.base_model.features.parameters():
                param.requires_grad = False

        # Adaptive average pooling before classifier for robustness
        self.base_model.avgpool = nn.AdaptiveAvgPool2d(1)

        # Modify classifier for steering & throttle output
        self.base_model.classifier = nn.Sequential(
            nn.Linear(self.base_model.classifier[1].in_features, 256),  # Increase feature size
            nn.ReLU(),
            nn.Dropout(0.2),  # Optimized dropout for 30 FPS
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)  # Outputs: steering and throttle
        )

    def forward(self, x):
        output = self.base_model(x)
        steering = torch.tanh(output[:, 0])  # Range [-1, 1]
        throttle = torch.sigmoid(output[:, 1])  # Range [0, 1]
        return torch.stack((steering, throttle), dim=1)

# Example usage
if __name__ == "__main__":
    from torchsummary import summary  # For debugging
    
    # Instantiate model with pretrained weights
    model = EfficientNetB2(pretrained=True, fine_tune=False)  # Using ImageNet weights, freezing features
    
    # Simulate an input tensor (batch size: 1, channels: 3, image size: 260x260)
    input_tensor = torch.randn(1, 3, 260, 260)
    
    # Forward pass
    output = model(input_tensor)
    print("Output:", output)
    
    # Print model summary
    summary(model, (3, 260, 260))