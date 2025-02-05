import torch
import torch.nn as nn
from torchvision.models import efficientnet_b2

class EfficientNetB2(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        
        # Load EfficientNet-B2 model
        if pretrained:
            self.base_model = efficientnet_b2(weights="IMAGENET1K_V1")  # Load pre-trained ImageNet weights
        else:
            self.base_model = efficientnet_b2(weights=None)  # Train from scratch
        
        # Modify classifier for steering & throttle output
        self.base_model.classifier = nn.Sequential(
            nn.Linear(self.base_model.classifier.in_features, 128),  # Bottleneck layer
            nn.ReLU(),
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
    
    # Instantiate model WITHOUT pre-trained weights
    model = EfficientNetB2(pretrained=False)  # Train from scratch
    
    # Uncomment the line below to try with pre-trained weights
    # model = EfficientNetB2(pretrained=True)  # Use ImageNet weights
    
    # Simulate an input tensor (batch size: 1, channels: 3, image size: 260x260)
    input_tensor = torch.randn(1, 3, 260, 260)
    
    # Forward pass
    output = model(input_tensor)
    print("Output:", output)
    
    # Print model summary
    summary(model, (3, 260, 260))
