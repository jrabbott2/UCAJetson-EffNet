import torch
import torch.nn as nn
from torchvision.models import efficientnet_b2

class EfficientNetB2(nn.Module):
    def __init__(self):
        super().__init__()
        # Load pre-trained EfficientNet-B2 model
        self.base_model = efficientnet_b2(pretrained=False)
        
        # Modify the classifier to output 2 values for steering and throttle
        self.base_model.classifier = nn.Sequential(
            nn.Linear(self.base_model.classifier[1].in_features, 128),  # Bottleneck layer
            nn.ReLU(),
            nn.Linear(128, 2)  # Outputs: steering and throttle
        )

    def forward(self, x):
        return self.base_model(x)

# Example usage
if __name__ == "__main__":
    # Initialize the model
    model = EfficientNetB2()

    # Simulate an input tensor (batch size: 1, channels: 3, image size: 260x260)
    input_tensor = torch.randn(1, 3, 260, 260)

    # Forward pass
    output = model(input_tensor)
    print("Output:", output)
