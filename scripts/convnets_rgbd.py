import torch
import torch.nn as nn
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights

class EfficientNetB2_RGB_Depth(nn.Module):
    def __init__(self, pretrained=True, fine_tune=True):
        super().__init__()
        
        # RGB Model
        self.rgb_model = efficientnet_b2(weights=EfficientNet_B2_Weights.IMAGENET1K_V1 if pretrained else None)
        # Depth Model
        self.depth_model = efficientnet_b2(weights=None)  # Depth model without pretrained weights

        # Modify depth model to accept single-channel input
        self.depth_model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)

        # Freeze feature extractors if fine-tuning is disabled
        if pretrained and not fine_tune:
            for param in self.rgb_model.features.parameters():
                param.requires_grad = False
            for param in self.depth_model.features.parameters():
                param.requires_grad = False

        # Adaptive pooling for consistency
        self.rgb_model.avgpool = nn.AdaptiveAvgPool2d(1)
        self.depth_model.avgpool = nn.AdaptiveAvgPool2d(1)

        # Final classifier after feature concatenation
        combined_features = self.rgb_model.classifier[1].in_features + self.depth_model.classifier[1].in_features
        self.classifier = nn.Sequential(
            nn.Linear(combined_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)  # Steering and throttle outputs
        )

    def forward(self, rgb, depth):
        rgb_features = self.rgb_model.extract_features(rgb).flatten(1)
        depth_features = self.depth_model.extract_features(depth).flatten(1)

        # Concatenate features
        combined = torch.cat((rgb_features, depth_features), dim=1)
        output = self.classifier(combined)

        # Activations for outputs
        steering = torch.tanh(output[:, 0])  # Range [-1, 1]
        throttle = torch.sigmoid(output[:, 1])  # Range [0, 1]

        return torch.stack((steering, throttle), dim=1)

# Example usage
if __name__ == "__main__":
    from torchsummary import summary

    model = EfficientNetB2_RGB_Depth(pretrained=True, fine_tune=False)

    # Simulate RGB and Depth inputs
    rgb_input = torch.randn(1, 3, 260, 260)
    depth_input = torch.randn(1, 1, 260, 260)

    # Forward pass
    output = model(rgb_input, depth_input)
    print("Output:", output)

    # Summary for RGB and Depth models
    print("\nRGB Model Summary:")
    summary(model.rgb_model, (3, 260, 260))
    print("\nDepth Model Summary:")
    summary(model.depth_model, (1, 260, 260))
