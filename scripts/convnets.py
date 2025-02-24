import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights

class EfficientNetB2(nn.Module):
    def __init__(self, 
                 pretrained=True, 
                 fine_tune=False, 
                 max_throttle=1.0,
                 input_size=(260, 260)):
        super().__init__()
        
        # Configuration
        self.max_throttle = max_throttle
        self.input_size = input_size
        
        # Load pretrained EfficientNet-B2
        self.base_model = efficientnet_b2(weights=EfficientNet_B2_Weights.IMAGENET1K_V1 if pretrained else None)
        
        # Freeze backbone if needed
        if pretrained and not fine_tune:
            for param in self.base_model.features.parameters():
                param.requires_grad = False

        # Custom classifier head
        in_features = self.base_model.classifier[1].in_features
        self.base_model.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 2)  # Steering + throttle
        )

    def forward(self, x):
        x = self.base_model.features(x)
        x = self.base_model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.base_model.classifier(x)
        
        steering = torch.tanh(x[:, 0])  # [-1, 1]
        throttle = torch.sigmoid(x[:, 1]) * self.max_throttle  # [0, max_throttle]
        
        return torch.stack((steering, throttle), dim=1)

# Training Utilities
def get_model_summary(model):
    from torchsummary import summary
    return summary(model, (3, 260, 260), device='cpu')

def create_optimizer(model, lr=1e-4):
    return torch.optim.Adam(model.parameters(), lr=lr)

# Example Usage
if __name__ == "__main__":
    model = EfficientNetB2(pretrained=True, fine_tune=False, max_throttle=1.0)
    print(get_model_summary(model))
    optimizer = create_optimizer(model)
    criterion = nn.MSELoss()
