import torch
import torch.nn as nn

class MultiModalNet(nn.Module):
    """
    A Convolutional Neural Network to process image, depth, and lidar data for steering and throttle prediction.
    """
    def __init__(self):
        super(MultiModalNet, self).__init__()
        
        # Image processing CNN layers
        self.conv_image_1 = nn.Conv2d(3, 24, kernel_size=5, stride=2)
        self.conv_image_2 = nn.Conv2d(24, 32, kernel_size=5, stride=2)
        self.conv_image_3 = nn.Conv2d(32, 64, kernel_size=5, stride=2)
        self.conv_image_4 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Depth data CNN layers (assuming depth image input size is similar to RGB)
        self.conv_depth_1 = nn.Conv2d(1, 16, kernel_size=5, stride=2)
        self.conv_depth_2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.conv_depth_3 = nn.Conv2d(32, 64, kernel_size=5, stride=2)

        # Lidar data processing layer (simple MLP)
        self.fc_lidar_1 = nn.Linear(360, 128)  # Assuming lidar scans are stored as 360-length vectors
        self.fc_lidar_2 = nn.Linear(128, 64)

        # Combined fully connected layers
        self.fc1 = nn.Linear(64*8*13 + 64*8*13 + 64, 256)  # Combine image, depth, and lidar features
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 2)  # Output steering and throttle

        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, image, depth, lidar):
        # Image processing
        x_img = self.relu(self.conv_image_1(image))
        x_img = self.relu(self.conv_image_2(x_img))
        x_img = self.relu(self.conv_image_3(x_img))
        x_img = self.relu(self.conv_image_4(x_img))
        x_img = self.flatten(x_img)

        # Depth processing
        x_depth = self.relu(self.conv_depth_1(depth))
        x_depth = self.relu(self.conv_depth_2(x_depth))
        x_depth = self.relu(self.conv_depth_3(x_depth))
        x_depth = self.flatten(x_depth)

        # Lidar processing
        x_lidar = self.relu(self.fc_lidar_1(lidar))
        x_lidar = self.relu(self.fc_lidar_2(x_lidar))

        # Concatenate image, depth, and lidar features
        x_combined = torch.cat((x_img, x_depth, x_lidar), dim=1)

        # Fully connected layers
        x = self.relu(self.fc1(x_combined))
        x = self.relu(self.fc2(x))
        output = self.fc3(x)
        
        return output
