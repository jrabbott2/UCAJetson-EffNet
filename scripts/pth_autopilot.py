import os
import sys
import json
import torch
import pygame
import cv2 as cv
from torchvision import transforms
from time import time  # For frame rate calculation
from hardware_rgb import (
    get_realsense_frame, setup_realsense_camera, setup_serial, 
    setup_joystick, encode_dutycylce, encode
)
from torchvision.models import efficientnet_b2  # Import EfficientNet-B2

# Adds dummy to run Pygame without a display
os.environ["SDL_VIDEODRIVER"] = "dummy"

# Initialize only the required Pygame modules
pygame.display.init()
pygame.joystick.init()
js = pygame.joystick.Joystick(0)

# SETUP
# Load EfficientNet-B2 model
model_path = os.path.join('models', 'efficientnet_b2_final.pth')  # Adjust if needed
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"🔄 Loading EfficientNet-B2 model from {model_path}...")

model = efficientnet_b2(weights=None).to(device)  # Load without pretrained weights
classifier_input_features = model.classifier[1].in_features

# Modify classifier for steering & throttle outputs
model.classifier = torch.nn.Sequential(
    torch.nn.Linear(classifier_input_features, 256),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.2),
    torch.nn.Linear(256, 128),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.2),
    torch.nn.Linear(128, 2)  # Output: Steering & Throttle
).to(device)

# Load trained weights
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()  # Ensure the model is in evaluation mode

# Load configs
params_file_path = os.path.join(sys.path[0], 'config.json')
with open(params_file_path) as params_file:
    params = json.load(params_file)

# Constants
STEERING_AXIS = params['steering_joy_axis']
STEERING_CENTER = params['steering_center']
THROTTLE_AXIS = params['throttle_joy_axis']
THROTTLE_STALL = params['throttle_stall']
STOP_BUTTON = params['stop_btn']
PAUSE_BUTTON = params['pause_btn']

# Initialize hardware
try:
    ser_pico = setup_serial(port='/dev/ttyACM0', baudrate=115200)
except:
    ser_pico = setup_serial(port='/dev/ttyACM1', baudrate=115200)
cam = setup_realsense_camera()
js = setup_joystick()

# Image Preprocessing (EfficientNet-B2 requires 260x260 input)
transform = transforms.Compose([
    transforms.ToPILImage(),  # Convert OpenCV image to PIL
    transforms.Resize((260, 260)),  # Resize to 260x260
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet Normalization
])

is_paused = True
frame_counts = 0

# Frame rate calculation variables
prev_time = time()
frame_count = 0
fps = 0

# Initialize Pygame for joystick handling
pygame.init()

# MAIN LOOP
try:
    while True:
        ret, frame = get_realsense_frame(cam)
        if not ret or frame is None:
            print("No frame received. TERMINATE!")
            break

        # Handle joystick inputs
        for e in pygame.event.get():
            if e.type == pygame.JOYBUTTONDOWN:
                if js.get_button(PAUSE_BUTTON):
                    is_paused = not is_paused
                    print(f"Autopilot {'paused' if is_paused else 'resumed'}")
                elif js.get_button(STOP_BUTTON):
                    print("E-STOP PRESSED. TERMINATE!")
                    break

        # Process the frame for prediction
        img_tensor = transform(frame).unsqueeze(0).to(device)  # Add batch dimension

        # Perform inference
        with torch.no_grad():
            pred = model(img_tensor).squeeze()

        pred_st, pred_th = pred[0].item(), pred[1].item()  # Convert tensor to scalar

        # Clip predictions to valid range
        st_trim = max(min(pred_st, 0.999), -0.999)
        th_trim = max(min(pred_th, 0.999), -0.999)

        # Encode and send commands
        if not is_paused:
            msg = encode_dutycylce(st_trim, th_trim, params)
        else:
            msg = encode(STEERING_CENTER, THROTTLE_STALL)

        ser_pico.write(msg)

        # Calculate and print frame rate
        frame_count += 1
        current_time = time()
        if current_time - prev_time >= 1.0:
            fps = frame_count / (current_time - prev_time)
            print(f"Autopilot Frame Rate: {fps:.2f} FPS")
            prev_time = current_time
            frame_count = 0

except KeyboardInterrupt:
    print("Terminated by user.")
finally:
    pygame.joystick.quit()
    ser_pico.close()
    cv.destroyAllWindows()
