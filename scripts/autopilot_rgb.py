import os
import sys
import json
import torch
from hardware_rgb import get_realsense_frame, setup_realsense_camera, setup_serial, setup_joystick, encode_dutycylce
from torchvision import transforms
import pygame
import cv2 as cv
from torchvision.models import efficientnet_b2
from time import time

# Adds dummy to run Pygame without a display
os.environ["SDL_VIDEODRIVER"] = "dummy"

# Initialize only the required Pygame modules
pygame.display.init()
pygame.joystick.init()
js = pygame.joystick.Joystick(0)

# SETUP
# Load model
onnx_model_path = os.path.join('models', 'efficientnet_b2_rgb.onnx')  # Update ONNX file path
model = torch.hub.load('pytorch/vision:v0.10.0', 'efficientnet_b2', pretrained=False)
model.classifier = torch.nn.Linear(model.classifier[1].in_features, 2)
model.eval()

# Load configs
params_file_path = os.path.join(sys.path[0], 'config.json')
with open(params_file_path) as params_file:
    params = json.load(params_file)

# Constants
STEERING_AXIS = params['steering_joy_axis']
STEERING_CENTER = params['steering_center']
STEERING_RANGE = params['steering_range']
THROTTLE_AXIS = params['throttle_joy_axis']
THROTTLE_STALL = params['throttle_stall']
THROTTLE_FWD_RANGE = params['throttle_fwd_range']
THROTTLE_REV_RANGE = params['throttle_rev_range']
THROTTLE_LIMIT = params['throttle_limit']
RECORD_BUTTON = params['record_btn']
STOP_BUTTON = params['stop_btn']
PAUSE_BUTTON = params['pause_btn']

# Initialize hardware
try:
    ser_pico = setup_serial(port='/dev/ttyACM0', baudrate=115200)
except:
    ser_pico = setup_serial(port='/dev/ttyACM1', baudrate=115200)

cam = setup_realsense_camera()
js = setup_joystick()

# Preprocessing for input
to_tensor = transforms.ToTensor()
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
        # Get RGB frame
        ret, frame = get_realsense_frame(cam)
        if not ret or frame is None:
            print("No frame received. TERMINATE!")
            break

        for e in pygame.event.get():
            if e.type == pygame.JOYBUTTONDOWN:
                if js.get_button(PAUSE_BUTTON):
                    is_paused = not is_paused
                    print(f"Autopilot {'paused' if is_paused else 'resumed'}.")
                elif js.get_button(STOP_BUTTON):
                    print("E-STOP PRESSED. TERMINATE!")
                    break

        # Process the frame for prediction
        img_tensor = to_tensor(cv.resize(frame, (240, 240))).unsqueeze(0)  # Resize to 240x240 and add batch dimension
        pred_st, pred_th = model(img_tensor).squeeze()

        # Clip predictions to valid range
        st_trim = max(min(float(pred_st), 0.999), -0.999)
        th_trim = max(min(float(pred_th), 0.999), -0.999)

        # Encode and send commands
        if not is_paused:
            msg = encode_dutycylce(st_trim, th_trim, params)
        else:
            duty_st, duty_th = STEERING_CENTER, THROTTLE_STALL  # Set to center and stall if paused
            msg = encode_dutycylce(duty_st, duty_th, params)

        # Send message to hardware
        if ser_pico is not None:
            ser_pico.write(msg)

        # Frame rate calculation and display
        frame_count += 1
        if time() - prev_time >= 1:
            fps = frame_count
            frame_count = 0
            prev_time = time()
            print(f"FPS: {fps}")

except KeyboardInterrupt:
    print("Autopilot terminated by user.")

finally:
    if ser_pico:
        ser_pico.close()
    pygame.quit()
    print("Autopilot cleanup complete.")
