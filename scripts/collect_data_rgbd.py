import os
import sys
import json
import csv
from time import time
from datetime import datetime
import pygame
import pyrealsense2 as rs
import numpy as np
import cv2
import serial

# Load configuration parameters
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

# Initialize serial communication for microcontroller
try:
    ser_pico = serial.Serial(port='/dev/ttyACM1', baudrate=115200)
except:
    ser_pico = serial.Serial(port='/dev/ttyACM0', baudrate=115200)

# Initialize Pygame for joystick handling
pygame.init()
pygame.joystick.init()
js = pygame.joystick.Joystick(0)

# Create directories for storing collected data
data_dir = os.path.join('data', datetime.now().strftime("%Y-%m-%d-%H-%M"))
rgb_image_dir = os.path.join(data_dir, 'rgb_images/')
depth_image_dir = os.path.join(data_dir, 'depth_images/')
label_path = os.path.join(data_dir, 'labels.csv')
os.makedirs(rgb_image_dir, exist_ok=True)
os.makedirs(depth_image_dir, exist_ok=True)

# Initialize RealSense camera pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 480, 270, rs.format.bgr8, 30)  # RGB
config.enable_stream(rs.stream.depth, 480, 270, rs.format.z16, 30)  # Depth

# Start streaming
pipeline.start(config)
for i in reversed(range(90)):  # Warm-up phase
    frames = pipeline.wait_for_frames()
    if not frames:
        print("No frames received! Exiting...")
        sys.exit()
    if not i % 30:
        print(f"Starting in {i//30} seconds...")

# Initialize variables
is_recording = False
ax_val_st = 0.
ax_val_th = 0.
frame_counts = 0
start_time = time()

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        # Convert frames to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # Normalize depth image (Min-Max Scaling to 0-1 for training)
        depth_image = cv2.normalize(depth_image, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        # Resize to match EfficientNet-B2 input size (260x260)
        color_image_resized = cv2.resize(color_image, (260, 260))
        depth_image_resized = cv2.resize(depth_image, (260, 260))

        # Convert depth image to uint8 (0-255) for visualization only
        depth_display = (depth_image_resized * 255).astype(np.uint8)

        # Display RGB and Depth side by side
        stacked_display = np.hstack((color_image_resized, cv2.cvtColor(depth_display, cv2.COLOR_GRAY2BGR)))
        cv2.imshow('RGB and Depth Stream', stacked_display)

        # Handle joystick input events
        for e in pygame.event.get():
            if e.type == pygame.JOYAXISMOTION:
                ax_val_st = round(js.get_axis(STEERING_AXIS), 2)
                ax_val_th = round(js.get_axis(THROTTLE_AXIS), 2)
            elif e.type == pygame.JOYBUTTONDOWN:
                if js.get_button(RECORD_BUTTON):
                    is_recording = not is_recording
                    print("Recording:", is_recording)
                elif js.get_button(STOP_BUTTON):
                    print("Emergency Stop! Exiting...")
                    ser_pico.write(b"END,END\n")
                    raise KeyboardInterrupt

        # Calculate PWM values for steering and throttle
        act_st = -ax_val_st
        act_th = -ax_val_th

        if act_th > 0:
            duty_th = THROTTLE_STALL + int((THROTTLE_FWD_RANGE - THROTTLE_STALL) * act_th)
        elif act_th < 0:
            duty_th = THROTTLE_STALL - int((THROTTLE_STALL - THROTTLE_REV_RANGE) * abs(act_th))
        else:
            duty_th = THROTTLE_STALL

        duty_st = STEERING_CENTER - STEERING_RANGE + int(STEERING_RANGE * (act_st + 1))

        # Send control signals to microcontroller
        ser_pico.write(f"{duty_st},{duty_th}\n".encode('utf-8'))

        # Save data if recording is enabled
        if is_recording:
            # Save RGB and Depth images separately
            cv2.imwrite(os.path.join(rgb_image_dir, f"{frame_counts}_rgb.png"), color_image_resized)
            np.save(os.path.join(depth_image_dir, f"{frame_counts}_depth.npy"), depth_image_resized)  # Save depth as .npy

            # Save depth as .png for visualization (scaled to 0-255)
            depth_visual = (depth_image_resized * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(depth_image_dir, f"{frame_counts}_depth_debug.png"), depth_visual)

            # Log joystick values with filenames
            with open(label_path, 'a+', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([f"{frame_counts}_rgb.png", f"{frame_counts}_depth.npy", ax_val_st, ax_val_th])

            frame_counts += 1  # Increment frame counter

        # Exit loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Data collection terminated.")

finally:
    pipeline.stop()
    pygame.quit()
    ser_pico.close()
    cv2.destroyAllWindows()
