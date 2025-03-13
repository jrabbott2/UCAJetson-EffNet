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

# Initialize hardware (serial communication and joystick)
try:
    ser_pico = serial.Serial(port='/dev/ttyACM1', baudrate=115200)
except:
    ser_pico = serial.Serial(port='/dev/ttyACM0', baudrate=115200)

# Initialize Pygame for joystick handling
pygame.init()
pygame.joystick.init()
js = pygame.joystick.Joystick(0)

# Initialize storage for saving data
data_dir = os.path.join('data', datetime.now().strftime("%Y-%m-%d-%H-%M"))
rgb_image_dir = os.path.join(data_dir, 'rgb_images/')
depth_image_dir = os.path.join(data_dir, 'depth_images/')
label_path = os.path.join(data_dir, 'labels.csv')
os.makedirs(rgb_image_dir, exist_ok=True)
os.makedirs(depth_image_dir, exist_ok=True)

# Initialize RealSense camera pipeline for RGB and Depth
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 480, 270, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 480, 270, rs.format.z16, 30)

# Start streaming from the camera
pipeline.start(config)

# Countdown before starting data collection
for i in reversed(range(90)):
    frames = pipeline.wait_for_frames()
    if frames is None:
        print("No frame received. TERMINATE!")
        sys.exit()
    if not i % 30:
        print(i / 30)

# Initialize variables
is_recording = False
ax_val_st = 0.
ax_val_th = 0.
frame_counts = 0

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

        # Resize images
        resized_color = cv2.resize(color_image, (260, 260))
        normalized_depth = np.log1p(depth_image)
        normalized_depth = cv2.normalize(normalized_depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        resized_depth = cv2.resize(normalized_depth, (260, 260))

        # Display
        cv2.imshow('RGB', resized_color)
        cv2.imshow('Depth', resized_depth)

        # Handle joystick input
        for e in pygame.event.get():
            if e.type == pygame.JOYAXISMOTION:
                ax_val_st = round(js.get_axis(STEERING_AXIS), 2)
                ax_val_th = round(js.get_axis(THROTTLE_AXIS), 2)
            elif e.type == pygame.JOYBUTTONDOWN:
                if js.get_button(RECORD_BUTTON):
                    print("Collecting data")
                    is_recording = not is_recording
                elif js.get_button(STOP_BUTTON):
                    print("E-STOP PRESSED. TERMINATE!")
                    msg = ("END,END\n").encode('utf-8')
                    ser_pico.write(msg)
                    raise KeyboardInterrupt

        # Calculate control values
        act_st = -ax_val_st
        act_th = -ax_val_th

        if act_th > 0:
            duty_th = THROTTLE_STALL + int((THROTTLE_FWD_RANGE - THROTTLE_STALL) * act_th)
        elif act_th < 0:
            duty_th = THROTTLE_STALL - int((THROTTLE_STALL - THROTTLE_REV_RANGE) * abs(act_th))
        else:
            duty_th = THROTTLE_STALL

        duty_st = STEERING_CENTER - STEERING_RANGE + int(STEERING_RANGE * (act_st + 1))

        # Send control signals
        ser_pico.write((f"{duty_st},{duty_th}\n").encode('utf-8'))

        # Save data
        if is_recording:
            cv2.imwrite(os.path.join(rgb_image_dir, f"{frame_counts}_rgb.png"), resized_color)
            cv2.imwrite(os.path.join(depth_image_dir, f"{frame_counts}_depth.png"), resized_depth)
            with open(label_path, 'a+', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([f"{frame_counts}_rgb.png", f"{frame_counts}_depth.png", ax_val_st, ax_val_th])
            frame_counts += 1

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Terminated by user.")

finally:
    pipeline.stop()
    pygame.quit()
    ser_pico.close()
    cv2.destroyAllWindows()
