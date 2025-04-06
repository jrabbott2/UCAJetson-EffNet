import os
import sys
import json
import csv
from time import time
from datetime import datetime
import pygame
import pyrealsense2 as rs  # Import the RealSense library
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
THROTTLE_RANGE = params['throttle_range']
RECORD_BUTTON = params['record_btn']
STOP_BUTTON = params['stop_btn']

# Initialize hardware (serial communication and joystick)
try:
    ser_pico = serial.Serial(port='/dev/ttyACM0', baudrate=115200)
except:
    ser_pico = serial.Serial(port='/dev/ttyACM1', baudrate=115200)

# Initialize Pygame for joystick handling
pygame.init()
pygame.joystick.init()
js = pygame.joystick.Joystick(0)

# Initialize storage for saving data
data_dir = os.path.join('data', datetime.now().strftime("%Y-%m-%d-%H-%M"))
image_dir = os.path.join(data_dir, 'rgb_images/')
image_dir_2 = os.path.join(data_dir, 'rgb_images_2/')
label_path = os.path.join(data_dir, 'labels.csv')
label_path_2 = os.path.join(data_dir, 'labels_2.csv')
os.makedirs(image_dir, exist_ok=True)
os.makedirs(image_dir_2, exist_ok=True)

# Initialize RealSense camera pipeline for RGB only
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 480, 270, rs.format.bgr8, 30)  # BGR stream with 480x270 resolution at 30 FPS

# Start streaming from the camera
pipeline.start(config)
for i in reversed(range(90)):
    frames = pipeline.wait_for_frames()
    if frames is None:
        print("No frame received. TERMINATE!")
        sys.exit()
    if not i % 30:
        print(i / 30)  # Countdown 3, 2, 1 seconds

# Init timer for FPS computing
start_stamp = time()
frame_counts = 0

# Initialize variables
is_recording = False
ax_val_st = 0.
ax_val_th = 0.

try:
    while True:
        # Wait for a new set of frames from the camera
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # Convert color frame to numpy array
        color_image = np.asanyarray(color_frame.get_data())

        # Resize to 260x260 for EfficientNet-B2
        resized_color_image = cv2.resize(color_image, (260, 260), interpolation=cv2.INTER_AREA)
        resized_color_image_2 = cv2.resize(color_image, (160,120), interpolation=cv2.INTER_AREA)

        # Display the RGB image
        cv2.imshow('RealSense Stream - RGB Only', resized_color_image)

        # Handle joystick input events
        for e in pygame.event.get():
            if e.type == pygame.JOYAXISMOTION:
                ax_val_st = round(js.get_axis(STEERING_AXIS), 2)
                ax_val_th = round(js.get_axis(THROTTLE_AXIS), 2)
            elif e.type == pygame.JOYBUTTONDOWN:
                if js.get_button(RECORD_BUTTON):
                    is_recording = not is_recording  # Toggle recording
                    print(f"Recording {'ON' if is_recording else 'OFF'}")
                elif js.get_button(STOP_BUTTON):
                    print("E-STOP PRESSED. TERMINATE!")
                    msg = ("END,END\n").encode('utf-8')
                    ser_pico.write(msg)
                    raise KeyboardInterrupt

        # Calculate steering and throttle values
        act_st = -ax_val_st
        act_th = -ax_val_th  # throttle action: -1: max forward, 1: max backward

        # Refined throttle control with correct forward and reverse mapping
        if act_th > 0:
            duty_th = THROTTLE_STALL + int(THROTTLE_RANGE * act_th)
        elif act_th < 0:
            duty_th = THROTTLE_STALL - int(THROTTLE_RANGE * abs(act_th))
        else:
            duty_th = THROTTLE_STALL

        # Encode steering values
        duty_st = STEERING_CENTER - STEERING_RANGE + int(STEERING_RANGE * (act_st + 1))

        # Send control signals to the microcontroller
        ser_pico.write((f"{duty_st},{duty_th}\n").encode('utf-8'))

        # Save data if recording is active
        if is_recording:
            # Save the RGB image as PNG
            cv2.imwrite(os.path.join(image_dir, f"{frame_counts}_rgb.png"), resized_color_image)
            cv2.imwrite(os.path.join(image_dir_2, f"{frame_counts}_rgb.png"), resized_color_image_2)

            # Log joystick values with image name
            with open(label_path, 'a+', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([f"{frame_counts}_rgb.png", ax_val_st, ax_val_th])
            
            with open(label_path_2, 'a+', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([f"{frame_counts}_rgb.png", ax_val_st, ax_val_th])

            frame_counts += 1  # Increment frame counter

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Terminated by user.")

finally:
    # Cleanup
    pipeline.stop()
    pygame.quit()
    ser_pico.close()
    cv2.destroyAllWindows()
