import os
import sys
import json
import csv
from time import time, sleep
from datetime import datetime
import threading
import pygame
import pyrealsense2 as rs
import numpy as np
import cv2
import serial
from collections import deque  # Import deque from collections

# Load configs
params_file_path = os.path.join(sys.path[0], 'config.json')
try:
    with open(params_file_path) as params_file:
        params = json.load(params_file)
except Exception as e:
    print(f"Failed to load config file: {e}")
    sys.exit(1)

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

# JPEG quality (adjust as needed)
JPEG_QUALITY = 95

# Frame rate limiter (adjust as needed)
TARGET_FPS = 30
FRAME_DELAY = 1.0 / TARGET_FPS

# Initialize hardware (serial communication and joystick)
try:
    ser_pico = serial.Serial(port='/dev/ttyACM1', baudrate=115200)
except Exception as e:
    print(f"Failed to connect to /dev/ttyACM1: {e}. Trying /dev/ttyACM0...")
    try:
        ser_pico = serial.Serial(port='/dev/ttyACM0', baudrate=115200)
    except Exception as e:
        print(f"Failed to connect to serial port: {e}")
        sys.exit(1)

# Initialize Pygame for joystick handling
pygame.init()
pygame.joystick.init()
js = pygame.joystick.Joystick(0)
js.init()

# Initialize storage for saving data
data_dir = os.path.join('data', datetime.now().strftime("%Y-%m-%d-%H-%M"))
image_dir = os.path.join(data_dir, 'rgb_images/')
label_path = os.path.join(data_dir, 'labels.csv')
os.makedirs(image_dir, exist_ok=True)

# Initialize RealSense camera pipeline for RGB only
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 480, 270, rs.format.bgr8, TARGET_FPS)  # Fixed resolution: 480x270

# Start streaming from the camera
pipeline.start(config)

# Frame buffer for multithreading
frame_buffer = deque(maxlen=1)  # Now works because deque is imported

# Function to capture frames in a separate thread
def capture_frames():
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if color_frame:
            frame_buffer.append(np.asanyarray(color_frame.get_data()))

# Start the frame capture thread
capture_thread = threading.Thread(target=capture_frames, daemon=True)
capture_thread.start()

# Initialize variables
is_recording = False
ax_val_st = 0.0
ax_val_th = 0.0
frame_counts = 0
last_frame_time = time()

# Open CSV file for writing (only once)
csv_file = None
csv_writer = None

try:
    while True:
        current_time = time()
        elapsed_time = current_time - last_frame_time

        # Frame rate limiter
        if elapsed_time < FRAME_DELAY:
            sleep(FRAME_DELAY - elapsed_time)
        last_frame_time = time()

        # Get the latest frame from the buffer
        if not frame_buffer:
            continue
        color_image = frame_buffer[-1]

        # Resize to 260x260 for EfficientNet-B2
        resized_color_image = cv2.resize(color_image, (260, 260), interpolation=cv2.INTER_AREA)

        # Display the RGB image (optional, for debugging)
        cv2.imshow('RealSense Stream - RGB Only', resized_color_image)

        # Handle joystick input events
        for e in pygame.event.get():
            if e.type == pygame.JOYAXISMOTION:
                ax_val_st = round(js.get_axis(STEERING_AXIS), 2)
                ax_val_th = round(js.get_axis(THROTTLE_AXIS), 2)
            elif e.type == pygame.JOYBUTTONDOWN:
                if js.get_button(RECORD_BUTTON):
                    is_recording = not is_recording  # Toggle recording
                    if is_recording:
                        print("Recording started")
                        csv_file = open(label_path, 'a+', newline='')
                        csv_writer = csv.writer(csv_file)
                    else:
                        print("Recording stopped")
                        if csv_file:
                            csv_file.close()
                            csv_file = None
                            csv_writer = None
                elif js.get_button(STOP_BUTTON):
                    print("E-STOP PRESSED. TERMINATE!")
                    ser_pico.write("END,END\n".encode('utf-8'))
                    raise KeyboardInterrupt

        # Calculate steering and throttle values
        act_st = -ax_val_st
        act_th = -ax_val_th

        if act_th > 0:
            duty_th = THROTTLE_STALL + int((THROTTLE_FWD_RANGE - THROTTLE_STALL) * act_th)
        elif act_th < 0:
            duty_th = THROTTLE_STALL - int((THROTTLE_STALL - THROTTLE_REV_RANGE) * abs(act_th))
        else:
            duty_th = THROTTLE_STALL

        duty_st = STEERING_CENTER - STEERING_RANGE + int(STEERING_RANGE * (act_st + 1))

        # Send control signals to the microcontroller
        ser_pico.write(f"{duty_st},{duty_th}\n".encode('utf-8'))

        # Save data if recording is active
        if is_recording and csv_writer:
            image_path = os.path.join(image_dir, f"{frame_counts}_rgb.jpg")
            cv2.imwrite(image_path, resized_color_image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            csv_writer.writerow([f"{frame_counts}_rgb.jpg", ax_val_st, ax_val_th])
            frame_counts += 1

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
    if csv_file:
        csv_file.close()
    cv2.destroyAllWindows()