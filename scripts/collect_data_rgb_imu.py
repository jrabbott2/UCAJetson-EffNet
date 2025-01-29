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
label_path = os.path.join(data_dir, 'labels.csv')
os.makedirs(rgb_image_dir, exist_ok=True)

# Initialize RealSense camera pipeline for RGB and IMU
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 424, 240, rs.format.bgr8, 60)  # RGB stream
config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 200)  # Accelerometer stream
config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200)   # Gyroscope stream

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
        accel_frame = frames.first_or_default(rs.stream.accel)
        gyro_frame = frames.first_or_default(rs.stream.gyro)

        if not color_frame or not accel_frame or not gyro_frame:
            continue

        # Convert color frame to numpy array
        color_image = np.asanyarray(color_frame.get_data())

        # Resize to 240x240 for EfficientNet-B2
        resized_color_image = cv2.resize(color_image, (240, 240))

        # Extract IMU data
        accel_data = accel_frame.as_motion_frame().get_motion_data()
        gyro_data = gyro_frame.as_motion_frame().get_motion_data()

        # Display live IMU data
        print(f"Accel (m/s²): x={accel_data.x:.3f}, y={accel_data.y:.3f}, z={accel_data.z:.3f}")
        print(f"Gyro (rad/s): x={gyro_data.x:.3f}, y={gyro_data.y:.3f}, z={gyro_data.z:.3f}")

        # Display the RGB image
        cv2.imshow('RealSense Stream - RGB with IMU', resized_color_image)

        # Handle joystick input events
        for e in pygame.event.get():
            if e.type == pygame.JOYAXISMOTION:
                ax_val_st = round(js.get_axis(STEERING_AXIS), 2)
                ax_val_th = round(js.get_axis(THROTTLE_AXIS), 2)
            elif e.type == pygame.JOYBUTTONDOWN:
                if js.get_button(RECORD_BUTTON):
                    print("Collecting data")
                    is_recording = not is_recording  # Toggle recording
                elif js.get_button(STOP_BUTTON):
                    print("E-STOP PRESSED. TERMINATE!")
                    msg = ("END,END\n").encode('utf-8')
                    ser_pico.write(msg)
                    raise KeyboardInterrupt

        # Calculate steering and throttle values
        act_st = -ax_val_st
        act_th = -ax_val_th

        # Refined throttle control with correct forward and reverse mapping
        if act_th > 0:
            duty_th = THROTTLE_STALL + int((THROTTLE_FWD_RANGE - THROTTLE_STALL) * act_th)
        elif act_th < 0:
            duty_th = THROTTLE_STALL - int((THROTTLE_STALL - THROTTLE_REV_RANGE) * abs(act_th))
        else:
            duty_th = THROTTLE_STALL

        # Encode steering values
        duty_st = STEERING_CENTER - STEERING_RANGE + int(STEERING_RANGE * (act_st + 1))

        # Send control signals to the microcontroller
        ser_pico.write((f"{duty_st},{duty_th}\n").encode('utf-8'))

        # Save data if recording is active
        if is_recording:
            # Save the RGB image
            cv2.imwrite(os.path.join(rgb_image_dir, f"{frame_counts}_rgb.png"), resized_color_image)

            # Log joystick and IMU values with image name
            with open(label_path, 'a+', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    f"{frame_counts}_rgb.png",
                    accel_data.x, accel_data.y, accel_data.z,
                    gyro_data.x, gyro_data.y, gyro_data.z,
                    ax_val_st, ax_val_th
                ])

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
