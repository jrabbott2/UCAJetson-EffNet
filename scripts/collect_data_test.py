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
from threading import Thread
from queue import Queue

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
RECORD_BUTTON = params['record_btn']
STOP_BUTTON = params['stop_btn']

# Initialize hardware
def init_serial():
    ports = ['/dev/ttyACM0', '/dev/ttyACM1', '/dev/ttyUSB0', '/dev/ttyUSB1']
    for port in ports:
        try:
            ser = serial.Serial(port=port, baudrate=115200, timeout=0.1)
            print(f"✅ Serial connected on {port}")
            return ser
        except serial.SerialException:
            continue
    raise RuntimeError("❌ No available serial ports found")

# Initialize RealSense
def setup_realsense():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 480, 270, rs.format.bgr8, 30)
    pipeline.start(config)
    print("✅ RealSense initialized (480x270 @30FPS)")
    return pipeline

# Joystick setup
def setup_joystick():
    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        raise RuntimeError("❌ No joystick detected")
    js = pygame.joystick.Joystick(0)
    js.init()
    print(f"✅ Joystick connected: {js.get_name()}")
    return js

# Corrected control calculations
def calculate_controls(ax_val_st, ax_val_th):
    """Convert joystick inputs to duty cycles with proper direction"""
    # Steering calculation (left=-1, right=+1)
    steering = -ax_val_st
    duty_st = STEERING_CENTER - STEERING_RANGE + int(STEERING_RANGE * (steering + 1))

    # Throttle calculation (forward=+1, reverse=-1)
    throttle = -ax_val_th  # Removed inversion
    
    if throttle > 0:  # Forward
        duty_th = THROTTLE_STALL + int((THROTTLE_FWD_RANGE - THROTTLE_STALL) * throttle)
    elif throttle < 0:  # Reverse
        duty_th = THROTTLE_STALL - int((THROTTLE_STALL - THROTTLE_REV_RANGE) * abs(throttle))
    else:  # Neutral
        duty_th = THROTTLE_STALL

    return duty_st, duty_th

def encode(duty_st, duty_th):
    return f"{duty_st},{duty_th}\n".encode('utf-8')

# Main collection loop
def main():
    # Hardware setup
    pipeline = setup_realsense()
    ser = init_serial()
    js = setup_joystick()

    # Data storage
    data_dir = os.path.join('data', datetime.now().strftime("%Y-%m-%d-%H-%M"))
    img_dir = os.path.join(data_dir, 'rgb_images')
    os.makedirs(img_dir, exist_ok=True)
    
    label_path = os.path.join(data_dir, 'labels.csv')
    open(label_path, 'w').close()  # Clear existing data

    # Thread-safe queues
    frame_queue = Queue(maxsize=30)
    control_queue = Queue()

    def capture_frames():
        """Dedicated frame capture thread"""
        frame_count = 0
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if color_frame:
                frame = np.asanyarray(color_frame.get_data())
                resized = cv2.resize(frame, (260, 260))
                frame_queue.put((frame_count, resized))
                frame_count += 1

    def process_controls():
        """Control handling thread"""
        while True:
            for event in pygame.event.get():
                if event.type == pygame.JOYAXISMOTION:
                    st = js.get_axis(STEERING_AXIS)
                    th = js.get_axis(THROTTLE_AXIS)
                    duty_st, duty_th = calculate_controls(st, th)
                    ser.write(encode(duty_st, duty_th))
                    
                    # Debug print (verify directions)
                    print(f"Steering: {st:.2f} -> {duty_st} | Throttle: {th:.2f} -> {duty_th}")
                    
                elif event.type == pygame.JOYBUTTONDOWN:
                    if event.button == STOP_BUTTON:
                        ser.write(b"END,END\n")
                        os._exit(0)

    # Start threads
    Thread(target=capture_frames, daemon=True).start()
    Thread(target=process_controls, daemon=True).start()

    try:
        frame_count = 0
        while True:
            if not frame_queue.empty():
                idx, frame = frame_queue.get()
                cv2.imshow('Preview', frame)
                cv2.waitKey(1)
                
    except KeyboardInterrupt:
        print("\n🚨 Collection stopped")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        ser.close()

if __name__ == "__main__":
    main()