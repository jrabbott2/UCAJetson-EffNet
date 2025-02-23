import os
import sys
import json
import csv
import time
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
THROTTLE_LIMIT = params['throttle_limit']
RECORD_BUTTON = params['record_btn']
STOP_BUTTON = params['stop_btn']

# Initialize hardware
def init_serial():
    ports = ['/dev/ttyACM0', '/dev/ttyACM1']
    for port in ports:
        try:
            return serial.Serial(port=port, baudrate=115200, timeout=0.1)
        except serial.SerialException:
            continue
    raise RuntimeError("No microcontroller found!")

ser_pico = init_serial()

# Initialize Pygame for joystick
pygame.init()
pygame.joystick.init()
js = pygame.joystick.Joystick(0)
js.init()

# Data saving setup
data_dir = os.path.join('data', datetime.now().strftime("%Y-%m-%d-%H-%M"))
image_dir = os.path.join(data_dir, 'rgb_images')
os.makedirs(image_dir, exist_ok=True)

# RealSense pipeline configuration
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 480, 270, rs.format.bgr8, 30)

# Start pipeline with warmup
pipeline.start(config)
warmup_frames = 90  # 3 seconds at 30 FPS
for _ in range(warmup_frames):
    pipeline.wait_for_frames()

# Thread-safe data saving queue
save_queue = Queue(maxsize=100)
stop_signal = object()

# Pre-allocated image buffer
resized_buffer = np.empty((260, 260, 3), dtype=np.uint8)

def save_worker():
    csv_file = open(os.path.join(data_dir, 'labels.csv'), 'a', newline='')
    writer = csv.writer(csv_file)
    
    while True:
        item = save_queue.get()
        if item is stop_signal:
            break
        img_path, image, row = item
        cv2.imwrite(img_path, image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        writer.writerow(row)
        if save_queue.qsize() % 10 == 0:  # Flush periodically
            csv_file.flush()
    
    csv_file.close()

# Start save worker thread
saver_thread = Thread(target=save_worker, daemon=True)
saver_thread.start()

# Main loop variables
is_recording = False
frame_counts = 0
last_display_time = time.time()
display_interval = 0.1  # Update display at 10 FPS

try:
    while True:
        # Get frames without blocking (use poll instead of wait_for_frames)
        frames = pipeline.try_wait_for_frames(timeout_ms=50)
        if not frames:
            continue

        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # Process image
        color_image = np.asanyarray(color_frame.get_data())
        cv2.resize(color_image, (260, 260), dst=resized_buffer, interpolation=cv2.INTER_AREA)

        # Throttled display
        current_time = time.time()
        if current_time - last_display_time > display_interval:
            cv2.imshow('Preview', resized_buffer)
            last_display_time = current_time

        # Direct joystick polling (no event loop)
        ax_val_st = round(js.get_axis(STEERING_AXIS), 2)
        ax_val_th = round(js.get_axis(THROTTLE_AXIS), 2)

        # Handle buttons
        if js.get_button(RECORD_BUTTON):
            is_recording = not is_recording
            print(f"Recording {'STARTED' if is_recording else 'STOPPED'}")
            time.sleep(0.3)  # Debounce

        if js.get_button(STOP_BUTTON):
            print("E-STOP ACTIVATED")
            ser_pico.write(b"END,END\n")
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

        # Send controls
        ser_pico.write(f"{duty_st},{duty_th}\n".encode('utf-8'))

        # Save data
        if is_recording:
            img_name = f"{frame_counts}_rgb.jpg"
            img_path = os.path.join(image_dir, img_name)
            
            try:
                save_queue.put_nowait((
                    img_path,
                    resized_buffer.copy(),  # Copy from pre-allocated buffer
                    [img_name, ax_val_st, ax_val_th]
                ))
                frame_counts += 1
            except:
                print("Queue full - dropped frame!")

        # Exit check
        if cv2.pollKey() == ord('q'):
            break

except KeyboardInterrupt:
    print("\nShutting down...")

finally:
    pipeline.stop()
    pygame.quit()
    ser_pico.close()
    cv2.destroyAllWindows()
    save_queue.put(stop_signal)
    saver_thread.join(timeout=5)
    print(f"Saved {frame_counts} frames to {data_dir}")