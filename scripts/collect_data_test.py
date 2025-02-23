import os
import sys
import json
import time
import pygame
import cv2
import pyrealsense2 as rs
import numpy as np
import serial
from threading import Thread, Lock
from time import sleep

# Load configuration
config_path = os.path.join(os.path.dirname(__file__), "config.json")
with open(config_path, "r") as config_file:
    params = json.load(config_file)

# Global variables for inter-thread communication
current_frame = None
frame_lock = Lock()
joystick_events = []
event_lock = Lock()

def setup_realsense_camera():
    """Configure RealSense camera pipeline for RGB stream."""
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 480, 270, rs.format.bgr8, 30)
    pipeline.start(config)
    print("✅ RealSense camera initialized")
    return pipeline

def get_realsense_frame(pipeline):
    """Capture frames from RealSense camera."""
    global current_frame
    try:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            return False
        
        color_image = np.asanyarray(color_frame.get_data())
        with frame_lock:
            current_frame = cv2.resize(color_image, (260, 260), interpolation=cv2.INTER_AREA)
        return True
    except Exception as e:
        print(f"⚠️ Frame capture error: {e}")
        return False

def setup_serial():
    """Initialize serial connection."""
    ports = ["/dev/ttyACM0", "/dev/ttyACM1", "/dev/ttyUSB0"]
    for port in ports:
        try:
            ser = serial.Serial(port=port, baudrate=115200, timeout=0.1)
            print(f"✅ Serial connected on {port}")
            return ser
        except serial.SerialException:
            continue
    print("❌ No serial devices found!")
    return None

def setup_joystick():
    """Initialize and return the first detected joystick."""
    pygame.init()
    pygame.joystick.init()
    
    if pygame.joystick.get_count() == 0:
        raise RuntimeError("❌ No joystick detected!")
    
    js = pygame.joystick.Joystick(0)
    js.init()
    print(f"✅ Joystick initialized: {js.get_name()}")
    return js

def process_joystick(ser, params):
    """Handle joystick data processing in a separate thread."""
    global current_frame
    is_recording = False
    frame_count = 0
    data_dir = os.path.join('data', time.strftime("%Y-%m-%d-%H-%M"))
    os.makedirs(data_dir, exist_ok=True)
    
    while True:
        events = []
        with event_lock:
            events = joystick_events.copy()
            joystick_events.clear()
        
        for e in events:
            if e.type == pygame.JOYAXISMOTION:
                ax_val_st = e.value if e.axis == params['steering_joy_axis'] else 0
                ax_val_th = e.value if e.axis == params['throttle_joy_axis'] else 0
                
                # Send controls
                if ser:
                    msg = encode_dutycycle(ax_val_st, ax_val_th, params)
                    ser.write(msg)
                
                # Save data if recording
                if is_recording and current_frame is not None:
                    img_dir = os.path.join(data_dir, 'rgb_images')
                    os.makedirs(img_dir, exist_ok=True)
                    img_path = os.path.join(img_dir, f"{frame_count}_rgb.jpg")
                    label_path = os.path.join(data_dir, 'labels.csv')
                    
                    cv2.imwrite(img_path, current_frame)
                    with open(label_path, 'a') as f:
                        f.write(f"{frame_count}_rgb.jpg,{ax_val_st:.4f},{ax_val_th:.4f}\n")
                    frame_count += 1
            
            elif e.type == pygame.JOYBUTTONDOWN:
                if e.button == params['record_btn']:
                    is_recording = not is_recording
                    print(f"🎥 Recording {'started' if is_recording else 'stopped'}")
                elif e.button == params['stop_btn']:
                    print("🛑 E-STOP PRESSED! Terminating.")
                    if ser:
                        ser.write(b"END,END\n")
                    os._exit(0)
        
        sleep(0.05)

def encode_dutycycle(ax_val_st, ax_val_th, params):
    """Calculate duty cycle for steering and throttle."""
    st_center = params['steering_center']
    st_range = params['steering_range']
    th_stall = params['throttle_stall']
    th_fwd = params['throttle_fwd_range']
    th_rev = params['throttle_rev_range']

    # Steering calculation
    duty_st = st_center - st_range + int(st_range * (-ax_val_st + 1))
    
    # Throttle calculation
    if ax_val_th > 0:
        duty_th = th_stall + int((th_fwd - th_stall) * ax_val_th)
    elif ax_val_th < 0:
        duty_th = th_stall - int((th_stall - th_rev) * abs(ax_val_th))
    else:
        duty_th = th_stall
    
    return f"{duty_st},{duty_th}\n".encode()

if __name__ == "__main__":
    # Initialize hardware
    pipeline = setup_realsense_camera()
    ser = setup_serial()
    js = setup_joystick()
    
    # Start processing thread
    Thread(target=process_joystick, args=(ser, params), daemon=True).start()

    try:
        while True:
            # Capture events in main thread
            with event_lock:
                joystick_events.extend(pygame.event.get())
            
            # Update camera feed
            if get_realsense_frame(pipeline) and current_frame is not None:
                cv2.imshow('RGB Stream', current_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("🚪 Exiting program.")
    finally:
        pipeline.stop()
        pygame.quit()
        if ser:
            ser.close()
        cv2.destroyAllWindows()