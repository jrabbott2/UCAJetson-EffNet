import serial
import pygame
import cv2 as cv
import pyrealsense2 as rs
import numpy as np
import json
import os
import sys
from threading import Thread

# Load configuration
config_path = os.path.join(os.path.dirname(__file__), "config.json")
with open(config_path, "r") as config_file:
    params = json.load(config_file)

def setup_realsense_camera():
    """
    Configure RealSense camera pipeline for RGB stream at 480x270 resolution at 30 FPS.
    """
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 480, 270, rs.format.bgr8, 30)  # ✅ Set to 30 FPS
    pipeline.start(config)
    print("✅ RealSense camera initialized with 480x270 resolution at 30 FPS")
    return pipeline

def get_realsense_frame(pipeline):
    """
    Capture frames from RealSense camera with improved stability.
    Returns resized color frame as a NumPy array.
    """
    try:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            return False, None

        color_image = np.asanyarray(color_frame.get_data())

        # Resize to target resolution (260x260) for EfficientNet-B2 with INTER_AREA for better quality
        color_image_resized = cv.resize(color_image, (260, 260), interpolation=cv.INTER_AREA)

        return True, color_image_resized
    except Exception as e:
        print(f"⚠️ RealSense Frame Capture Error: {e}")
        return False, None

def setup_serial(port=None, baudrate=115200):
    """
    Initialize a serial connection. If port is not specified, attempt automatic detection.
    """
    ports_to_try = [port] if port else ["/dev/ttyUSB0", "/dev/ttyUSB1", "/dev/ttyACM0", "/dev/ttyACM1"]

    for p in ports_to_try:
        try:
            ser = serial.Serial(port=p, baudrate=baudrate, timeout=0.1)  # ✅ Timeout prevents hangs
            print(f"✅ Serial connected on {ser.name}")
            return ser
        except serial.SerialException as e:
            print(f"⚠️ Error opening serial port {p}: {e}")

    print("❌ No available serial ports found!")
    return None

def setup_joystick():
    """
    Initialize and return the first detected joystick.
    """
    pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        raise Exception("❌ No joystick detected!")
    js = pygame.joystick.Joystick(0)
    js.init()
    print(f"✅ Joystick initialized: {js.get_name()}")
    return js

def encode_dutycylce(ax_val_st, ax_val_th, params):
    """
    Calculate duty cycle for steering and throttle based on joystick input.
    """
    # Constants from configuration
    STEERING_CENTER = params['steering_center']
    STEERING_RANGE = params['steering_range']
    THROTTLE_STALL = params['throttle_stall']
    THROTTLE_FWD_RANGE = params['throttle_fwd_range']
    THROTTLE_REV_RANGE = params['throttle_rev_range']

    # Calculate steering duty cycle
    act_st = -ax_val_st
    duty_st = STEERING_CENTER - STEERING_RANGE + int(STEERING_RANGE * (act_st + 1))

    # Calculate throttle duty cycle
    act_th = -ax_val_th
    if act_th > 0:
        duty_th = THROTTLE_STALL + int((THROTTLE_FWD_RANGE - THROTTLE_STALL) * act_th)
    elif act_th < 0:
        duty_th = THROTTLE_STALL - int((THROTTLE_STALL - THROTTLE_REV_RANGE) * abs(act_th))
    else:
        duty_th = THROTTLE_STALL

    return encode(duty_st, duty_th)

def encode(duty_st, duty_th):
    """
    Encode steering and throttle values into a message format.
    """
    return f"{duty_st},{duty_th}\n".encode('utf-8')

def process_joystick(js, ser):
    """
    Handle joystick input in a separate thread for non-blocking execution.
    """
    is_recording = False

    while True:
        for e in pygame.event.get():
            if e.type == pygame.JOYAXISMOTION:
                ax_val_st = js.get_axis(params['steering_joy_axis'])
                ax_val_th = js.get_axis(params['throttle_joy_axis'])
                msg = encode_dutycylce(ax_val_st, ax_val_th, params)
                if ser:
                    ser.write(msg)  # ✅ Non-blocking Serial Writing

            elif e.type == pygame.JOYBUTTONDOWN:
                if e.button == params['record_btn']:
                    is_recording = not is_recording
                    print(f"🎥 Recording {'ON' if is_recording else 'OFF'}")
                elif e.button == params['stop_btn']:
                    print("🛑 E-STOP PRESSED! Terminating.")
                    ser.write(b"END,END\n")
                    os._exit(0)
                elif e.button == params['pause_btn']:
                    print("⏸️ Paused")
                    cv.waitKey(-1)  # Pause until any key is pressed

        sleep(0.05)  # ✅ Prevents CPU overuse

if __name__ == "__main__":
    # Setup RealSense camera
    pipeline = setup_realsense_camera()

    # Setup serial communication
    ser = setup_serial()
    if not ser:
        sys.exit(1)

    # Setup joystick
    js = setup_joystick()

    # Start joystick processing in a separate thread
    Thread(target=process_joystick, args=(js, ser), daemon=True).start()

    try:
        while True:
            # Get frames from RealSense camera
            success, color_frame = get_realsense_frame(pipeline)
            if not success:
                continue

            # Display color frame
            cv.imshow('RGB Stream', color_frame)

            # Exit the loop if 'q' is pressed
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("🚪 Exiting program.")

    finally:
        # Cleanup resources
        pipeline.stop()
        pygame.quit()
        if ser:
            ser.close()
        cv.destroyAllWindows()
