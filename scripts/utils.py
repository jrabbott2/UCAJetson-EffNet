import serial
import pygame
import cv2 as cv
import pyrealsense2 as rs
import numpy as np
import json
import os

# Load configuration from test_config.json
config_path = os.path.join(os.path.dirname(__file__), "config.json")
with open(config_path, "r") as config_file:
    params = json.load(config_file)


def setup_realsense_camera():
    """
    Configure RealSense camera pipeline for RGB stream at 424x240 resolution.
    """
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 480, 270, rs.format.bgr8, 30)  # RGB stream at 30 FPS
    pipeline.start(config)
    return pipeline


def get_realsense_frame(pipeline):
    """
    Capture frames from the RealSense camera pipeline.
    Returns resized color frame as a NumPy array.
    """
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()

    if not color_frame:
        return False, None

    color_image = np.asanyarray(color_frame.get_data())
    color_image_resized = cv.resize(color_image, (260, 260))  # EfficientNet input resolution
    return True, color_image_resized


def setup_serial(port, baudrate=115200):
    """
    Initialize a serial connection.
    """
    try:
        ser = serial.Serial(port=port, baudrate=baudrate)
        print(f"Serial connected on {ser.name}")
        return ser
    except serial.SerialException as e:
        print(f"Error opening serial port {port}: {e}")
        return None


def setup_joystick():
    """
    Initialize and return the first detected joystick.
    """
    pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        raise Exception("No joystick detected!")
    js = pygame.joystick.Joystick(0)
    js.init()
    print(f"Joystick initialized: {js.get_name()}")
    return js


def encode(ax_val_st, ax_val_th, params):
    """
    Calculate duty cycle for steering and throttle based on joystick input.
    """
    act_st = -ax_val_st
    duty_st = params['steering_center'] - params['steering_range'] + int(params['steering_range'] * (act_st + 1))

    act_th = -ax_val_th
    if act_th > 0:
            duty_th = THROTTLE_STALL + int(min(THROTTLE_RANGE * act_th, THROTTLE_RANGE * THROTTLE_LIMIT))
        duty_th = params['throttle_stall'] + int(min(params['throttle_range'] * act_th, params['throttle_range'] * params['throttle_limit']))
    elif act_th < 0:
        duty_th = params['throttle_stall'] - int(min(params['throttle_range'] * abs(act_th), params['throttle_range'] * params['throttle_limit']))
    else:
        duty_th = params['throttle_stall']

    return f"{duty_st},{duty_th}\n".encode('utf-8')

