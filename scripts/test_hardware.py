import serial
import pygame
import cv2 as cv
import pyrealsense2 as rs
import numpy as np
import json
import os

# Load configuration from config.json
config_path = os.path.join(os.path.dirname(__file__), "config.json")
with open(config_path, "r") as config_file:
    params = json.load(config_file)

def setup_realsense_camera():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 480, 270, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 480, 270, rs.format.z16, 30)
    pipeline.start(config)
    return pipeline

def get_realsense_frame(pipeline):
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()

    if not color_frame or not depth_frame:
        return False, None, None

    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())
    return True, color_image, depth_image

def setup_serial(port, baudrate=115200):
    try:
        ser = serial.Serial(port=port, baudrate=baudrate)
        print(f"Serial connected on {ser.name}")
        return ser
    except serial.SerialException as e:
        print(f"Error opening serial port {port}: {e}")
        return None

def setup_joystick():
    pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        raise Exception("No joystick detected!")
    js = pygame.joystick.Joystick(0)
    js.init()
    print(f"Joystick initialized: {js.get_name()}")
    return js

def encode_dutycycle(ax_val_st, ax_val_th, params):
    STEERING_CENTER = params['steering_center']
    STEERING_RANGE = params['steering_range']
    THROTTLE_STALL = params['throttle_stall']
    THROTTLE_RANGE = params['throttle_range']

    act_st = -ax_val_st
    duty_st = STEERING_CENTER - STEERING_RANGE + int(STEERING_RANGE * (act_st + 1))

    act_th = -ax_val_th
    if act_th > 0:
        duty_th = THROTTLE_STALL + int(THROTTLE_RANGE * act_th)
    elif act_th < 0:
        duty_th = THROTTLE_STALL - int(THROTTLE_RANGE * abs(act_th))
    else:
        duty_th = THROTTLE_STALL

    return encode(duty_st, duty_th)

def encode(duty_st, duty_th):
    return f"{duty_st},{duty_th}\n".encode('utf-8')
