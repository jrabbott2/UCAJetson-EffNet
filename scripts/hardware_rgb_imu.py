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
    """
    Configure RealSense camera pipeline for RGB and IMU streams.
    """
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 424, 240, rs.format.bgr8, 60)  # RGB stream at 60 FPS
    config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200)  # Gyro stream at 200 FPS
    config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 200)  # Accel stream at 200 FPS
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

    # Resize to target resolution (240x240) for EfficientNet-B2
    color_image_resized = cv.resize(color_image, (240, 240))

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

def read_imu_data(pipeline):
    """
    Read IMU data from the RealSense pipeline.
    Returns gyro and accel data as a tuple of NumPy arrays.
    """
    frames = pipeline.wait_for_frames()
    gyro_frame = frames.first_or_default(rs.stream.gyro)
    accel_frame = frames.first_or_default(rs.stream.accel)

    if not gyro_frame or not accel_frame:
        return None, None

    gyro_data = np.array([gyro_frame.as_motion_frame().get_motion_data().x,
                          gyro_frame.as_motion_frame().get_motion_data().y,
                          gyro_frame.as_motion_frame().get_motion_data().z])

    accel_data = np.array([accel_frame.as_motion_frame().get_motion_data().x,
                           accel_frame.as_motion_frame().get_motion_data().y,
                           accel_frame.as_motion_frame().get_motion_data().z])

    return gyro_data, accel_data

if __name__ == "__main__":
    # Setup RealSense camera
    pipeline = setup_realsense_camera()

    # Setup serial communication
    ser = setup_serial("/dev/ttyUSB0")
    if not ser:
        sys.exit(1)

    # Setup joystick
    js = setup_joystick()

    is_recording = False

    try:
        while True:
            # Get frames from RealSense camera
            success, color_frame = get_realsense_frame(pipeline)
            if not success:
                continue

            # Get IMU data
            gyro_data, accel_data = read_imu_data(pipeline)
            if gyro_data is not None and accel_data is not None:
                print(f"Gyro: {gyro_data}, Accel: {accel_data}")

            # Display color frame
            cv.imshow('RGB Stream', color_frame)

            # Handle joystick events
            for e in pygame.event.get():
                if e.type == pygame.JOYAXISMOTION:
                    ax_val_st = js.get_axis(params['steering_joy_axis'])
                    ax_val_th = js.get_axis(params['throttle_joy_axis'])
                elif e.type == pygame.JOYBUTTONDOWN:
                    if e.button == params['record_btn']:
                        is_recording = not is_recording
                        print("Recording toggled:", "ON" if is_recording else "OFF")
                    elif e.button == params['stop_btn']:
                        print("E-STOP PRESSED!")
                        ser.write(b"END,END\n")
                        raise KeyboardInterrupt
                    elif e.button == params['pause_btn']:
                        print("Paused")
                        cv.waitKey(-1)  # Pause until any key is pressed

            # Encode and send steering/throttle commands
            if 'ax_val_st' in locals() and 'ax_val_th' in locals():
                msg = encode_dutycylce(ax_val_st, ax_val_th, params)
                ser.write(msg)

            # Exit the loop if 'q' is pressed
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Exiting program.")

    finally:
        # Cleanup resources
        pipeline.stop()
        pygame.quit()
        if ser:
            ser.close()
        cv.destroyAllWindows()
