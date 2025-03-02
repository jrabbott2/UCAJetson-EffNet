import serial
import pygame
import cv2 as cv
import pyrealsense2 as rs
import numpy as np
import json
import os
import sys
from threading import Thread, Lock
import time

class HardwareController:
    def __init__(self):
        self.config = self.load_config()
        self.pipeline = None
        self.ser = None
        self.js = None
        self.frame_lock = Lock()
        self.control_lock = Lock()
        self.current_frame = None
        self.current_controls = (0, 0)
        self.running = True
        self.is_paused = False

    def load_config(self):
        """Load and validate configuration file"""
        config_path = os.path.join(os.path.dirname(__file__), "test_config.json")
        with open(config_path) as f:
            config = json.load(f)
        
        required_keys = {
            'steering_joy_axis', 'throttle_joy_axis', 'record_btn',
            'stop_btn', 'pause_btn', 'steering_center', 'steering_range',
            'throttle_stall', 'throttle_fwd_range', 'throttle_rev_range'
        }
        if not required_keys.issubset(config.keys()):
            raise ValueError("Missing required configuration parameters")
        return config

    def setup_hardware(self):
        self.ser = self.setup_serial('/dev/ttyACM0', 115200)  # Ensure serial is set up before use
        """Initialize camera, serial, and joystick"""
        self.pipeline = self.setup_camera()
        self.ser = self.setup_serial('/dev/ttyACM0', 115200)
        self.js = self.setup_joystick()
        print("✅ Hardware setup complete")

    def setup_camera(self):
        """Initialize RealSense camera with retry mechanism"""
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 480, 270, rs.format.bgr8, 30)
        
        for _ in range(3):  # Retry mechanism
            try:
                pipeline.start(config)
                print("✅ RealSense camera initialized")
                return pipeline
            except RuntimeError as e:
                print(f"Camera initialization failed: {e}")
                time.sleep(1)
        raise RuntimeError("Failed to initialize camera after 3 attempts")

    def get_current_frame(self):
        """Thread-safe frame access"""
        with self.frame_lock:
            return self.current_frame.copy() if self.current_frame is not None else None

    def send_autopilot_controls(self, steering, throttle):
        """Thread-safe control command sending"""
        with self.control_lock:
            duty_st, duty_th = self.calculate_duty_cycles(steering, throttle)
            cmd = f"{duty_st},{duty_th}\n".encode()
            self.ser.write(cmd)
            self.current_controls = (duty_st, duty_th)

    def calculate_duty_cycles(self, steering, throttle):
        """Convert steering and throttle values to PWM duty cycles"""
        st = np.clip(-steering, -1.0, 1.0)
        th = np.clip(-throttle, -1.0, 1.0)

        params = self.config
        duty_st = params['steering_center'] - params['steering_range'] + \
                 int(params['steering_range'] * (st + 1))

        if th > 0:
            duty_th = params['throttle_stall'] + \
                     int((params['throttle_fwd_range'] - params['throttle_stall']) * th)
        else:
            duty_th = params['throttle_stall'] - \
                     int((params['throttle_stall'] - params['throttle_rev_range']) * abs(th))
        
        return duty_st, duty_th

    def shutdown(self):
        if self.ser:
            self.ser.close()
        """Graceful shutdown procedure"""
        self.running = False
        self.pipeline.stop()
        self.ser.close()
        pygame.quit()
        cv.destroyAllWindows()
        print("System shutdown complete")

if __name__ == "__main__":
    controller = HardwareController()
    
    try:
        controller.setup_hardware()
        Thread(target=controller.capture_frames, daemon=True).start()
        
        while controller.running:
            frame = controller.get_current_frame()
            if frame is not None:
                cv.imshow('Camera Feed', frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"Critical error: {e}")
    finally:
        controller.shutdown()
