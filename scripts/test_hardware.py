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
        self.frame_available = False  # Track frame availability

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

    def setup_serial(self):
        """Initialize serial connection with retry mechanism"""
        ports = ["/dev/ttyACM0", "/dev/ttyACM1", "/dev/ttyUSB0", "/dev/ttyUSB1"]
        for port in ports:
            try:
                ser = serial.Serial(port, 115200, timeout=0.1)
                print(f"✅ Serial connected on {port}")
                return ser
            except serial.SerialException:
                continue
        raise RuntimeError("No available serial ports found")

    def setup_hardware(self):
        self.pipeline = self.setup_camera()
        self.ser = self.setup_serial()
        self.js = self.setup_joystick()
        print("✅ Hardware setup complete")
        
        # Start frame capture thread
        Thread(target=self.capture_frames, daemon=True).start()

    def setup_joystick(self):
        """Initialize and return the first detected joystick."""
        pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            raise Exception("No joystick detected!")
        js = pygame.joystick.Joystick(0)
        js.init()
        print(f"✅ Joystick initialized: {js.get_name()}")
        return js

    def setup_camera(self):
        """Ensure camera is initialized properly with error handling"""
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 480, 270, rs.format.bgr8, 30)
        for _ in range(5):  # Increased retries for stability
            try:
                pipeline.start(config)
                print("✅ RealSense camera initialized and streaming")
                return pipeline
            except RuntimeError as e:
                print(f"⚠️ Camera initialization failed: {e}")
                time.sleep(2)  # Wait before retrying
        raise RuntimeError("🚨 Failed to initialize camera after multiple attempts!")

    def capture_frames(self):
        """Continuously capture frames in a separate thread"""
        while self.running:
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            
            if color_frame:
                frame = np.asanyarray(color_frame.get_data())
                resized = cv.resize(frame, (260, 260), cv.INTER_AREA)
                with self.frame_lock:
                    self.current_frame = resized
                    self.frame_available = True  # Mark frame as available

    def get_current_frame(self):
        """Thread-safe frame access, ensure a frame is available before returning"""
        with self.frame_lock:
            if self.frame_available:
                self.frame_available = False  # Reset flag after reading
                return self.current_frame.copy()
            else:
                return None

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
        """Graceful shutdown procedure"""
        self.running = False
        if self.pipeline:
            self.pipeline.stop()
        if self.ser:
            self.ser.close()
        pygame.quit()
        cv.destroyAllWindows()
        print("System shutdown complete")

if __name__ == "__main__":
    controller = HardwareController()
    
    try:
        controller.setup_hardware()
        
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
