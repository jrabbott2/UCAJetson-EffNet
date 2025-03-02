import serial
import pygame
import cv2
import pyrealsense2 as rs
import numpy as np
import json
import os
import sys
from threading import Thread, Lock
import time

class HardwareController:
    def __init__(self):
        self.is_paused = False  # Added missing attribute
        self.emergency_stop = False  # Added missing attribute
        self.config = self.load_config()
        self.pipeline = None
        self.ser = None
        self.js = None
        self.frame_lock = Lock()
        self.control_lock = Lock()
        self.current_frame = None
        self.current_controls = (0, 0)
        self.running = True
        self.recording = False
    def send_autopilot_controls(self, steering, throttle):
        """Send autopilot control signals to the vehicle"""
        duty_st, duty_th = self.calculate_duty_cycles(steering, throttle)
        cmd = f"{duty_st},{duty_th}".encode()
        self.ser.write(cmd)
        self.current_controls = (duty_st, duty_th)

    def calculate_duty_cycles(self, steering, throttle):
        """Convert steering and throttle values to duty cycles"""
        st = np.clip(-steering, -1.0, 1.0)
        th = np.clip(-throttle, -1.0, 1.0)
        params = self.config
        duty_st = params['steering_center'] - params['steering_range'] + int(params['steering_range'] * (st + 1))
        if th > 0:
            duty_th = params['throttle_stall'] + int((params['throttle_fwd_range'] - params['throttle_stall']) * th)
        else:
            duty_th = params['throttle_stall'] - int((params['throttle_stall'] - params['throttle_rev_range']) * abs(th))
        return duty_st, duty_th


    def load_config(self):
        """Load and validate configuration file"""
        config_path = os.path.join(os.path.dirname(__file__), "config.json")
        with open(config_path) as f:
            config = json.load(f)
        
        required_keys = {
            'steering_joy_axis', 'throttle_joy_axis', 'record_btn',
            'stop_btn', 'steering_center', 'steering_range',
            'throttle_stall', 'throttle_fwd_range', 'throttle_rev_range'
        }
        if not required_keys.issubset(config.keys()):
            raise ValueError("Missing required configuration parameters")
        return config

    def setup_hardware(self):
        """Initialize all hardware components"""
        self.pipeline = self.setup_camera()
        self.ser = self.setup_serial()
        self.js = self.setup_joystick()
        print("✅ Hardware setup complete")
        
        # Start worker threads
        Thread(target=self.capture_frames, daemon=True).start()
        Thread(target=self.process_controls, daemon=True).start()

    def setup_camera(self):
        """Initialize RealSense camera with error recovery"""
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

    def setup_serial(self):
        """Auto-detect and initialize serial connection"""
        ports = ["/dev/ttyUSB0", "/dev/ttyUSB1", "/dev/ttyACM0", "/dev/ttyACM1"]
        for port in ports:
            try:
                ser = serial.Serial(port, 115200, timeout=0.1)
                print(f"✅ Serial connected on {port}")
                return ser
            except serial.SerialException:
                continue
        raise RuntimeError("No available serial ports found")

    def setup_joystick(self):
        """Initialize joystick with pygame"""
        pygame.init()
        pygame.joystick.init()
        
        if pygame.joystick.get_count() == 0:
            raise RuntimeError("No joystick detected")
            
        js = pygame.joystick.Joystick(0)
        js.init()
        print(f"✅ Joystick initialized: {js.get_name()}")
        return js

    def capture_frames(self):
        """Dedicated frame capture thread with error handling"""
        while self.running:
            try:
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                
                if color_frame:
                    frame = np.asanyarray(color_frame.get_data())
                    resized = cv2.resize(frame, (260, 260), cv2.INTER_AREA)
                    
                    with self.frame_lock:
                        self.current_frame = resized
            except Exception as e:
                print(f"Frame capture error: {e}")
                time.sleep(0.1)
    
    def process_controls(self):
        """Joystick processing thread with rate limiting"""
        last_send = time.time()
        
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.JOYAXISMOTION:
                    st = self.js.get_axis(self.config['steering_joy_axis'])
                    th = self.js.get_axis(self.config['throttle_joy_axis'])
                    
                    # Rate limit control updates
                    if time.time() - last_send > 0.02:  # 50Hz
                        self.send_controls(st, th)
                        last_send = time.time()

    def get_current_frame(self):
        """Thread-safe frame access"""
        with self.frame_lock:
            return self.current_frame.copy() if self.current_frame is not None else None

    def shutdown(self):
        """Graceful shutdown procedure"""
        self.running = False
        if self.pipeline:
            self.pipeline.stop()
        if self.ser:
            self.ser.close()
        pygame.quit()
        cv2.destroyAllWindows()
        print("System shutdown complete")

if __name__ == "__main__":
    controller = HardwareController()
    
    try:
        controller.setup_hardware()
    except Exception as e:
        print(f"Critical error: {e}")
    finally:
        controller.shutdown()
