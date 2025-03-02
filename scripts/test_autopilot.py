import os
import sys
import json
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import tensorrt as trt
import cv2
import time
from hardware_test import HardwareController  # Updated import

# TensorRT Initialization
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

class TensorRTInference:
    def __init__(self, engine_path):
        self.engine = self.load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        
        # Allocate device memory
        self.input_shape = self.engine.get_tensor_shape(self.engine.get_tensor_name(0))
        self.output_shape = self.engine.get_tensor_shape(self.engine.get_tensor_name(1))
        
        self.d_input = cuda.mem_alloc(int(np.prod(self.input_shape)) * 2)  # FP16
        self.d_output = cuda.mem_alloc(int(np.prod(self.output_shape)) * 2)

    def load_engine(self, engine_path):
        with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def infer(self, image):
        # Preprocess and copy to device
        processed = self.preprocess(image)
        cuda.memcpy_htod_async(self.d_input, processed, self.stream)
        
        # Execute inference
        self.context.execute_async_v2(
            bindings=[int(self.d_input), int(self.d_output)],
            stream_handle=self.stream.handle
        )
        
        # Retrieve results
        output = np.empty(self.output_shape, dtype=np.float16)
        cuda.memcpy_dtoh_async(output, self.d_output, self.stream)
        self.stream.synchronize()
        
        return output.flatten()

    def preprocess(self, frame):
        # Convert to CHW format and normalize
        frame = cv2.resize(frame, (260, 260), cv2.INTER_AREA).astype(np.float32)
        frame = frame.transpose(2, 0, 1) / 255.0
        frame = (frame - np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3,1,1)) \
              / np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3,1,1)
        return frame.astype(np.float16).ravel()

class AutopilotSystem:
    def __init__(self, model_timestamp):
        self.controller = HardwareController()
        self.trt_engine = TensorRTInference(
            f"models/TensorRT_EfficientNetB2_RGB_{model_timestamp}.trt"
        )
        self.fps = 0
        self.frame_count = 0
        self.last_time = time.time()

    def run(self):
        try:
            self.controller.setup_hardware()
            
            while True:
                start_time = time.time()
                
                # Get frame from hardware controller
                frame = self.controller.get_current_frame()
                if frame is None:
                    continue
                
                # Run inference
                pred = self.trt_engine.infer(frame)
                steering, throttle = pred[0], pred[1]
                
                # Send controls if not paused
                if not self.controller.is_paused:
                    self.controller.send_autopilot_controls(steering, throttle)
                
                # Update FPS counter
                self.update_fps(steering, throttle)
                
                # Emergency stop check
                if self.controller.emergency_stop:
                    break

        finally:
            self.controller.shutdown()

    def update_fps(self, steering, throttle):
        self.frame_count += 1
        elapsed = time.time() - self.last_time
        if elapsed > 1.0:
            self.fps = self.frame_count / elapsed
            print(f"FPS: {self.fps:.2f} | Steering: {steering:.2f} | Throttle: {throttle:.2f}")
            self.frame_count = 0
            self.last_time = time.time()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python autopilot_test.py <model_timestamp>")
    
    autopilot = AutopilotSystem(sys.argv[1])
    autopilot.run()
