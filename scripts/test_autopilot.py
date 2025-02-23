import os
import sys
import json
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import tensorrt as trt
import pygame
import cv2 as cv
from time import time, sleep
from hardware_rgb import (
    get_realsense_frame, setup_realsense_camera, setup_serial, 
    setup_joystick, encode_dutycylce
)

# Pygame Dummy Display (Prevent UI issues)
os.environ["SDL_VIDEODRIVER"] = "dummy"
pygame.init()
pygame.joystick.init()

# Ensure correct CLI input
if len(sys.argv) != 2:
    print("❌ Error: Need to specify the model's timestamp folder!")
    sys.exit(1)
else:
    data_datetime = sys.argv[1]  # Example: "2025-02-16-15-30"

# Load TensorRT engine
engine_path = f"models/TensorRT_EfficientNetB2_RGB_{data_datetime}.trt"
if not os.path.exists(engine_path):
    print(f"❌ TensorRT model file not found: {engine_path}")
    sys.exit(1)

print(f"🔄 Loading TensorRT model from {engine_path}...")

# TensorRT Logger
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# Load TensorRT engine
def load_trt_engine(engine_path):
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

engine = load_trt_engine(engine_path)
context = engine.create_execution_context()

# Allocate Device Memory (Fixing PyCUDA TypeError)
input_shape = tuple(engine.get_tensor_shape(engine.get_tensor_name(0)))  # ✅ Use new method
output_shape = tuple(engine.get_tensor_shape(engine.get_tensor_name(1)))  # ✅ Use new method

input_size = int(np.prod(input_shape) * np.dtype(np.float16).itemsize)  
output_size = int(np.prod(output_shape) * np.dtype(np.float16).itemsize)  

d_input = cuda.mem_alloc(input_size)
d_output = cuda.mem_alloc(output_size)
stream = cuda.Stream()

# Ensure the camera is set up
cam = setup_realsense_camera()
js = setup_joystick()

ser_pico = None  # Prevents crashes if serial fails

# ✅ **Optimized Image Preprocessing (Fix Normalization Error)**
def preprocess_image(frame):
    """
    EfficientNet-B2 expects 260x260 images with ImageNet normalization.
    """
    frame_resized = cv.resize(frame, (260, 260), interpolation=cv.INTER_AREA).astype("float32") / 255.0
    frame_resized = np.transpose(frame_resized, (2, 0, 1))  # Change to CHW format

    # **✅ Fix Normalization (reshape constants)**
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    frame_resized = (frame_resized - mean) / std  # ✅ Now correctly broadcasted

    return np.ascontiguousarray(frame_resized, dtype=np.float16)  # Convert to FP16

# Run inference
def infer_tensorrt(image):
    """
    Performs inference using TensorRT with PyCUDA memory management.
    """
    np.copyto(cuda.pagelocked_empty(input_shape, dtype=np.float16), image.ravel())

    # Copy data to GPU
    cuda.memcpy_htod_async(d_input, image, stream)
    
    # Run inference
    context.execute_async_v2(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)

    # Copy results back
    output = np.empty(output_shape, dtype=np.float16)
    cuda.memcpy_dtoh_async(output, d_output, stream)
    stream.synchronize()

    return output.flatten()

# MAIN LOOP
is_paused = True
frame_counts = 0
prev_time = time()
frame_count = 0
fps = 0

try:
    ser_pico = setup_serial(port='/dev/ttyACM0', baudrate=115200)  # ✅ Serial Setup Moved Inside Try
except:
    ser_pico = setup_serial(port='/dev/ttyACM1', baudrate=115200)

try:
    while True:
        ret, frame = get_realsense_frame(cam)
        if not ret or frame is None:
            print("No frame received. TERMINATE!")
            break

        for e in pygame.event.get():
            if e.type == pygame.JOYBUTTONDOWN:
                if js.get_button(1):  # PAUSE Button
                    is_paused = not is_paused
                    print(f"Autopilot {'paused' if is_paused else 'resumed'}")
                elif js.get_button(0):  # STOP Button
                    print("E-STOP PRESSED. TERMINATE!")
                    break

        # Preprocess image
        img_input = preprocess_image(frame)

        # Run TensorRT inference
        pred_st, pred_th = infer_tensorrt(img_input)

        # Clip predictions
        st_trim = max(min(float(pred_st), 0.999), -0.999)
        th_trim = max(min(float(pred_th), 0.999), -0.999)

        # Encode and send commands
        if not is_paused:
            msg = encode_dutycylce(st_trim, th_trim, {})
        else:
            msg = encode(0, 0)

        # Send to Pico
        if ser_pico:
            ser_pico.write(msg)

        # Calculate frame rate
        frame_count += 1
        current_time = time()
        if current_time - prev_time >= 1.0:
            fps = frame_count / (current_time - prev_time)
            print(f"🚀 Autopilot Frame Rate: {fps:.2f} FPS")
            prev_time = current_time
            frame_count = 0

except KeyboardInterrupt:
    print("Terminated by user.")
finally:
    pygame.quit()
    if ser_pico:
        ser_pico.close()  # ✅ Prevents crashes if serial wasn't initialized
    cam.stop()
    cv.destroyAllWindows()
