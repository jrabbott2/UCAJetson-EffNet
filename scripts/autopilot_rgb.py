import os
import sys
import json
import pygame
import cv2 as cv
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
from time import time
from hardware_rgb import get_realsense_frame, setup_realsense_camera, setup_serial, setup_joystick, encode_dutycylce

# Adds dummy to run Pygame without a display
os.environ["SDL_VIDEODRIVER"] = "dummy"

# Pass in command line argument for data folder name (must match TensorRT conversion)
if len(sys.argv) != 2:
    print("Error: Need to specify the data folder name for TensorRT engine!")
    sys.exit(1)
else:
    data_datetime = sys.argv[1]  # Example: "2025-02-05-14-30"

# Load TensorRT Engine
trt_engine_path = os.path.join("models", f"TensorRT_EfficientNetB2_RGB_{data_datetime}.trt")

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def load_trt_engine(engine_path):
    """ Load TensorRT engine from file """
    if not os.path.exists(engine_path):
        print(f"Error: TensorRT engine file not found at {engine_path}")
        sys.exit(1)
    
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    return engine

# Load the TensorRT engine
engine = load_trt_engine(trt_engine_path)
context = engine.create_execution_context()

# Allocate buffers for input/output
input_shape = (1, 3, 260, 260)  # Batch size 1, RGB channels, 260x260
output_shape = (1, 2)  # Output: [steering, throttle]

d_input = cuda.mem_alloc(np.prod(input_shape) * np.dtype(np.float32).itemsize)
d_output = cuda.mem_alloc(np.prod(output_shape) * np.dtype(np.float32).itemsize)
bindings = [int(d_input), int(d_output)]
stream = cuda.Stream()

def infer_tensorrt(image):
    """ Perform inference using TensorRT """
    image = cv.resize(image, (260, 260), interpolation=cv.INTER_AREA)
    image = image.astype(np.float32) / 255.0  # Normalize
    image = np.transpose(image, (2, 0, 1))  # HWC to CHW
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    cuda.memcpy_htod_async(d_input, image, stream)
    context.execute_async_v2(bindings, stream.handle, None)
    output = np.empty(output_shape, dtype=np.float32)
    cuda.memcpy_dtoh_async(output, d_output, stream)
    stream.synchronize()
    
    return output[0]

# Load configs
params_file_path = os.path.join(sys.path[0], "config.json")
with open(params_file_path) as params_file:
    params = json.load(params_file)

# Constants
STEERING_AXIS = params["steering_joy_axis"]
STEERING_CENTER = params["steering_center"]
STEERING_RANGE = params["steering_range"]
THROTTLE_AXIS = params["throttle_joy_axis"]
THROTTLE_STALL = params["throttle_stall"]
THROTTLE_FWD_RANGE = params["throttle_fwd_range"]
THROTTLE_REV_RANGE = params["throttle_rev_range"]
THROTTLE_LIMIT = params["throttle_limit"]
RECORD_BUTTON = params["record_btn"]
STOP_BUTTON = params["stop_btn"]
PAUSE_BUTTON = params["pause_btn"]

# Initialize hardware
try:
    ser_pico = setup_serial(port="/dev/ttyACM0", baudrate=115200)
except:
    ser_pico = setup_serial(port="/dev/ttyACM1", baudrate=115200)

cam = setup_realsense_camera()
js = setup_joystick()

is_paused = True
frame_counts = 0

# Frame rate calculation variables
prev_time = time()
frame_count = 0
fps = 0

# Initialize Pygame for joystick handling
pygame.init()

# MAIN LOOP
try:
    while True:
        # Get RGB frame
        ret, frame = get_realsense_frame(cam)
        if not ret or frame is None:
            print("No frame received. TERMINATE!")
            break

        for e in pygame.event.get():
            if e.type == pygame.JOYBUTTONDOWN:
                if js.get_button(PAUSE_BUTTON):
                    is_paused = not is_paused
                    print(f"Autopilot {'paused' if is_paused else 'resumed'}.")
                elif js.get_button(STOP_BUTTON):
                    print("E-STOP PRESSED. TERMINATE!")
                    break

        # Run inference using TensorRT
        pred_st, pred_th = infer_tensorrt(frame)

        # Clip predictions to valid range
        st_trim = max(min(float(pred_st), 1.0), -1.0)
        th_trim = max(min(float(pred_th), 1.0), -1.0)

        # Encode and send commands
        if not is_paused:
            msg = encode_dutycylce(st_trim, th_trim, params)
            ser_pico.write(msg)
        elif is_paused and frame_counts == 0:  # Only send stall command once
            msg = encode_dutycylce(STEERING_CENTER, THROTTLE_STALL, params)
            ser_pico.write(msg)

        # Frame rate calculation and display
        frame_count += 1
        elapsed_time = time() - prev_time
        if elapsed_time >= 1:
            fps = frame_count / elapsed_time
            frame_count = 0
            prev_time = time()
            print(f"FPS: {fps:.2f}")

except KeyboardInterrupt:
    print("Autopilot terminated by user.")

finally:
    if ser_pico:
        ser_pico.close()
    pygame.quit()
    print("Autopilot cleanup complete.")
