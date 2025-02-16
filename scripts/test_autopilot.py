import os
import sys
import json
import pygame
import cv2 as cv
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from time import time, sleep
from threading import Thread
from hardware_rgb import get_realsense_frame, setup_realsense_camera, setup_serial, setup_joystick, encode_dutycylce

# SDL Dummy to prevent pygame display errors
os.environ["SDL_VIDEODRIVER"] = "dummy"

# Initialize Pygame
pygame.init()
pygame.joystick.init()

# Ensure correct CLI argument
if len(sys.argv) != 2:
    print("❌ Error: Need to specify the data folder name for the TensorRT model!")
    sys.exit(1)
else:
    data_datetime = sys.argv[1]  # Example: "2025-02-09-13-51"

# Load TensorRT Engine
trt_engine_path = os.path.join("models", f"TensorRT_EfficientNetB2_RGB_{data_datetime}.trt")

if not os.path.exists(trt_engine_path):
    print(f"❌ Error: TensorRT engine file not found at {trt_engine_path}")
    sys.exit(1)

# TensorRT Logger
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# Load the TensorRT engine
def load_trt_engine(trt_path):
    with open(trt_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

engine = load_trt_engine(trt_engine_path)
context = engine.create_execution_context()

# Allocate memory
input_shape = (1, 3, 260, 260)  # Match EfficientNet-B2 input size
output_shape = (1, 2)  # Output: Steering, Throttle

d_input = cuda.mem_alloc(np.prod(input_shape) * np.float16().nbytes)
d_output = cuda.mem_alloc(np.prod(output_shape) * np.float16().nbytes)
bindings = [int(d_input), int(d_output)]
stream = cuda.Stream()

# Define preprocessing function
def preprocess_frame(image):
    """ Resize, normalize, and prepare frame for TensorRT inference """
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image = cv.resize(image, (260, 260), interpolation=cv.INTER_AREA)
    image = image.astype(np.float32) / 255.0  # Normalize to [0,1]
    image = np.transpose(image, (2, 0, 1))  # Convert to CHW format
    return np.expand_dims(image, axis=0).astype(np.float16)  # FP16 for TensorRT

# Define inference function
def infer_tensorrt(image):
    """ Perform inference using TensorRT """
    img_array = preprocess_frame(image)
    
    # Copy data to GPU
    cuda.memcpy_htod_async(d_input, img_array, stream)
    
    # Run inference
    start_time = time()
    context.execute_async_v2(bindings, stream.handle, None)
    inference_time = time() - start_time
    
    # Copy output from GPU
    output = np.empty(output_shape, dtype=np.float16)
    cuda.memcpy_dtoh_async(output, d_output, stream)
    stream.synchronize()

    print(f"Inference Time: {inference_time:.4f} sec | Output: {output}")  # Debug output
    return output.flatten()

# Load configuration settings
params_file_path = os.path.join(sys.path[0], "config.json")
with open(params_file_path) as params_file:
    params = json.load(params_file)

# Constants
STEERING_CENTER = params["steering_center"]
THROTTLE_STALL = params["throttle_stall"]
STOP_BUTTON = params["stop_btn"]
PAUSE_BUTTON = params["pause_btn"]

# Initialize hardware
ser_pico = None
serial_ports = ["/dev/ttyACM0", "/dev/ttyACM1"]

for port in serial_ports:
    try:
        ser_pico = setup_serial(port=port, baudrate=115200)
        if ser_pico:
            break
    except:
        print(f"⚠️ Serial not found on {port}")

if ser_pico is None:
    print("❌ No available serial ports found! Exiting.")
    sys.exit(1)

cam = setup_realsense_camera()
js = setup_joystick()

is_paused = True
previous_steering = None
previous_throttle = None
prev_frame = None  # Track last frame

# Frame rate calculation variables
prev_time = time()
frame_count = 0
fps = 0

def process_joystick():
    """ Handle joystick input in a separate thread """
    global is_paused
    while True:
        for event in pygame.event.get():
            if event.type == pygame.JOYBUTTONDOWN:
                if js.get_button(PAUSE_BUTTON):
                    is_paused = not is_paused
                    print(f"Autopilot {'paused' if is_paused else 'resumed'}")
                elif js.get_button(STOP_BUTTON):
                    print("E-STOP PRESSED. TERMINATING!")
                    os._exit(0)
        sleep(0.1)  # Avoid CPU overload

# Start joystick processing in a separate thread
Thread(target=process_joystick, daemon=True).start()

try:
    while True:
        ret, frame = get_realsense_frame(cam)
        if not ret or frame is None:
            print("❌ No frame received. TERMINATE!")
            break

        if prev_frame is not None and np.array_equal(frame, prev_frame):
            print("⚠️ Warning: RealSense frame is not changing! Potential camera issue.")
        prev_frame = frame.copy()

        if not is_paused:
            pred_st, pred_th = infer_tensorrt(frame)
            st_trim = max(min(float(pred_st), 1.0), -1.0)
            th_trim = max(min(float(pred_th), 1.0), -1.0)

            if (st_trim != previous_steering) or (th_trim != previous_throttle):
                if ser_pico:
                    msg = encode_dutycylce(st_trim, th_trim, params)
                    ser_pico.write(msg)
                previous_steering = st_trim
                previous_throttle = th_trim

        # Calculate FPS
        frame_count += 1
        elapsed_time = time() - prev_time
        if elapsed_time >= 1:
            fps = frame_count / elapsed_time
            frame_count = 0
            prev_time = time()
            print(f"📸 FPS: {fps:.2f}")

except KeyboardInterrupt:
    print("Autopilot terminated by user.")

finally:
    print("Shutting down autopilot safely...")
    pygame.event.post(pygame.event.Event(pygame.QUIT))  # Ensure all events are processed
    sleep(1)  # Allow time for threads to close
    if ser_pico:
        ser_pico.close()
    pygame.quit()
    print("✅ Autopilot cleanup complete.")
