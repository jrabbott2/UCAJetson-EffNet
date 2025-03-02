import os
import sys
import json
import pygame
import cv2 as cv
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
from time import time, sleep
from threading import Thread
from test_hardware import HardwareController

# Adds dummy to run Pygame without a display
os.environ["SDL_VIDEODRIVER"] = "dummy"

# Initialize Pygame early
pygame.init()
pygame.joystick.init()

if len(sys.argv) != 2:
    print("Error: Need to specify the data folder name for TensorRT engine!")
    sys.exit(1)
else:
    data_datetime = sys.argv[1]

trt_engine_path = os.path.join("models", f"TensorRT_EfficientNetB2_RGB_{data_datetime}.trt")

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def load_trt_engine(engine_path):
    if not os.path.exists(engine_path):
        print(f"Error: TensorRT engine file not found at {engine_path}")
        sys.exit(1)
    
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    return engine

engine = load_trt_engine(trt_engine_path)
context = engine.create_execution_context()

input_shape = (1, 3, 260, 260)
output_shape = (1, 2)

d_input = cuda.mem_alloc(int(np.prod(input_shape) * np.dtype(np.float32).itemsize))
d_output = cuda.mem_alloc(int(np.prod(output_shape) * np.dtype(np.float32).itemsize))
bindings = [int(d_input), int(d_output)]
stream = cuda.Stream()

def infer_tensorrt(image):
    try:
        image = cv.resize(image, (260, 260), interpolation=cv.INTER_AREA)
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        image = np.ascontiguousarray(np.expand_dims(image, axis=0))

        cuda.memcpy_htod_async(d_input, image, stream)
        if not context.execute_async_v2(bindings, stream.handle, None):
            print("❌ TensorRT inference execution failed!")
            return None, None

        output = np.empty(output_shape, dtype=np.float32)
        cuda.memcpy_dtoh_async(output, d_output, stream)
        stream.synchronize()

        return output[0]
    
    except Exception as e:
        print(f"❌ TensorRT Inference Error: {e}")
        return None, None

params_file_path = os.path.join(sys.path[0], "config.json")
with open(params_file_path) as params_file:
    params = json.load(params_file)

STEERING_CENTER = params["steering_center"]
THROTTLE_STALL = params["throttle_stall"]
STOP_BUTTON = params["stop_btn"]
PAUSE_BUTTON = params["pause_btn"]

ser_pico = None
serial_ports = ["/dev/ttyACM0", "/dev/ttyACM1"]

for port in serial_ports:
    try:
        ser_pico = HardwareController.setup_serial(port=port, baudrate=115200)
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
previous_pause_state = None
prev_button_state = 0  # Track previous button state

try:
    while True:
        pygame.event.pump()
        
        # Detect button press and print all button states
        button_states = [js.get_button(i) for i in range(js.get_numbuttons())]
        print(f"Button States: {button_states}")  # Print all button states
        button_state = js.get_button(PAUSE_BUTTON)

        if button_state == 1 and prev_button_state == 0:  # Toggle only on new press
            is_paused = not is_paused
            print(f"🟢 Autopilot {'Resumed' if not is_paused else 'Paused'}")  
            sleep(0.2)  # Prevent rapid toggling
        prev_button_state = button_state  # Store last state
        
        ret, frame = get_realsense_frame(cam)
        if not ret or frame is None:
            print("No frame received. TERMINATE!")
            break
        
        pred_st, pred_th = infer_tensorrt(frame)
        if pred_st is None or pred_th is None:
            continue

        st_trim = max(min(float(pred_st), 1.0), -1.0)
        th_trim = max(min(float(pred_th), 1.0), -1.0)

        if is_paused:
            if previous_pause_state is not True:
                if ser_pico:
                    msg = encode_dutycylce(STEERING_CENTER, THROTTLE_STALL, params)
                    ser_pico.write(msg)
                previous_pause_state = True
        else:
            previous_pause_state = False
            if (st_trim != previous_steering) or (th_trim != previous_throttle):
                if ser_pico:
                    msg = encode_dutycylce(st_trim, th_trim, params)
                    ser_pico.write(msg)
                previous_steering = st_trim
                previous_throttle = th_trim

except KeyboardInterrupt:
    print("Autopilot terminated by user.")

finally:
    if ser_pico:
        ser_pico.close()
    pygame.quit()
    print("Autopilot cleanup complete.")
