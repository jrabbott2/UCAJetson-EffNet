import os
import sys
import json
import pygame
import cv2 as cv
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
from time import time
from hardware_test import HardwareController

# Adds dummy to run Pygame without a display
os.environ["SDL_VIDEODRIVER"] = "dummy"

# Load configs
params_file_path = os.path.join(sys.path[0], 'test_config.json')
with open(params_file_path) as params_file:
    params = json.load(params_file)

# Constants
STEERING_AXIS = params['steering_joy_axis']
THROTTLE_AXIS = params['throttle_joy_axis']
PAUSE_BUTTON = params['pause_btn']
STOP_BUTTON = params['stop_btn']

# Initialize Hardware Controller
controller = HardwareController()
controller.setup_hardware()

# Load TensorRT engine
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
data_datetime = sys.argv[1]  # Example: "2025-02-16-15-30"
engine_path = f"/home/ucajetson/UCAJetson-EffNet/models/TensorRT_EfficientNetB2_RGB_{data_datetime}.trt"

with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

# Allocate buffers for TensorRT inference
def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_tensor_shape(binding))
        dtype = np.float32  # Assumed default type
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        if engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
            inputs.append((host_mem, device_mem))
        else:
            outputs.append((host_mem, device_mem))
    return inputs, outputs, bindings, stream

inputs, outputs, bindings, stream = allocate_buffers(engine)

# Frame rate calculation variables
prev_time = time()
frame_count = 0
fps = 0

# MAIN LOOP
try:
    while True:
        frame = controller.get_current_frame()
        if frame is None or frame.size == 0:
            print("⚠️ Warning: No frame received. Check camera connection!")
            break

        for e in pygame.event.get():
            if e.type == pygame.JOYBUTTONDOWN:
                if controller.js.get_button(PAUSE_BUTTON):
                    controller.is_paused = not controller.is_paused
                elif controller.js.get_button(STOP_BUTTON):
                    print("E-STOP PRESSED. TERMINATE!")
                    controller.ser.write(b"END,END\n")
                    raise KeyboardInterrupt

        # Normalize and prepare tensor
        frame = frame.astype(np.float32) / 255.0
        img_tensor = frame.transpose(2, 0, 1)  # Shape (3, 260, 260)
        img_tensor = np.expand_dims(img_tensor, axis=0)  # Add batch dimension

        # Copy to TensorRT input buffer
        np.copyto(inputs[0][0], img_tensor.ravel())

        # Run inference
        cuda.memcpy_htod_async(inputs[0][1], inputs[0][0], stream)
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(outputs[0][0], outputs[0][1], stream)
        stream.synchronize()

        # Retrieve and process predictions
        pred_st, pred_th = outputs[0][0][:2]
        st_trim = max(min(float(pred_st), 0.999), -0.999)
        th_trim = max(min(float(pred_th), 0.999), -0.999)

        # Send autopilot controls
        if not controller.is_paused:
            controller.send_autopilot_controls(st_trim, th_trim)

        # Calculate and print frame rate
        frame_count += 1
        current_time = time()
        if current_time - prev_time >= 1.0:
            fps = frame_count / (current_time - prev_time)
            print(f"Autopilot Frame Rate: {fps:.2f} FPS")
            prev_time = current_time
            frame_count = 0

except KeyboardInterrupt:
    print("Terminated by user.")
finally:
    controller.shutdown()
    pygame.quit()
    cv.destroyAllWindows()
