import os
import sys
import json
from test_hardware import get_realsense_frame, setup_realsense_camera, setup_serial, setup_joystick, encode_dutycycle, encode
import pygame
import cv2 as cv
from time import time
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

os.environ["SDL_VIDEODRIVER"] = "dummy"
pygame.display.init()
pygame.joystick.init()
js = pygame.joystick.Joystick(0)

params_file_path = os.path.join(sys.path[0], 'config.json')
with open(params_file_path) as params_file:
    params = json.load(params_file)

STEERING_CENTER = params['steering_center']
THROTTLE_STALL = params['throttle_stall']
PAUSE_BUTTON = params['pause_btn']
STOP_BUTTON = params['stop_btn']

try:
    ser_pico = setup_serial(port='/dev/ttyACM0', baudrate=115200)
except:
    ser_pico = setup_serial(port='/dev/ttyACM1', baudrate=115200)
cam = setup_realsense_camera()
js = setup_joystick()
is_paused = True

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
data_datetime = sys.argv[1]
engine_path = f"/home/ucajetson/UCAJetson-EffNet/models/TensorRT_EfficientNetB2_RGBD_{data_datetime}.trt"

with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

def allocate_buffers(engine):
    inputs, outputs, bindings = [], [], []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_tensor_shape(binding))
        dtype = np.float32
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        if engine.binding_is_input(binding):
            inputs.append((host_mem, device_mem))
        else:
            outputs.append((host_mem, device_mem))
    return inputs, outputs, bindings, stream

inputs, outputs, bindings, stream = allocate_buffers(engine)

prev_time = time()
frame_count = 0
fps = 0

try:
    while True:
        ret, color_image, depth_image = get_realsense_frame(cam)  # Updated to return both RGB and Depth
        if not ret or color_image is None or depth_image is None:
            print("No frame received. TERMINATE!")
            break

        for e in pygame.event.get():
            if e.type == pygame.JOYBUTTONDOWN:
                if js.get_button(PAUSE_BUTTON):
                    is_paused = not is_paused
                elif js.get_button(STOP_BUTTON):
                    print("E-STOP PRESSED. TERMINATE!")
                    ser_pico.write(b"END,END\n")
                    raise KeyboardInterrupt

        # Resize and normalize RGB image
        color_resized = cv.resize(color_image, (260, 260)).astype(np.float32) / 255.0
        depth_resized = cv.resize(depth_image, (260, 260)).astype(np.float32) / 255.0
        depth_resized = np.expand_dims(depth_resized, axis=-1)  # (260, 260, 1)

        # Stack RGB and Depth into 4 channels
        rgbd_combined = np.concatenate((color_resized, depth_resized), axis=-1)
        img_tensor = rgbd_combined.transpose(2, 0, 1)  # CHW
        img_tensor = np.expand_dims(img_tensor, axis=0)  # Add batch dim

        np.copyto(inputs[0][0], img_tensor.ravel())

        cuda.memcpy_htod_async(inputs[0][1], inputs[0][0], stream)
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(outputs[0][0], outputs[0][1], stream)
        stream.synchronize()

        pred_st, pred_th = outputs[0][0][:2]
        st_trim = max(min(float(pred_st), 0.999), -0.999)
        th_trim = max(min(float(pred_th), 0.999), -0.999)

        msg = encode_dutycycle(st_trim, th_trim, params) if not is_paused else encode(STEERING_CENTER, THROTTLE_STALL)
        ser_pico.write(msg)

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
    pygame.joystick.quit()
    ser_pico.close()
    cv.destroyAllWindows()
