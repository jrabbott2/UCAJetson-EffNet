import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

trt_engine_path = "models/TensorRT_EfficientNetB2_RGB_2025-02-16-15-30.trt"
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def load_trt_engine(engine_path):
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

engine = load_trt_engine(trt_engine_path)
context = engine.create_execution_context()

input_shape = (1, 3, 260, 260)
output_shape = (1, 2)

d_input = cuda.mem_alloc(int(np.prod(input_shape) * np.dtype(np.float32).itemsize))
d_output = cuda.mem_alloc(int(np.prod(output_shape) * np.dtype(np.float32).itemsize))
bindings = [int(d_input), int(d_output)]
stream = cuda.Stream()

image = np.random.rand(*input_shape).astype(np.float32)

cuda.memcpy_htod_async(d_input, image, stream)
if context.execute_async_v2(bindings, stream.handle, None):
    output = np.empty(output_shape, dtype=np.float32)
    cuda.memcpy_dtoh_async(output, d_output, stream)
    stream.synchronize()
    print("✅ TensorRT Inference Successful:", output)
else:
    print("❌ TensorRT Execution Failed")
