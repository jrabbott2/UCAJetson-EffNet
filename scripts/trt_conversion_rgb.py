import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import os
import sys

# Pass in command line argument for data folder name (must match training folder)
if len(sys.argv) != 2:
    print("Error: Need to specify the data folder name for conversion!")
    sys.exit(1)
else:
    data_datetime = sys.argv[1]  # Example: "2025-02-05-14-30"

# Path to ONNX model inside the given data folder
onnx_model_path = f"/home/ucajetson/UCAJetson-EffNet/data/{data_datetime}/efficientnet_b2.onnx"

# Ensure output directory exists
output_dir = "/home/ucajetson/UCAJetson-EffNet/models/"
os.makedirs(output_dir, exist_ok=True)

# TensorRT engine will be saved with date & time in filename
trt_engine_path = os.path.join(output_dir, f"TensorRT_EfficientNetB2_RGB_{data_datetime}.trt")

# TensorRT logger
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# Function to convert ONNX to TensorRT
def build_engine(onnx_file_path, engine_file_path):
    if not os.path.exists(onnx_file_path):
        print(f"Error: ONNX model not found at {onnx_file_path}. Ensure training was completed!")
        return None

    with trt.Builder(TRT_LOGGER) as builder, \
         builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
         trt.OnnxParser(network, TRT_LOGGER) as parser:
        
        # Create builder config
        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 31  # 2GB workspace
        profile = builder.create_optimization_profile()

        # Parse the ONNX file
        with open(onnx_file_path, "rb") as model:
            if not parser.parse(model.read()):
                print("Failed to parse the ONNX file")
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None

        # Validate input shape
        input_tensor = network.get_input(0)
        input_shape = input_tensor.shape
        expected_shape = (1, 3, 260, 260)  # Static batch size, RGB channels, 260x260
        if input_shape != expected_shape:
            print(f"Error: ONNX model's input shape {input_shape} does not match expected {expected_shape}.")
            return None
        else:
            print(f"✅ Input shape is correct: {input_shape}")

        # Define input shape optimization profile
        profile.set_shape(input_tensor.name, (1, 3, 260, 260), (1, 3, 260, 260), (1, 3, 260, 260))
        config.add_optimization_profile(profile)

        # Build TensorRT engine
        print("Building TensorRT engine. This may take a few minutes...")
        try:
            engine = builder.build_engine(network, config)
        except Exception as e:
            print(f"❌ Error during engine building: {e}")
            return None

        if engine:
            with open(engine_file_path, "wb") as f:
                f.write(engine.serialize())
            print(f"✅ TensorRT engine saved to: {engine_file_path}")
        else:
            print("❌ Failed to build the TensorRT engine.")
        return engine

# Convert the ONNX model to TensorRT
build_engine(onnx_model_path, trt_engine_path)
