import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import os

# Path to your ONNX model
onnx_model_path = "/home/ucajetson/UCAJetson/data/2024-12-10-19-19/EfficientNetB2_RGBD_IMU-15ep-0.002lr-0.0051mse.onnx"  # Update ONNX file path
trt_engine_path = "/home/ucajetson/UCAJetson/models/TensorRT_EfficientNetB2_RGBD_IMU.trt"  # The output TensorRT engine file

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# Function to convert ONNX to TensorRT
def build_engine(onnx_file_path, engine_file_path):
    with trt.Builder(TRT_LOGGER) as builder, \
         builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
         trt.OnnxParser(network, TRT_LOGGER) as parser:
        
        # Create the config object and set max_workspace_size here
        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 30  # 1GB workspace
        builder.max_batch_size = 1

        # Parse the ONNX file
        with open(onnx_file_path, "rb") as model:
            if not parser.parse(model.read()):
                print("Failed to parse the ONNX file")
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None

        # Validate input shapes
        input_rgbd_shape = network.get_input(0).shape
        input_imu_shape = network.get_input(1).shape
        expected_rgbd_shape = (-1, 4, 240, 240)  # Batch size -1 (dynamic), RGB-D channels, 240x240 resolution
        expected_imu_shape = (-1, 6)  # Batch size -1 (dynamic), 6 IMU features

        if input_rgbd_shape != expected_rgbd_shape or input_imu_shape != expected_imu_shape:
            print(f"Error: Input shapes do not match expected shapes.\n"
                  f"RGBD shape: {input_rgbd_shape}, expected: {expected_rgbd_shape}\n"
                  f"IMU shape: {input_imu_shape}, expected: {expected_imu_shape}")
            return None
        else:
            print(f"Input shapes are correct: RGBD {input_rgbd_shape}, IMU {input_imu_shape}")

        # Build the TensorRT engine with the config
        print("Building TensorRT engine. This may take a few minutes...")
        engine = builder.build_engine(network, config)
        if engine:
            with open(engine_file_path, "wb") as f:
                f.write(engine.serialize())
            print(f"TensorRT engine saved to {engine_file_path}")
        else:
            print("Failed to build the TensorRT engine.")
        return engine

# Convert the ONNX model to TensorRT
build_engine(onnx_model_path, trt_engine_path)
