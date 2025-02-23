import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import os
import sys
import time

# Enhanced error handling and resource management
class TRTConverter:
    def __init__(self, data_datetime):
        self.data_datetime = data_datetime
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.engine = None
        self.context = None
        
        # Path configuration
        self.onnx_path = f"/home/ucajetson/UCAJetson-EffNet/data/{data_datetime}/efficientnet_b2.onnx"
        self.trt_path = f"/home/ucajetson/UCAJetson-EffNet/models/TensorRT_EfficientNetB2_RGB_{data_datetime}.trt"
        
        # Device configuration for Jetson Orin
        self.device = cuda.Device(0)
        self.ctx = self.device.make_context()
        
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.ctx.pop()
        del self.ctx

    def build_engine(self):
        """Build TensorRT engine with optimized settings for Jetson Orin NX"""
        # 1. Validate ONNX file
        if not os.path.exists(self.onnx_path):
            raise FileNotFoundError(f"ONNX file not found at {self.onnx_path}")

        # 2. Create builder configuration
        builder = trt.Builder(self.logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, self.logger)
        
        config = builder.create_builder_config()
        config.max_workspace_size = 4 << 30  # 4GB workspace for Orin NX
        config.set_flag(trt.BuilderFlag.FP16)
        config.set_flag(trt.BuilderFlag.REFIT)  # Enable model refitting

        # 3. Dynamic shape optimization
        profile = builder.create_optimization_profile()
        input_name = network.get_input(0).name
        
        # Optimal shapes for 30 FPS pipeline
        profile.set_shape(
            input_name,
            min=(1, 3, 260, 260),   # Minimum batch size
            opt=(4, 3, 260, 260),   # Optimal batch size
            max=(8, 3, 260, 260)    # Maximum batch size
        )
        config.add_optimization_profile(profile)

        # 4. Parse and validate ONNX model
        with open(self.onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                raise ValueError("ONNX parsing failed. Errors:\n" + 
                                "\n".join([str(parser.get_error(e)) 
                                         for e in range(parser.num_errors)]))

        # 5. Build engine with timing instrumentation
        start_time = time.time()
        serialized_engine = builder.build_serialized_network(network, config)
        
        if not serialized_engine:
            raise RuntimeError("Engine build failed")
            
        # 6. Save and verify engine
        with open(self.trt_path, "wb") as f:
            f.write(serialized_engine)
            
        # Verify engine can be loaded
        runtime = trt.Runtime(self.logger)
        self.engine = runtime.deserialize_cuda_engine(serialized_engine)
        self.context = self.engine.create_execution_context()
        
        print(f"✅ Conversion successful! ({time.time()-start_time:.2f}s)")
        print(f"Engine saved to: {self.trt_path}")
        return self.trt_path

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python trt_conversion_rgb.py <data_folder>")
        
    try:
        with TRTConverter(sys.argv[1]) as converter:
            engine_path = converter.build_engine()
            
            # Verify engine dimensions
            print("\nEngine Specifications:")
            print(f"- Input shape: {converter.context.get_binding_shape(0)}")
            print(f"- Output shape: {converter.context.get_binding_shape(1)}")
            print(f"- FP16 Enabled: {converter.engine.get_tensor_mode(converter.engine[0]) == trt.TensorDataType.HALF}")
            
    except Exception as e:
        print(f"❌ Conversion failed: {str(e)}")
        sys.exit(1)