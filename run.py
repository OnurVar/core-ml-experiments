import coremltools as ct
import tensorflow as tf
import numpy as np
import concurrent.futures
import time
import argparse
import os

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run Core ML model inference.")
    parser.add_argument("--model", type=str, default="EfficientNetB7", choices=["ResNet50", "EfficientNetB7", "InceptionV3", "MobileNetV2", "DenseNet121"], help="Model to use for inference")
    parser.add_argument("--unit", type=str, default="ALL", choices=["ALL", "CPU_ONLY", "CPU_AND_GPU", "CPU_AND_NE"], help="Compute unit to use for inference")
    parser.add_argument("--iterations", type=int, default=10000, help="Number of iterations for continuous inferences")
    return parser.parse_args()

def load_model(model_name):
    model_dict = {
        "ResNet50": (tf.keras.applications.ResNet50, (1, 224, 224, 3)),
        "EfficientNetB7": (tf.keras.applications.EfficientNetB7, (1, 600, 600, 3)),
        "InceptionV3": (tf.keras.applications.InceptionV3, (1, 299, 299, 3)),
        "MobileNetV2": (tf.keras.applications.MobileNetV2, (1, 224, 224, 3)),
        "DenseNet121": (tf.keras.applications.DenseNet121, (1, 224, 224, 3))
    }
    return model_dict[model_name]

def convert_model(model, input_shape, model_name):
    if not os.path.exists(f"{model_name}.mlpackage"):
        coreml_model = ct.convert(
            model, 
            inputs=[ct.TensorType(shape=input_shape)], 
            minimum_deployment_target=ct.target.macOS13,
        )
        coreml_model.save(f"{model_name}.mlpackage")
        print(f"Model converted and saved as {model_name}.mlpackage")
    else:
        print(f"Model already exists. Skipping conversion.")

def load_coreml_model(model_name, compute_unit):
    compute_unit_dict = {
        "ALL": ct.ComputeUnit.ALL,
        "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
        "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
        "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE
    }
    print(f"Loading Core ML model with compute unit: {compute_unit}")
    return ct.models.MLModel(f"{model_name}.mlpackage", compute_units=compute_unit_dict[compute_unit])

def run_inference(mlmodel, input_data):
    return mlmodel.predict(input_data)

def measure_inference_time(mlmodel, input_data, description):
    start_time = time.time()
    output = mlmodel.predict(input_data)
    end_time = time.time()
    print(f"{description} took {end_time - start_time:.2f} seconds")
    return output

def main():
    args = parse_arguments()
    print(f"Selected model: {args.model}")
    print(f"Selected compute unit: {args.unit}")
    print(f"Selected iterations: {args.iterations}")

    model_class, input_shape = load_model(args.model)
    model = model_class(weights='imagenet')
    model_name = args.model

    convert_model(model, input_shape, model_name)
    mlmodel_ane = load_coreml_model(model_name, args.unit)

    input_data_batch = {"input_1": np.random.rand(1, input_shape[1], input_shape[2], 3).astype(np.float32)}

    # Warmup session
    print("Starting warmup session...")
    for _ in range(10):
        run_inference(mlmodel_ane, input_data_batch)
    print("Warmup session completed.")

    measure_inference_time(mlmodel_ane, input_data_batch, "Inference for batch size 32")

    start_time = time.time()
    for _ in range(args.iterations):  # Use iterations value
        run_inference(mlmodel_ane, input_data_batch)
    end_time = time.time()
    print(f"Continuous inferences for {args.iterations} iterations took {end_time - start_time:.2f} seconds")

    inputs = [{"input_1": np.random.rand(1, input_shape[1], input_shape[2], 3).astype(np.float32)} for _ in range(10)]
    start_time = time.time()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(lambda x: run_inference(mlmodel_ane, x), inputs))
    end_time = time.time()
    print(f"Parallel inferences (10 concurrent) took {end_time - start_time:.2f} seconds")

    input_data_high_res = {"input_1": np.random.rand(1, input_shape[1], input_shape[2], 3).astype(np.float32)}  # Match input shape
    measure_inference_time(mlmodel_ane, input_data_high_res, f"Inference for high-resolution input ({input_shape[1]}x{input_shape[2]})")

    input_data_large_batch = [{"input_1": np.random.rand(1, input_shape[1], input_shape[2], 3).astype(np.float32)} for _ in range(64)]
    start_time = time.time()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(lambda x: run_inference(mlmodel_ane, x), input_data_large_batch))
    end_time = time.time()
    print(f"Parallel inferences for batch size 64 took {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
