"""
 Copyright (c) 2022 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

"""
This script is tested with TensorFlow v2.9.1 and OpenVINO v2022.2.0
Usage Example below. -tf, -ov are required, rest are optional:
python run_ov_tf_perf.py \
-tf https://tfhub.dev/google/bit/m-r50x1/1 \
-ov ov_irs/bit_m_r50x1_1/saved_model.xml \
-d CPU \
-i https://upload.wikimedia.org/wikipedia/commons/6/6e/Golde33443.jpg \
-s '1,128,128,3' \
-t 10
"""

import os, sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import tensorflow_hub as hub
from openvino.runtime import Core
import cv2
import numpy as np
import time
import subprocess
import argparse
import pathlib
from urllib.parse import urlparse

# For top 5 labels.
MAX_PREDS = 5
args = ()


def parse_args():
    global args

    parser = argparse.ArgumentParser(
        description="Script to benchmark BiT model with TensorFlow and OpenVINO"
    )
    optional = parser._action_groups.pop()
    required = parser.add_argument_group("required arguments")
    required.add_argument(
        "-tf", "--tfhub_url", help="TensorFlow HUB BiT model URL", required=True
    )
    required.add_argument(
        "-ov", "--ov_xml", help="Path to OpenVINO model XML file", required=True
    )
    optional.add_argument(
        "-d", "--target_device", help="Specify a target device to infer on. ", required=False,
        default="CPU",
    )
    optional.add_argument(
        "-i", "--input_image", help="Input Image URL or Path to image.", required=False,
        default="",
    )
    optional.add_argument(
        "-s", "--shape", help="Set shape for input 'N,W,H,C'. For example: '1,128,128,3' ", required=False,
        default="1,128,128,3",
    )
    optional.add_argument(
        "-t", "--bench_time", help="Benchmark duration in seconds", required=False, default=10,
    )
    parser._action_groups.append(optional)

    return parser.parse_args()


# Pre-process input test image
def preprocess_image(input_image):

    url_parsed = urlparse(input_image)
    # If input_image is URL, download the image as 'test-image.'
    if url_parsed.scheme not in (""):  # Not a local file
        file_ext = pathlib.Path(url_parsed.path).suffix
        inp_file_name = f"test-image{file_ext}"
        print(f"\nDownloading input test image {input_image}")
        output = subprocess.check_output(
            f"curl {input_image} --output {inp_file_name}", shell=True
        )
    else:
        inp_file_name = input_image
    
    print("\nPre-processing input image...")
    
    # Setup the input shape
    [bs, w, h, c] = args.shape

    if os.path.exists(inp_file_name):
        image = cv2.imread(inp_file_name)
        image = np.array(image)
        img_resized = tf.image.resize(image, [w, h])
        img_reshaped = tf.reshape(img_resized, [bs, w, h, c])
        image = tf.image.convert_image_dtype(img_reshaped, tf.float32)
    else:
        print(f"Input image not found: {inp_file_name}. Initializing with random image of size {args.shape}." )
        image = np.random.random(size=(bs, w, h, c))

    return image


def benchmark_tf(test_image):
    # Load model into KerasLayer
    tf_module = hub.KerasLayer(args.tfhub_url)

    latency_arr = []
    end = time.time() + int(args.bench_time)

    print(f"\n==== Benchmarking TensorFlow inference for {args.bench_time}sec on {args.target_device} ====")
    print(f"Input shape: {test_image.shape}")
    print(f"Model: {args.tfhub_url} ")

    while time.time() < end:
        start_time = time.time()
        tf_result = tf_module(test_image)
        latency = time.time() - start_time
        latency_arr.append(latency)

    tf_result = tf.reshape(tf_result, [-1])
    top5_label_idx = np.argsort(tf_result)[-MAX_PREDS::][::-1]

    avg_latency = np.array(latency_arr).mean()
    fps = 1 / avg_latency

    print(f"Avg Latency: {avg_latency:.4f} sec, FPS: {fps:.2f}")
    print(f"TF Inference top5 label index(s): {top5_label_idx}")

    return avg_latency, fps, top5_label_idx


def benchmark_ov(test_image):
    # Initialize the OV inference engine
    ie = Core()

    # Load and compile the OV model
    ov_model = ie.read_model(args.ov_xml)
    compiled_model = ie.compile_model(model=ov_model, device_name=args.target_device)

    # get the names of input and output layers of the model
    input_layer = compiled_model.input(0)
    output_layer = compiled_model.output(0)

    latency_arr = []
    end = time.time() + int(args.bench_time)
    print(f"\n==== Benchmarking OpenVINO inference for {args.bench_time}sec on {args.target_device} ====")
    print(f"Input shape: {test_image.shape}")
    print(f"Model: {args.ov_xml} ")

    while time.time() < end:
        start_time = time.time()
        ov_result = compiled_model([test_image])
        latency = time.time() - start_time
        latency_arr.append(latency)

    # Save the result for accuracy verificaiton
    ov_result = ov_result[output_layer]
    ov_result = np.reshape(ov_result, [-1])
    top5_label_idx = np.argsort(ov_result)[-MAX_PREDS::][::-1]

    avg_latency = np.array(latency_arr).mean()
    fps = 1 / avg_latency

    print(f"Avg Latency: {avg_latency:.4f} sec, FPS: {fps:.2f}")
    print(f"OV Inference top5 label index(s): {top5_label_idx} \n")

    return avg_latency, fps, top5_label_idx


def main():
    global args
    args = parse_args()

    if isinstance(args.shape, str):
        args.shape = [int(i) for i in args.shape.split(",")]
        if len(args.shape) != 4:
            sys.exit( "Input shape error. Set shape 'N,W,H,C'. For example: '1,128,128,3' " )

    test_image = preprocess_image(args.input_image)

    ov_avg_latency, ov_fps, ov_top5_label_idx = benchmark_ov(test_image)
    tf_avg_latency, ov_fps, tf_top5_label_idx = benchmark_tf(test_image)

    acc_diff = ov_top5_label_idx - tf_top5_label_idx
    if np.sum(acc_diff) == 0:
        print(f"\nBoth TensorFlow and OpenVINO reported same accuracy.")
    else:
        print(f"\n Accuracy MISMATCHED for TensorFlow and OpenVINO !")

    speedup_ov = tf_avg_latency / ov_avg_latency
    print(f"\nSpeedup on {args.target_device} with OpenVINO: {speedup_ov:.1f}x \n")


if __name__ == "__main__":
    main()
