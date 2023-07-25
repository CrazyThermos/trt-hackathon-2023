#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from __future__ import print_function

import time
import numpy as np
import tensorrt as trt
import argparse
from PIL import ImageDraw
from cuda import cudart

import sys, os
sys.path.insert(1, os.path.join(sys.path[0], ".."))
import common
trt.init_libnvinfer_plugins(None, "")

TRT_LOGGER = trt.Logger()

def get_args():
    parser = argparse.ArgumentParser('Export Ernie TensorRT', add_help=False)
    parser.add_argument('--onnx', default='./models/vae_encoder.onnx', type=str, help='Path of onnx file to load')
    parser.add_argument('--trt', default='./engines/vae_encoder.plan', type=str, help='Path of trt engine to save')
    parser.add_argument('--fp16', action='store_true', default=False, help='Enable FP16 mode or not, default is TF32 if it is supported')
    parser.add_argument('--int8', action='store_true', default=False, help='Enable INT8 mode or not, default is TF32 if it is supported')
    parser.add_argument('--log_level', default=1, type=int, help='Logger level. (0:VERBOSE, 1:INFO, 2:WARNING, 3:ERROR, 4:INTERNAL_ERROR)')
    parser.add_argument('--ln', action='store_true', default=True, help='Replace ops with LayernormPlugin or not')
    args = parser.parse_args()
    return args

def get_engine(args, onnx_file_path, engine_file_path="", shape=None):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""

    def build_engine():
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
            common.EXPLICIT_BATCH
        ) as network, builder.create_builder_config() as config, trt.OnnxParser(
            network, TRT_LOGGER
        ) as parser, trt.Runtime(
            TRT_LOGGER
        ) as runtime:
            config.max_workspace_size = 1 << 40  # 256MiB
            if args.fp16:
                config.set_flag(trt.BuilderFlag.FP16)
            if args.int8:
                # config.set_flag(trt.BuilderFlag.INT8)
                # config.int8_calibrator = calibrator.MyCalibrator(calibrationDataPath, nCalibration, (1, 1, nHeight, nWidth), cacheFile)
                # builder.max_batch_size = 1
                print("Unsupport int8Mode!!!")
            builder.max_batch_size = 1
            # Parse model file
            if not os.path.exists(onnx_file_path):
                print(
                    "ONNX file {} not found, please generate it.".format(onnx_file_path)
                )
                exit(0)
            print("Loading ONNX file from path {}...".format(onnx_file_path))
            with open(onnx_file_path, "rb") as model:
                print("Beginning ONNX file parsing")
                if not parser.parse(model.read()):
                    print("ERROR: Failed to parse the ONNX file.")
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    return None

            print("Completed parsing of ONNX file")
            print("Building an engine from file {}; this may take a while...".format(onnx_file_path))
            plan = builder.build_serialized_network(network, config)
            engine = runtime.deserialize_cuda_engine(plan)
            print("Completed creating Engine")
            with open(engine_file_path, "wb") as f:
                f.write(plan)
            return engine

    if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine()


def main(args, onnx_file_path, engine_file_path="", shape=None):
    """Create a TensorRT engine for ONNX-based YOLOv3-608 and run inference."""

    # Try to load a previously generated YOLOv3-608 network graph in ONNX format:
    # Do inference with TensorRT
    trt_outputs = []
    with get_engine(args, onnx_file_path, engine_file_path, shape) as engine, engine.create_execution_context() as context:
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)

        nIO = engine.num_io_tensors
        lTensorName = [engine.get_tensor_name(i) for i in range(nIO)]
        nInput = [engine.get_tensor_mode(lTensorName[i]) for i in range(nIO)].count(trt.TensorIOMode.INPUT)

        context = engine.create_execution_context()
        context.set_input_shape(lTensorName[0], [1, 4, 32, 48])

        for i in range(nIO):
            print("[%2d]%s->" % (i, "Input " if i < nInput else "Output"), engine.get_tensor_dtype(lTensorName[i]), engine.get_tensor_shape(lTensorName[i]), context.get_tensor_shape(lTensorName[i]), lTensorName[i])

        bufferH = []

        input0 = np.random.randn(1, 4, 32, 48).astype(np.float32)

        bufferH.append(np.ascontiguousarray(input0))


        for i in range(nInput, nIO):
            bufferH.append(np.empty(context.get_tensor_shape(lTensorName[i]), dtype=trt.nptype(engine.get_tensor_dtype(lTensorName[i]))))
        bufferD = []
        for i in range(nIO):
            bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])

        for i in range(nInput):
            cudart.cudaMemcpy(bufferD[i], bufferH[i].ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

        for i in range(nIO):
            context.set_tensor_address(lTensorName[i], int(bufferD[i]))
        
        time_start = time.time()  
        context.execute_async_v3(0)
        time_end = time.time()  

        for i in range(nInput, nIO):
            cudart.cudaMemcpy(bufferH[i].ctypes.data, bufferD[i], bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

        for i in range(nIO):
            print(lTensorName[i])
            print(bufferH[i])

        for b in bufferD:
            cudart.cudaFree(b)

        print("Succeeded running model in TensorRT!")
        print("inference_time:{}".format(time_end-time_start))


if __name__ == "__main__":
    onnx_file_path = "./models/vae_encoder.onnx"
    engine_file_path = "./engines/vae_encoder.plan"
    shape = None
    args = get_args()
    main(args, onnx_file_path, engine_file_path, shape)
    # get_engine(args, onnx_file_path, engine_file_path, shape)
