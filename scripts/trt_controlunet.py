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
    parser.add_argument('--onnx', default='./models/controlunet.onnx', type=str, help='Path of onnx file to load')
    parser.add_argument('--trt', default='./engines/controlunet.trt', type=str, help='Path of trt engine to save')
    parser.add_argument('--fp16', action='store_true', default=False, help='Enable FP16 mode or not, default is TF32 if it is supported')
    parser.add_argument('--int8', action='store_true', default=False, help='Enable INT8 mode or not, default is TF32 if it is supported')
    parser.add_argument('--only_export', action='store_true', default=False, help='Only export onnx to trt')
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
            # Parse model file
            if not os.path.exists(onnx_file_path):
                print(
                    "ONNX file {} not found, please generate it.".format(onnx_file_path)
                )
                exit(0)
            print("Loading ONNX file from path {}...".format(onnx_file_path))
            with open(onnx_file_path, "rb") as model:
                print("Beginning ONNX file parsing")
                os.chdir("./models")
                if not parser.parse(model.read()):
                    print("ERROR: Failed to parse the ONNX file.")
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    return None
            os.chdir("../")

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
        # Do inference
        # Set host input to the image. The common.do_inference function will copy the input to the GPU before executing.

        nIO = engine.num_io_tensors
        lTensorName = [engine.get_tensor_name(i) for i in range(nIO)]
        nInput = [engine.get_tensor_mode(lTensorName[i]) for i in range(nIO)].count(trt.TensorIOMode.INPUT)

        context = engine.create_execution_context()
        _, stream = cudart.cudaStreamCreate()
        context.set_input_shape(lTensorName[0], [1, 4, 32, 48])
        context.set_input_shape(lTensorName[1], [1])
        context.set_input_shape(lTensorName[2], [1, 77, 768])
        context.set_input_shape(lTensorName[3], [1, 320, 32, 48])
        context.set_input_shape(lTensorName[4], [1, 320, 32, 48])
        context.set_input_shape(lTensorName[5], [1, 320, 32, 48])
        context.set_input_shape(lTensorName[6], [1, 320, 16, 24])
        context.set_input_shape(lTensorName[7], [1, 640, 16, 24])
        context.set_input_shape(lTensorName[8], [1, 640, 16, 24])
        context.set_input_shape(lTensorName[9], [1, 640, 8, 12])
        context.set_input_shape(lTensorName[10], [1, 1280, 8, 12])
        context.set_input_shape(lTensorName[11], [1, 1280, 8, 12])
        context.set_input_shape(lTensorName[12], [1, 1280, 4, 6])
        context.set_input_shape(lTensorName[13], [1, 1280, 4, 6])
        context.set_input_shape(lTensorName[14], [1, 1280, 4, 6])
        context.set_input_shape(lTensorName[15], [1, 1280, 4, 6])

        for i in range(nIO):
            print("[%2d]%s->" % (i, "Input " if i < nInput else "Output"), engine.get_tensor_dtype(lTensorName[i]), engine.get_tensor_shape(lTensorName[i]), context.get_tensor_shape(lTensorName[i]), lTensorName[i])

        bufferH = []
        input0 = np.random.randn(1, 4, 32, 48).astype(np.float16)
        input1 = np.array([951]).astype(np.float16)
        input2 = np.random.randn(1, 77, 768).astype(np.float16)
        input3 = np.random.randn(1,320,32,48).astype(np.float16)
        input4 = np.random.randn(1,320,32,48).astype(np.float16)
        input5 = np.random.randn(1,320,32,48).astype(np.float16)
        input6 = np.random.randn(1,320,16,24).astype(np.float16)
        input7 = np.random.randn(1,640,16,24).astype(np.float16)
        input8 = np.random.randn(1,640,16,24).astype(np.float16)
        input9 = np.random.randn(1,640,8,12).astype(np.float16)
        input10 = np.random.randn(1,1280,8,12).astype(np.float16)
        input11 = np.random.randn(1,1280,8,12).astype(np.float16)
        input12 = np.random.randn(1,1280,4,6).astype(np.float16)
        input13 = np.random.randn(1,1280,4,6).astype(np.float16)
        input14 = np.random.randn(1,1280,4,6).astype(np.float16)
        input15 = np.random.randn(1,1280,4,6).astype(np.float16)

        bufferH.append(np.ascontiguousarray(input0))
        bufferH.append(np.ascontiguousarray(input1))
        bufferH.append(np.ascontiguousarray(input2))
        bufferH.append(np.ascontiguousarray(input3))
        bufferH.append(np.ascontiguousarray(input4))
        bufferH.append(np.ascontiguousarray(input5))
        bufferH.append(np.ascontiguousarray(input6))
        bufferH.append(np.ascontiguousarray(input7))
        bufferH.append(np.ascontiguousarray(input8))
        bufferH.append(np.ascontiguousarray(input9))
        bufferH.append(np.ascontiguousarray(input10))
        bufferH.append(np.ascontiguousarray(input11))
        bufferH.append(np.ascontiguousarray(input12))
        bufferH.append(np.ascontiguousarray(input13))
        bufferH.append(np.ascontiguousarray(input14))
        bufferH.append(np.ascontiguousarray(input15))


        for i in range(nInput, nIO):
            bufferH.append(np.empty(context.get_tensor_shape(lTensorName[i]), dtype=trt.nptype(engine.get_tensor_dtype(lTensorName[i]))))
        bufferD = []
        for i in range(nIO):
            bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])

        for i in range(nInput):
            cudart.cudaMemcpyAsync(bufferD[i], bufferH[i].ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)

        for i in range(nIO):
            context.set_tensor_address(lTensorName[i], int(bufferD[i]))
        
        time_start = time.time()  
        context.execute_async_v3(stream)
        time_end = time.time()  

        for i in range(nInput, nIO):
            cudart.cudaMemcpyAsync(bufferH[i].ctypes.data, bufferD[i], bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)

        # CUDA Graph capture
        cudart.cudaStreamBeginCapture(stream, cudart.cudaStreamCaptureMode.cudaStreamCaptureModeGlobal)
        for i in range(nInput):
            cudart.cudaMemcpyAsync(bufferD[i], bufferH[i].ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
        context.execute_async_v3(stream)        
        for i in range(nInput, nIO):
            cudart.cudaMemcpyAsync(bufferH[i].ctypes.data, bufferD[i], bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
        #cudart.cudaStreamSynchronize(stream)  # no need to synchronize within the CUDA graph capture
        _, graph = cudart.cudaStreamEndCapture(stream)
        _, graphExe = cudart.cudaGraphInstantiate(graph, 0)
        bufferH[-1] *= 0
        cudart.cudaGraphLaunch(graphExe, stream)
        cudart.cudaStreamSynchronize(stream)
        
        for i in range(nIO):
            print(lTensorName[i])
        print(bufferH[-1])

        for b in bufferD:
            cudart.cudaFree(b)
        cudart.cudaStreamDestroy(stream)
        
        print("Succeeded running model in TensorRT!")
        print("inference_time:{}".format(time_end-time_start))



if __name__ == "__main__":
    onnx_file_path = "./models/controlunet.onnx"
    engine_file_path = "./engines/controlunet.plan"
    shape = None
    args = get_args()
    if args.only_export:
        get_engine(args, onnx_file_path, engine_file_path, shape)
    else:
        main(args, onnx_file_path, engine_file_path, shape)