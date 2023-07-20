from __future__ import print_function

import time
import numpy as np
import tensorrt as trt
import torch
import argparse
from PIL import ImageDraw
from cuda import cudart

import sys, os

sys.path.insert(1, os.path.join(sys.path[0], ".."))
trt.init_libnvinfer_plugins(None, "")
TRT_LOGGER = trt.Logger()
USE_TRT_INFERENCE = False
controlnet_engine_file_path = "./engines/controlnet.trt"
controlunet_engine_file_path = "./engines/controlunet.trt"
controlnet_engine = None
controlunet_engine = None
controlnet_context = None
controlunet_context = None

def get_trt_inference():
    global USE_TRT_INFERENCE
    return USE_TRT_INFERENCE

def trt_setup():
    global controlunet_engine
    global controlunet_context
    global controlnet_engine
    global controlnet_context
    global USE_TRT_INFERENCE
    
    controlunet_engine = read_engine(controlunet_engine_file_path)
    controlunet_context = set_controlunet_context(controlunet_engine)
    controlnet_engine = read_engine(controlnet_engine_file_path)
    controlnet_context = set_controlnet_context(controlnet_engine)
    USE_TRT_INFERENCE = True

def read_engine(path):
    if os.path.exists(path):
        print("Reading engine from file {}".format(path))
        with open(path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())


def set_controlnet_context(controlnet_engine):
        engine = controlnet_engine
        nIO = engine.num_io_tensors
        lTensorName = [engine.get_tensor_name(i) for i in range(nIO)]
        nInput = [engine.get_tensor_mode(lTensorName[i]) for i in range(nIO)].count(trt.TensorIOMode.INPUT)

        context = engine.create_execution_context()
        context.set_input_shape(lTensorName[0], [1, 4, 32, 48])
        context.set_input_shape(lTensorName[1], [1, 3, 256, 384])
        context.set_input_shape(lTensorName[2], [1])
        context.set_input_shape(lTensorName[3], [1, 77, 768])


        # for i in range(nIO):
        #     print("[%2d]%s->" % (i, "Input " if i < nInput else "Output"), engine.get_tensor_dtype(lTensorName[i]), engine.get_tensor_shape(lTensorName[i]), context.get_tensor_shape(lTensorName[i]), lTensorName[i])
        return context

def trt_controlnet_run(x, hint, timesteps, context):
        
        global controlnet_engine
        global controlnet_engine
    
        engine = controlnet_engine
        context = controlnet_context
        nIO = engine.num_io_tensors
        lTensorName = [engine.get_tensor_name(i) for i in range(nIO)]
        nInput = [engine.get_tensor_mode(lTensorName[i]) for i in range(nIO)].count(trt.TensorIOMode.INPUT)
        
        bufferH = []

        bufferH.append(np.ascontiguousarray(x.cpu().numpy()))
        bufferH.append(np.ascontiguousarray(hint.cpu().numpy()))
        bufferH.append(np.ascontiguousarray(timesteps.cpu().numpy()))
        bufferH.append(np.ascontiguousarray(context.cpu().numpy()))


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

        return torch.tensor(bufferH[-1]).cuda()




def set_controlunet_context(controlunet_engine):
    engine = controlunet_engine
    nIO = engine.num_io_tensors
    lTensorName = [engine.get_tensor_name(i) for i in range(nIO)]
    nInput = [engine.get_tensor_mode(lTensorName[i]) for i in range(nIO)].count(trt.TensorIOMode.INPUT)

    context = engine.create_execution_context()
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
    # for i in range(nIO):
    #     print("[%2d]%s->" % (i, "Input " if i < nInput else "Output"), engine.get_tensor_dtype(lTensorName[i]), engine.get_tensor_shape(lTensorName[i]), context.get_tensor_shape(lTensorName[i]), lTensorName[i])

    return context

def trt_controlunet_run(x, timesteps=None, context_=None, control=None, only_mid_control=False):
    global controlunet_engine
    global controlunet_engine
    
    engine = controlunet_engine
    context = controlunet_context
    nIO = engine.num_io_tensors
    lTensorName = [engine.get_tensor_name(i) for i in range(nIO)]
    nInput = [engine.get_tensor_mode(lTensorName[i]) for i in range(nIO)].count(trt.TensorIOMode.INPUT)
   
    bufferH = []
    bufferH.append(np.ascontiguousarray(x.cpu().numpy()))
    bufferH.append(np.ascontiguousarray(timesteps.cpu().numpy()))
    bufferH.append(np.ascontiguousarray(context_.cpu().numpy()))
    bufferH.append(np.ascontiguousarray(control[0].cpu().numpy()))
    bufferH.append(np.ascontiguousarray(control[1].cpu().numpy()))
    bufferH.append(np.ascontiguousarray(control[2].cpu().numpy()))
    bufferH.append(np.ascontiguousarray(control[3].cpu().numpy()))
    bufferH.append(np.ascontiguousarray(control[4].cpu().numpy()))
    bufferH.append(np.ascontiguousarray(control[5].cpu().numpy()))
    bufferH.append(np.ascontiguousarray(control[6].cpu().numpy()))
    bufferH.append(np.ascontiguousarray(control[7].cpu().numpy()))
    bufferH.append(np.ascontiguousarray(control[8].cpu().numpy()))
    bufferH.append(np.ascontiguousarray(control[9].cpu().numpy()))
    bufferH.append(np.ascontiguousarray(control[10].cpu().numpy()))
    bufferH.append(np.ascontiguousarray(control[11].cpu().numpy()))
    bufferH.append(np.ascontiguousarray(control[12].cpu().numpy()))

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

    return torch.tensor(bufferH[-1]).cuda()


