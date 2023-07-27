import os
from polygraphy import cuda
from scripts.engine import *
from scripts.onnx2trt import get_shape_dict, getOnnxPath, getEnginePath

engines = {}
stream = cuda.Stream()
USE_CUDAGRAPH = False


def get_trt_inference(model_name):
    if model_name in engines:
        return True
    return False

def setup_engine(engine_dir='./engine/',onnx_dir='./onnx/'):
    max_device_memory = 8 << 30   
    shared_device_memory = cuda.DeviceArray.raw((max_device_memory,))
    models = ['controlunet','controlnet','vae']
    for model_name in models:
        engine_path = getEnginePath(model_name, engine_dir)
        engine = Engine(engine_path)
        engine.load()
        onnx_opt_path = getOnnxPath(model_name, onnx_dir)
        max_device_memory = max(max_device_memory, 0)
        engine.activate(reuse_device_memory=shared_device_memory.ptr)
        engines[model_name] = engine 

def launch_controlunet(x, timesteps=None, context=None, control=None,):
    model_name = 'controlunet'
    device = 'cuda'
    opt_batch_size = 1
    opt_image_height = 256
    opt_image_width = 384
    if isinstance(control, list):
        feed_dict = {
            'x': x,
            'timestep': timesteps,
            'context': context,
            'control': control[0],
            'onnx::Add_4': control[1],
            'onnx::Add_5': control[2],
            'onnx::Add_6': control[3],
            'onnx::Add_7': control[4],
            'onnx::Add_8': control[5],
            'onnx::Add_9': control[6],
            'onnx::Add_10': control[7],
            'onnx::Add_11': control[8],
            'onnx::Add_12': control[9],
            'onnx::Add_13': control[10],
            'onnx::Add_14': control[11],
            'onnx::Cast_15': control[12]
        }
    engines[model_name].allocate_buffers(shape_dict=get_shape_dict(model_name, opt_batch_size, opt_image_height, opt_image_width), device=device)
    res = engines[model_name].infer(feed_dict,stream,USE_CUDAGRAPH)['latents']
    return res

def launch_controlnet( x, hint, timesteps, context):
    model_name = 'controlnet'
    device = 'cuda'
    opt_batch_size = 1
    opt_image_height = 256
    opt_image_width = 384
    feed_dict = {
        'x': x,
        'hint': hint,
        'timestep': timesteps,
        'context': context,
    }
    engines[model_name].allocate_buffers(shape_dict=get_shape_dict(model_name, opt_batch_size, opt_image_height, opt_image_width), device=device)
    temp = engines[model_name].infer(feed_dict,stream,USE_CUDAGRAPH)
    res = []
    index = 0
    for key,value in temp.items():
        if index >=4:
            res.append(value)
        index +=1
    return res

def launch_clip(input_ids):
    model_name = 'clip'
    device = 'cuda'
    opt_batch_size = 1
    opt_image_height = 256
    opt_image_width = 384
    input_ids = input_ids.type(torch.int32)
    feed_dict = {
        'input_ids': input_ids
    }
    engines[model_name].allocate_buffers(shape_dict=get_shape_dict(model_name, opt_batch_size, opt_image_height, opt_image_width), device=device)
    res = engines[model_name].infer(feed_dict,stream,USE_CUDAGRAPH)['text_embeddings']
    return res

def launch_vae(latent):
    model_name = 'vae'
    device = 'cuda'
    opt_batch_size = 1
    opt_image_height = 256
    opt_image_width = 384
    feed_dict = {
        'latent': latent
    }
    engines[model_name].allocate_buffers(shape_dict=get_shape_dict(model_name, opt_batch_size, opt_image_height, opt_image_width), device=device)
    res = engines[model_name].infer(feed_dict,stream,USE_CUDAGRAPH)['images']
    return res