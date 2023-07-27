
import os
from engine import *

engines = {}

def check_dims(batch_size, image_height, image_width):
    _min_batch = 1
    _max_batch = 16
    min_latent_shape = 256//8
    max_latent_shape = 1024//8
    assert batch_size >= _min_batch and batch_size <= _max_batch
    assert image_height % 8 == 0 or image_width % 8 == 0
    latent_height = image_height // 8
    latent_width = image_width // 8
    assert latent_height >= min_latent_shape and latent_height <= max_latent_shape
    assert latent_width >= min_latent_shape and latent_width <= max_latent_shape
    return (latent_height, latent_width)

def get_minmax_dims( batch_size, image_height, image_width, static_batch, static_shape):
    _min_batch = 1
    _max_batch = 16
    min_image_shape = 256
    max_image_shape = 1024
    min_latent_shape = min_image_shape//8
    max_latent_shape = max_image_shape//8

    min_batch = batch_size if static_batch else _min_batch
    max_batch = batch_size if static_batch else _max_batch
    latent_height = image_height // 8
    latent_width = image_width // 8
    min_image_height = image_height if static_shape else min_image_shape
    max_image_height = image_height if static_shape else max_image_shape
    min_image_width = image_width if static_shape else min_image_shape
    max_image_width = image_width if static_shape else max_image_shape
    min_latent_height = latent_height if static_shape else min_latent_shape
    max_latent_height = latent_height if static_shape else max_latent_shape
    min_latent_width = latent_width if static_shape else min_latent_shape
    max_latent_width = latent_width if static_shape else max_latent_shape
    return (min_batch, max_batch, min_image_height, max_image_height, min_image_width, max_image_width, min_latent_height, max_latent_height, min_latent_width, max_latent_width)



def getOnnxPath( model_name, onnx_dir, opt=True):
    return os.path.join(onnx_dir, model_name+('.opt' if opt else '')+'.onnx')

def getEnginePath( model_name, engine_dir):
    return os.path.join(engine_dir, model_name+'.plan')

def get_input_profile(model_name,
                    batch_size, image_height, image_width,
                    static_batch, static_shape):
    unet_dim = 4
    text_maxlen = 77
    embedding_dim = 768
    latent_height, latent_width = check_dims(batch_size, image_height, image_width)
    min_batch, max_batch, min_image_height, max_image_height, min_image_width, max_image_width, min_latent_height, max_latent_height, min_latent_width, max_latent_width = get_minmax_dims(batch_size, image_height, image_width, static_batch, static_shape)
    if model_name == 'controlunet':
        return {
            'x': [(min_batch, unet_dim, min_latent_height, min_latent_width), (batch_size, unet_dim, latent_height, latent_width), (max_batch, unet_dim, max_latent_height, max_latent_width)],
            'context': [(min_batch, text_maxlen, embedding_dim), (batch_size, text_maxlen, embedding_dim), (max_batch, text_maxlen, embedding_dim)]
        }
    elif model_name == 'controlnet':
        return {
            'x': [(min_batch, unet_dim, min_latent_height, min_latent_width), (batch_size, unet_dim, latent_height, latent_width), (max_batch, unet_dim, max_latent_height, max_latent_width)],
            'hint': [(batch_size, 3, min_image_height,min_image_width), (batch_size, 3, max_image_height,max_image_width), (batch_size, 3, max_image_height,max_image_width)],
            'context': [(min_batch, text_maxlen, embedding_dim), (batch_size, text_maxlen, embedding_dim), (max_batch, text_maxlen, embedding_dim)],
        }
    elif model_name == 'clip':
        return {
            'input_ids': [(min_batch, text_maxlen), (batch_size, text_maxlen), (max_batch, text_maxlen)]
        }
    elif model_name == 'vae':
        return {
            'latent': [(min_batch, 4, min_latent_height, min_latent_width), (batch_size, 4, latent_height, latent_width), (max_batch, 4, max_latent_height, max_latent_width)]
        }
    
def get_shape_dict(model_name, batch_size, image_height=256, image_width=384):
    unet_dim = 4
    text_maxlen = 77
    embedding_dim = 768
    latent_height, latent_width = check_dims(batch_size, image_height, image_width)
    if model_name == 'controlunet':
        return {
            'x': (batch_size, unet_dim, latent_height, latent_width),
            'context': (batch_size, text_maxlen, embedding_dim),
            'latents': (batch_size, 4, latent_height, latent_width)
        }
    elif model_name == 'controlnet':
        return {
            'x': (batch_size, unet_dim, latent_height, latent_width),
            'hint': (batch_size, 3, image_height, image_width),
            'context': (batch_size, text_maxlen, embedding_dim),
        }
    elif model_name == 'clip':
        return {
            'input_ids': (batch_size, text_maxlen),
            'text_embeddings': (batch_size, text_maxlen, embedding_dim),
            'pooler_output': (batch_size, embedding_dim)
        }
    elif model_name == 'vae':
        return {
            'images': (batch_size, 3, image_height, image_width),
            'latent': (batch_size, 4, latent_height, latent_width)
        }
    
    
def get_sample_input_dict(model_name, batch_size, image_height, image_width, use_fp16, device):
    unet_dim = 4
    text_maxlen = 77
    embedding_dim = 768
    latent_height, latent_width = check_dims(batch_size, image_height, image_width)
    dtype = torch.float16 if use_fp16 else torch.float32
    if model_name == 'controlunet':
        return {
            'x': torch.randn(batch_size, unet_dim, latent_height, latent_width, dtype=torch.float32, device=device),
            'timestep': torch.tensor([1.], dtype=torch.float32, device=device),
            'context': torch.randn(batch_size, text_maxlen, embedding_dim, dtype=dtype, device=device),
            'control': torch.randn(batch_size,320,32,48,dtype=torch.float32, device=device),
            'onnx::Add_4': torch.randn(batch_size,320,32,48,dtype=torch.float32, device=device),
            'onnx::Add_5': torch.randn(batch_size,320,32,48,dtype=torch.float32, device=device),
            'onnx::Add_6': torch.randn(batch_size,320,16,24,dtype=torch.float32, device=device),
            'onnx::Add_7': torch.randn(batch_size,640,16,24,dtype=torch.float32, device=device),
            'onnx::Add_8': torch.randn(batch_size,640,16,24,dtype=torch.float32, device=device),
            'onnx::Add_9': torch.randn(batch_size,640,8,12,dtype=torch.float32, device=device),
            'onnx::Add_10': torch.randn(batch_size,1280,8,12,dtype=torch.float32, device=device),
            'onnx::Add_11': torch.randn(batch_size,1280,8,12,dtype=torch.float32, device=device),
            'onnx::Add_12': torch.randn(batch_size,1280,4,6,dtype=torch.float32, device=device),
            'onnx::Add_13': torch.randn(batch_size,1280,4,6,dtype=torch.float32, device=device),
            'onnx::Add_14': torch.randn(batch_size,1280,4,6,dtype=torch.float32, device=device),
            'onnx::Cast_15': torch.randn(batch_size,1280,4,6,dtype=torch.float32, device=device)
        }
    elif model_name == 'controlnet':
         return {
            'x': torch.randn(batch_size, unet_dim, latent_height, latent_width, dtype=torch.float32, device=device),
            'hint': torch.randn(batch_size, 3, image_height,image_width, dtype=torch.float32, device=device),
            'timestep': torch.tensor([1.], dtype=torch.float32, device=device),
            'context': torch.randn(batch_size, text_maxlen, embedding_dim, dtype=dtype, device=device)
         }
    elif model_name == 'clip':
        return {'input_ids': torch.zeros(batch_size, text_maxlen, dtype=torch.int32, device=device)}
    elif model_name == 'vae':
        return {'latent': torch.randn(batch_size, 4, latent_height, latent_width, dtype=torch.float32, device=device)}
    
def main(engine_dir,
        onnx_dir,
        opt_batch_size,
        opt_image_height,
        opt_image_width,
        force_build=False,
        static_batch=False,
        static_shape=True,
        enable_refit=False,
        enable_preview=False,
        enable_all_tactics=False,
        timing_cache=None,
        onnx_refit_dir=None
        ):
    device='cuda'
    models = ['controlunet','controlnet','clip','vae']
    max_workspace_size = 16 << 30
    for model_name in models:
        engine_path = getEnginePath(model_name, engine_dir)
        engine = Engine(engine_path)
        onnx_path = getOnnxPath(model_name, onnx_dir, opt=False)
        onnx_opt_path = getOnnxPath(model_name, onnx_dir)

        if force_build or not os.path.exists(engine.engine_path):
            engine.build(onnx_opt_path,
                fp16=True,
                input_profile=get_input_profile(model_name,
                    opt_batch_size, opt_image_height, opt_image_width,
                    static_batch=static_batch, static_shape=static_shape
                ),
                enable_refit=enable_refit,
                enable_preview=enable_preview,
                enable_all_tactics=enable_all_tactics,
                timing_cache=timing_cache,
                workspace_size=max_workspace_size)
        engines[model_name] = engine


    # Load and activate TensorRT engines
    max_device_memory = 8 << 30   
    for model_name in models:
        engine = engines[model_name]
        engine.load()
        max_device_memory = max(max_device_memory, 0)
        if onnx_refit_dir:
            onnx_refit_path = getOnnxPath(model_name, onnx_refit_dir)
            if os.path.exists(onnx_refit_path):
                engine.refit(onnx_opt_path, onnx_refit_path)

    shared_device_memory = cuda.DeviceArray.raw((max_device_memory,))
    for engine in engines.values():
        engine.activate(reuse_device_memory=shared_device_memory.ptr)
   
    stream = cuda.Stream()
    for model_name in models:
        engines[model_name].allocate_buffers(shape_dict=get_shape_dict(model_name, opt_batch_size, opt_image_height, opt_image_width), device=device)
        print(model_name + " allocate_buffers successful")
        feed_dict = get_sample_input_dict(model_name,opt_batch_size, opt_image_height, opt_image_width,use_fp16=True,device=device)
        res = engines[model_name].infer(feed_dict,stream,True)
        pass


if __name__ == "__main__":
    engine_dir='engine/'
    onnx_dir='onnx/'
    opt_batch_size = 1
    opt_image_height = 256
    opt_image_width =384
    main(engine_dir, onnx_dir, opt_batch_size, opt_image_height, opt_image_width)