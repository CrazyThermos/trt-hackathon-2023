import sys
import os
sys.path.append('./')
import torch
import numpy as np
import einops
import argparse
from cldm.model import create_model, load_state_dict
from annotator.util import resize_image, HWC3
from annotator.canny import CannyDetector
from cldm.ddim_hacked import DDIMSampler
import onnx
from onnx import shape_inference
import onnx_graphsurgeon as gs
from polygraphy.backend.onnx.loader import fold_constants

class Optimizer():
    def __init__(
        self,
        onnx_graph,
        verbose=False
    ):
        self.graph = gs.import_onnx(onnx_graph)
        self.verbose = verbose

    def info(self, prefix):
        if self.verbose:
            print(f"{prefix} .. {len(self.graph.nodes)} nodes, {len(self.graph.tensors().keys())} tensors, {len(self.graph.inputs)} inputs, {len(self.graph.outputs)} outputs")

    def cleanup(self, return_onnx=False):
        self.graph.cleanup().toposort()
        if return_onnx:
            return gs.export_onnx(self.graph)

    def select_outputs(self, keep, names=None):
        self.graph.outputs = [self.graph.outputs[o] for o in keep]
        if names:
            for i, name in enumerate(names):
                self.graph.outputs[i].name = name

    def fold_constants(self, return_onnx=False):
        onnx_graph = fold_constants(gs.export_onnx(self.graph), allow_onnxruntime_shape_inference=True)
        self.graph = gs.import_onnx(onnx_graph)
        if return_onnx:
            return onnx_graph

    def infer_shapes(self, return_onnx=False):
        onnx_graph = gs.export_onnx(self.graph)
        if onnx_graph.ByteSize() > 2147483648:
            raise TypeError("ERROR: model size exceeds supported 2GB limit")
        else:
            onnx_graph = shape_inference.infer_shapes(onnx_graph)

        self.graph = gs.import_onnx(onnx_graph)
        if return_onnx:
            return onnx_graph


def get_args():
    parser = argparse.ArgumentParser('Export Ernie TensorRT', add_help=False)
    parser.add_argument('--controlnet', action='store_true', default=False, help='Path of onnx file to load')
    parser.add_argument('--controlunet', action='store_true',default=False, help='Path of trt engine to save')
    parser.add_argument('--clip', action='store_true', default=False, help='Enable FP16 mode or not, default is TF32 if it is supported')
    parser.add_argument('--vae', action='store_true', default=False, help='Enable INT8 mode or not, default is TF32 if it is supported')
    args = parser.parse_args()
    return args

def export(args):
    EXPORT_CONTROLUNET = args.controlunet
    EXPORT_CONTROLNET = args.controlnet
    EXPORT_FROZENCLIPEMBEDDER = args.clip
    EXPORT_VAE = args.vae


    model = create_model(config_path='./models/cldm_v15.yaml').cpu()
    pretrained_weights = '/home/player/ControlNet/models/control_sd15_canny.pth'
    model.load_state_dict(load_state_dict(pretrained_weights,location='cuda'), strict=True)
    model = model.cuda()
    model.eval()    


    if EXPORT_CONTROLNET:
        x = torch.randn((1, 4, 32, 48)).float().cuda()
        hint = torch.randn((1, 3, 256,384)).float().cuda()
        timestep = torch.tensor([951]).cuda()
        context = torch.randn((1, 77, 768)).float().cuda()
        inputs = (x,hint,timestep,context)
        # ControlNet
        onnx_opt_path = './onnx/controlnet'
        with torch.inference_mode(), torch.autocast("cuda"):
            torch.onnx.export(model.control_model, 
                            inputs, 
                            onnx_opt_path+'.onnx',                                    
                            export_params=True,
                            do_constant_folding=True,
                            opset_version=17, 
                            verbose=True, 
                            input_names=['x','hint','timestep','context'], 
                            output_names=['control'],
                            dynamic_axes={'x': {0: 'B', 2: 'H', 3: 'W'},
                                        'hint': {0: 'B' },
                                        'context': {0: 'B'},
                            } ) 
        onnx_graph = onnx.load(onnx_opt_path+'.onnx', load_external_data=False)
        name = "ControlNet"
        opt = Optimizer(onnx_graph, verbose=True)
        opt.info(name + ': original')
        opt.cleanup()
        opt.info(name + ': cleanup')
        opt.fold_constants()
        opt.info(name + ': fold constants')
        opt.infer_shapes()
        opt.info(name + ': shape inference')
        onnx_opt_graph = opt.cleanup(return_onnx=True)
        opt.info(name + ': finished')
        onnx.save(onnx_opt_graph, onnx_opt_path+".opt.onnx")

    if EXPORT_CONTROLUNET:
        x = torch.randn((1, 4, 32, 48)).float().cuda()
        timestep = torch.tensor([951]).cuda()
        context = torch.randn((1, 77, 768)).float().cuda()
        control = [torch.randn(1,320,32,48).float().cuda(), 
                torch.randn(1,320,32,48).float().cuda(),
                torch.randn(1,320,32,48).float().cuda(),
                torch.randn(1,320,16,24).float().cuda(),
                torch.randn(1,640,16,24).float().cuda(),
                torch.randn(1,640,16,24).float().cuda(),
                torch.randn(1,640,8,12).float().cuda(),
                torch.randn(1,1280,8,12).float().cuda(),
                torch.randn(1,1280,8,12).float().cuda(),
                torch.randn(1,1280,4,6).float().cuda(),
                torch.randn(1,1280,4,6).float().cuda(),
                torch.randn(1,1280,4,6).float().cuda(),
                torch.randn(1,1280,4,6).float().cuda()
                ]
        only_mid_control = False
        inputs =  (x,timestep,context,control,only_mid_control)
        # ControlUNet
        onnx_opt_path = './onnx/controlunet'
        with torch.inference_mode(), torch.autocast("cuda"):
            torch.onnx.export(model.model.diffusion_model, 
                            inputs, 
                            onnx_opt_path+'.onnx', 
                            export_params=True,
                            do_constant_folding=True,
                            opset_version=17, 
                            input_names=['x','timestep','context','control'],
                            output_names=['latents'],
                            dynamic_axes={'x': {0: 'B', 2: 'H', 3: 'W'},
                                        'context': {0: 'B'},
                                        'latents': {0: 'B', 2: 'H', 3: 'W'},}
                                        ) 
        onnx_graph = onnx.load(onnx_opt_path+'.onnx', load_external_data=False)
        name = "ControlUNet"
        opt = Optimizer(onnx_graph, verbose=True)
        opt.info(name + ': original')
        opt.cleanup()
        opt.info(name + ': cleanup')
        opt.fold_constants()
        opt.info(name + ': fold constants')
        opt.infer_shapes()
        opt.info(name + ': shape inference')
        onnx_opt_graph = opt.cleanup(return_onnx=True)
        opt.info(name + ': finished')
        onnx.save(onnx_opt_graph, onnx_opt_path+".opt.onnx")

    if EXPORT_FROZENCLIPEMBEDDER:
        # text = ['longbody, lowres, bad anatomy, bad hands, missing fingers']
        # tokens = torch.tensor([[49406,   320,  3329,   267,   949,  3027,   267,  6519, 12609, 49407,
        #  49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
        #  49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
        #  49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
        #  49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
        #  49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
        #  49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
        #  49407, 49407, 49407, 49407, 49407, 49407, 49407]], dtype=torch.int32, device='cuda:0')
        tokens = torch.zeros(1, 77, dtype=torch.int32, device='cuda')
        # tokens = torch.LongTensor(tokens).cuda()
        
        output_hidden_states = False
        inputs = (tokens)
        # FrozenCLIPEmbedder
        batch_size = 1
        text_maxlen = 77
        onnx_opt_path = './onnx/clip'
        torch.onnx.export(model.cond_stage_model.transformer,
        inputs,
        onnx_opt_path+'.onnx',
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names= ['input_ids'],
        output_names=['text_embeddings', 'pooler_output'],
        dynamic_axes={
            'input_ids': {0: 'B'},
            'text_embeddings': {0: 'B'},
            'pooler_output':{0: 'B'}
        }
        )
        onnx_graph = onnx.load(onnx_opt_path+'.onnx', load_external_data=False)
        name = "CLIP"
        opt = Optimizer(onnx_graph, verbose=True)
        opt.info(name + ': original')
        opt.cleanup()
        opt.info(name + ': cleanup')
        opt.fold_constants()
        opt.info(name + ': fold constants')
        opt.infer_shapes()
        opt.info(name + ': shape inference')
        onnx_opt_graph = opt.cleanup(return_onnx=True)
        opt.info(name + ': finished')
        onnx.save(onnx_opt_graph, onnx_opt_path+".opt.onnx")

    if EXPORT_VAE:
        input0 = torch.randn((1, 4, 32, 48)).float().cuda()
        # AutoencoderKL
        onnx_opt_path = './onnx/vae'
        torch.onnx.export(model.first_stage_model.decoder, 
        input0, 
        onnx_opt_path+'.onnx',
        opset_version=18, 
        verbose=True, 
        input_names=['latent'], 
        output_names=['images'],
        dynamic_axes={
            'images': {0: 'B', 2: '8H', 3: '8W'},
            'latent': {0: 'B', 2: 'H', 3: 'W'}
        } ) 
        onnx_graph = onnx.load(onnx_opt_path+'.onnx', load_external_data=False)
        name = "VAE"
        opt = Optimizer(onnx_graph, verbose=True)
        opt.info(name + ': original')
        opt.cleanup()
        opt.info(name + ': cleanup')
        opt.fold_constants()
        opt.info(name + ': fold constants')
        opt.infer_shapes()
        opt.info(name + ': shape inference')
        onnx_opt_graph = opt.cleanup(return_onnx=True)
        opt.info(name + ': finished')
        onnx.save(onnx_opt_graph, onnx_opt_path+".opt.onnx")
    del model
    print("Exporting .pth model to onnx model has been successful!")


if __name__ == "__main__":
    args = get_args()
    export(args)