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


def get_args():
    parser = argparse.ArgumentParser('Export Ernie TensorRT', add_help=False)
    parser.add_argument('--controlnet', action='store_true', default=False, help='Path of onnx file to load')
    parser.add_argument('--controlunet', action='store_true',default=False, help='Path of trt engine to save')
    parser.add_argument('--clip', action='store_true', default=False, help='Enable FP16 mode or not, default is TF32 if it is supported')
    parser.add_argument('--vae', action='store_true', default=False, help='Enable INT8 mode or not, default is TF32 if it is supported')
    args = parser.parse_args()
    return args

def export(args):
    EXPORT_CONTROLUNET = args.controlnet
    EXPORT_CONTROLNET = args.controlunet
    EXPORT_FROZENCLIPEMBEDDER = args.clip
    EXPORT_VAE = args.vae


    model = create_model(config_path='./models/cldm_v15.yaml').cpu()
    pretrained_weights = '/home/player/ControlNet/models/control_sd15_canny.pth'
    model.load_state_dict(load_state_dict(pretrained_weights,location='cuda'), strict=True)
    model = model.cuda()
    model.eval()    


    if EXPORT_CONTROLNET:
        input0 = torch.randn((1, 4, 32, 48)).float().cuda()
        input1 = torch.randn((1, 3, 256,384)).float().cuda()
        input2 = torch.tensor([951]).cuda()
        input3 = torch.randn((1, 77, 768)).float().cuda()
        inputs = (input0,input1,input2,input3)
        # ControlNet
        torch.onnx.export(model.control_model, inputs, './models/controlnet.onnx', opset_version=18, verbose=True, input_names=['input0','input1','input2','input3'], output_names=['output0'] ) 

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
        torch.onnx.export(model.model.diffusion_model, inputs, './models/controlunet.onnx', opset_version=18, input_names=['input0','input1','input2','input3','input4'], output_names=['output0'] ) 

    if EXPORT_FROZENCLIPEMBEDDER:
        # text = ['longbody, lowres, bad anatomy, bad hands, missing fingers']
        tokens = torch.tensor([[49406,   320,  3329,   267,   949,  3027,   267,  6519, 12609, 49407,
         49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
         49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
         49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
         49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
         49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
         49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
         49407, 49407, 49407, 49407, 49407, 49407, 49407]], device='cuda:0')
        # tokens = torch.LongTensor(tokens).cuda()
        
        output_hidden_states = False
        inputs = (tokens)
        # FrozenCLIPEmbedder
        torch.onnx.export(model.cond_stage_model.transformer, inputs, './models/FrozenCLIPEmbedder.onnx', opset_version=18, verbose=True, input_names=['input0'], output_names=['output0'] ) 

    if EXPORT_VAE:
        input0 = torch.randn((1, 4, 32, 48)).float().cuda()
        # AutoencoderKL
        torch.onnx.export(model.first_stage_model.decoder, input0, './models/AutoencoderKL.onnx', opset_version=18, verbose=True, input_names=['input0'], output_names=['output0'] ) #指定模型的输入，以及onnx的输出路径

    print("Exporting .pth model to onnx model has been successful!")


if __name__ == "__main__":
    args = get_args()
    export(args)