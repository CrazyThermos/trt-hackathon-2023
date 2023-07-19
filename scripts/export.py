import sys
import os
sys.path.append('./')
import torch
import einops
from cldm.model import create_model, load_state_dict
from annotator.util import resize_image, HWC3
from annotator.canny import CannyDetector
from cldm.ddim_hacked import DDIMSampler

EXPORT_CONTROLUNET = True
EXPORT_CONTROLNET = False
EXPORT_FROZENCLIPEMBEDDER = False
EXPORT_VAE = False


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
    torch.onnx.export(model.control_model, inputs, './models/controlnet.onnx', verbose=True, input_names=['input0','input1','input2','input3'], output_names=['output0'] ) 

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
    torch.onnx.export(model.model.diffusion_model, inputs, './models/controlunet.onnx',  input_names=['input0','input1','input2','input3','input4'], output_names=['output0'] ) 

if EXPORT_FROZENCLIPEMBEDDER:
    text = ['longbody, lowres, bad anatomy, bad hands, missing fingers']
    # FrozenCLIPEmbedder
    torch.onnx.export(model.cond_stage_model, text, './models/FrozenCLIPEmbedder.onnx', verbose=True, input_names=['input0'], output_names=['output0'] ) 

if EXPORT_VAE:
    input0 = torch.randn((1, 3, 256,384)).float().cuda()
    # AutoencoderKL
    torch.onnx.export(model.first_stage_model, input0, './models/AutoencoderKL.onnx', verbose=True, input_names=['input0'], output_names=['output0'] ) #指定模型的输入，以及onnx的输出路径

print("Exporting .pth model to onnx model has been successful!")
