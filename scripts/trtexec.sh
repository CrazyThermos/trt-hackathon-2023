trtexec --loadEngine=./engines/controlunet.trt \
--shapes=input0:1x4x32x48,\
input1:1,\
input2:1x77x768,\
input3:1x320x32x48,\
input4:1x320x32x48,\
onnx::Add_5:1x320x32x48,\
onnx::Add_6:1x320x16x24,\
onnx::Add_7:1x640x16x24,\
onnx::Add_8:1x640x16x24,\
onnx::Add_9:1x640x8x12,\
onnx::Add_10:1x1280x8x12,\
onnx::Add_11:1x1280x8x12,\
onnx::Add_12:1x1280x4x6,\
onnx::Add_13:1x1280x4x6,\
onnx::Add_14:1x1280x4x6,\
onnx::Add_15:1x1280x4x6 \
--fp16

# trtexec --loadEngine=./engines/controlnet.trt \
# --shapes=input0:1x4x32x48,\
# input1:1x3x256x384,\
# input2:1,\
# input3:1x77x768 \
# --fp16 \
# --useCudaGraph 

# input0 = np.random.randn(1, 4, 32, 48).astype(np.float32)
# input1 = np.array([951]).astype(np.float32)
# input2 = np.random.randn(1, 77, 768).astype(np.float32)
# input3 = np.random.randn(1,320,32,48).astype(np.float32)
# input4 = np.random.randn(1,320,32,48).astype(np.float32)
# input5 = np.random.randn(1,320,32,48).astype(np.float32)
# input6 = np.random.randn(1,320,16,24).astype(np.float32)
# input7 = np.random.randn(1,640,16,24).astype(np.float32)
# input8 = np.random.randn(1,640,16,24).astype(np.float32)
# input9 = np.random.randn(1,640,8,12).astype(np.float32)
# input10 = np.random.randn(1,1280,8,12).astype(np.float32)
# input11 = np.random.randn(1,1280,8,12).astype(np.float32)
# input12 = np.random.randn(1,1280,4,6).astype(np.float32)
# input13 = np.random.randn(1,1280,4,6).astype(np.float32)
# input14 = np.random.randn(1,1280,4,6).astype(np.float32)
# input15 = np.random.randn(1,1280,4,6).astype(np.float32)