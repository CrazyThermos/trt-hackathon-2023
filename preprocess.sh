echo "preprocess"
ONNX_DIR=./onnx/
ENGINE_DIR=./engine/

if [ ! -d "$ONNX_DIR" ]; then
    mkdir ./onnx
else
    echo "onnx dir is existed!"
fi

if [ ! -d "$ENGINE_DIR" ]; then
    mkdir ./engine
else
    echo "engine dir is existed!"
fi

if [ ! -f "./onnx/controlunet.onnx" -o ! -f "./onnx/controlnet.onnx" -o ! -f "./onnx/vae.onnx" -o ! -f "./onnx/clip.onnx" ];then
    python3 ./scripts/export.py --controlnet --controlunet --vae --clip
    python3 ./scripts/onnx2trt.py 
else
    python3 ./scripts/onnx2trt.py 
fi
