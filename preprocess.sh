echo "preprocess"
FILE=./engines/
if [ ! -d "$FILE" ]; then
    mkdir ./engines
else
    echo "engines dir is existed!"
fi

if [ ! -f "./models/controlunet.onnx" ];then
    python3 ./scripts/export.py
    python3 ./scripts/trt_controlnet.py --fp16
    python3 ./scripts/trt_controlunet.py --fp16

else
    python3 ./scripts/trt_controlnet.py --fp16
    python3 ./scripts/trt_controlunet.py --fp16
fi
