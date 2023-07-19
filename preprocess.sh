echo "preprocess"
FILE=./engines/
if [ ! -d "$FILE" ]; then
    mkdir ./engines
else
    echo "engines dir is existed!"
fi

if [ ! -f "./models/controlunet.onnx" ];then
    python ./scripts/export.py
    python ./scripts/trt_controlunet.py
else
    python ./scripts/trt_controlunet.py
fi
