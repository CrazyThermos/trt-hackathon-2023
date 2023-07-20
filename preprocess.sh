echo "preprocess"
FILE=./engines/
if [ ! -d "$FILE" ]; then
    mkdir ./engines
else
    echo "engines dir is existed!"
fi

if [ ! -f "./models/controlunet.onnx" ];then
    python3 ./scripts/export.py
    python3 ./scripts/trt_controlunet.py
else
    python3 ./scripts/trt_controlunet.py
fi
