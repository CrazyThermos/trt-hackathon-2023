echo "preprocess"
if [ -f "./models/control_sd15_canny.pth" ];then
    echo "control_sd15_canny.pth exist!"
else
    echo "can't found control_sd15_canny.pth! start downloading..."
    wget -P ./models/ https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_canny.pth
fi