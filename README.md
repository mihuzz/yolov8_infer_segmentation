# yolov8_infer_segmentation
based on yolov8 and https://github.com/fish-kong/Yolov8-instance-seg-tensorrt  
convert yolov8n-seg.pt to onnx  
from onnx convert to trt:   
/usr/local/TensorRT-8.2.5.1/bin/trtexec --onnx=/way/to/your/yolov8n-seg.onnx --saveEngine=/path/to/save/yolov8n-seg.engine
mkdir build && cd build  
cmake ..  
make  
./yolov8_infer_segment --model=/path/to/saved/model/yolov8n-seg.engine --input=/path/to/video  
./yolov8_infer_segment --model=/path/to/saved/model/yolov8n-seg.engine --device=0 (webcam)  
![yolov8seg](https://user-images.githubusercontent.com/45326995/216149764-17ddf7e3-1f1e-48ee-aeb5-74ea5e31f2d4.gif)
