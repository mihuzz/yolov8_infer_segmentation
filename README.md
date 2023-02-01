# yolov8_infer_segmentation
based on yolov8 and https://github.com/fish-kong/Yolov8-instance-seg-tensorrt  
convert to onnx  
from onnx convert to trt:   
/usr/local/TensorRT-8.2.5.1/bin/trtexec --onnx=/way/to/your/yolov8n-seg.onnx --saveEngine=/path/to/save/yolov8n-seg.engine

