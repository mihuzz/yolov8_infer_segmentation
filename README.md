# yolov8_infer_segmentation
based on yolov8 and https://github.com/fish-kong/Yolov8-instance-seg-tensorrt  
convert to onnx
from onnx convert to trt:   
/usr/local/TensorRT-8.2.5.1/bin/trtexec --onnx=/home/mih/PycharmProjects/yomy8/weights4seg/yolov8n-seg.onnx --saveEngine=/home/mih/QtProjects/yolov8_infer_segment/yolov8n-seg.engine

