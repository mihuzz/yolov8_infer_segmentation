# yolov8_infer_segmentation
based on yolov8 https://github.com/ultralytics/ultralytics and https://github.com/fish-kong/Yolov8-instance-seg-tensorrt  
convert yolov8n-seg.pt to onnx:  
```python
from ultralytics import YOLO

# Load a model
# model = YOLO("yolov8n-seg.pt")  # load an official model
model = YOLO("/path/to/saved/weights/weights4seg/yolov8n-seg.pt")  # load a custom trained

# Export the model
model.export(format="engine")

```
from onnx convert to trt:   
```
/usr/local/TensorRT-8.2.5.1/bin/trtexec --onnx=/way/to/your/yolov8n-seg.onnx --saveEngine=/path/to/save/yolov8n-seg.engine  
```
mkdir build && cd build  
cmake ..  
make 
```
./yolov8_infer_segment --model=/path/to/saved/model/yolov8n-seg.engine --input=/path/to/video  
```
```
./yolov8_infer_segment --model=/path/to/saved/model/yolov8n-seg.engine --device=0  
```
![yolov8seg](https://user-images.githubusercontent.com/45326995/216149764-17ddf7e3-1f1e-48ee-aeb5-74ea5e31f2d4.gif)
