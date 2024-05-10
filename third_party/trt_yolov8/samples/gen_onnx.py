from ultralytics import YOLO

'''
NOTE: trt_yolov8 do not need ONNX format at all, this script just used for visualizing yolov8 networks on `https://netron.app`. 
'''

# Load a model
model = YOLO("../../../../vp_data/models/trt/others/yolov8n-seg.pt")  # load a pretrained model (recommended for training)

# Export the model
path = model.export(format="onnx")  # export the model to ONNX format which could be visualized on netron.app