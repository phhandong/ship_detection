import onnx
from onnxsim import simplify

onnx_model = onnx.load(r"E:\Python_Project\Ship_Detection\yolo8\weights\yolov8n.transd.onnx")
model_simp, check = simplify(onnx_model)
assert check, "Simplified ONNX model could not be Validated"
onnx.save(model_simp, r"E:\Python_Project\Ship_Detection\yolo8\weights\yolov8n.transd.sim.onnx")
