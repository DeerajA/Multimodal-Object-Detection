from ultralytics import YOLO
from container.state import state

MODEL_NAME = "yolo11n.pt"
model = YOLO(MODEL_NAME)

def getSize(img_path):
    state['completedAgents'].append("SizeAgent")

    results = model(img_path)
    height, width = results[0].orig_shape
    return(width, height)
