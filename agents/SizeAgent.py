from ultralytics import YOLO
from container.state import state

MODEL_NAME = "yolo11n.pt"
model = YOLO(MODEL_NAME)

def getSize(state):
    img_path = state['image_path']
    results = model(img_path)
    height, width = results[0].orig_shape

    final = (f"This is the size of the whole image, (height, width): {height, width}")
    return {
            "SizeAgent": final,
            "completedAgents": state['completedAgents'] + ["SizeAgent"]
        }