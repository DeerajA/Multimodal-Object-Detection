from ultralytics import YOLO
from container.state import state


MODEL_NAME = "yolo11n.pt"
model = YOLO(MODEL_NAME)


def getCount(img_path):
    state['completedAgents'].append("DetectionAgent")

    results = model(img_path)

    classes = results[0].boxes.cls
    names = results[0].names

    counts = {}
    for cls in classes:
        name = names[int(cls)]
        counts[name] = counts.get(name, 0) + 1

    return counts