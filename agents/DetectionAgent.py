from ultralytics import YOLO
from container.state import state


MODEL_NAME = "yolo11n.pt"
model = YOLO(MODEL_NAME)


def getCount(state):
    img_path = state['image_path']
    results = model(img_path, show=False, verbose=False)

    classes = results[0].boxes.cls
    names = results[0].names

    counts = {}
    for cls in classes:
        name = names[int(cls)]
        counts[name] = counts.get(name, 0) + 1
    counts = f"This what objects were detected and how many of them there are: {counts}"
    
    return {
            "DetectionAgent": counts,
            "completedAgents": state['completedAgents'] + ["DetectionAgent"]
        }
