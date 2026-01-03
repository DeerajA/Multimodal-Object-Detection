from ultralytics import YOLO
from container.state import state


MODEL_NAME = "yolo11n.pt"
model = YOLO(MODEL_NAME)


def getLocations(img_path):
    state['completedAgents'].append("LocationAgent")

    results = model(img_path)
    boxes = results[0].boxes.xyxy
    classes = results[0].boxes.cls
    names = results[0].names

    locations = []
    for box, cls in zip(boxes, classes):
        x1, y1, x2, y2 = box
        locations.append(f"Name: {names[int(cls)]}, Location: {x1.item(), y1.item(), x2.item(), y2.item()}")
    return locations