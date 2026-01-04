from ultralytics import YOLO
from container.state import state


MODEL_NAME = "yolo11n.pt"
model = YOLO(MODEL_NAME)


def getLocations(state):
    img_path = state['image_path']
    results = model(img_path)
    boxes = results[0].boxes.xyxy
    classes = results[0].boxes.cls
    names = results[0].names

    locations = []
    for box, cls in zip(boxes, classes):
        x1, y1, x2, y2 = box
        locations.append(f"Name: {names[int(cls)]}, Location: {x1.item(), y1.item(), x2.item(), y2.item()}")
    prompt = "This is all the objects on screen and then their x1,y1,x2,y2 " \
    "locations on the image, where (0,0) is top-left and as x gets higher it " \
    "moves right and as y gets higher it moves down. use that logic when computing " \
    "where something is in relation to something else"
    
    locations = f"{prompt} {locations}"
    return {
            "LocationAgent": locations,
            "completedAgents": state['completedAgents'] + ["LocationAgent"]
        }