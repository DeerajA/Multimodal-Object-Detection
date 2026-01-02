import os
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO


MODEL_NAME = "yolo11n.pt"
model = YOLO(MODEL_NAME)


def getCount(img_path):
    results = model(img_path)

    classes = results[0].boxes.cls
    names = results[0].names

    counts = {}
    for cls in classes:
        name = names[int(cls)]
        counts[name] = counts.get(name, 0) + 1

    return counts