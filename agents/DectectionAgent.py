import os
import cv2
import matplotlib.pyplot as plt
import neptune
from ultralytics import YOLO

def init_run(tags=None):
   run = neptune.init_run(
       project="deeraj/Yolo11",
       api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI1ODhjNWQ5OC00ZDY2LTQwZTktYTQyMS01ZjY2YTg4MDk4ZjcifQ==",
   )
   return run

MODEL_NAME = "yolo11n.pt"
model = YOLO(MODEL_NAME)

# Logging
run = init_run(['yolo-detection'])
run["model/task"] = model.task
run["model/name"] = MODEL_NAME

#------------------------IMAGE------------------------------------------------------------------------|
img1_path = "images/image2.png"

def get_count(img1_path):
    results = model(img1_path)

    classes = results[0].boxes.cls
    names = results[0].names

    counts = {}
    for cls in classes:
        name = names[int(cls)]
        counts[name] = counts.get(name, 0) + 1

    return counts