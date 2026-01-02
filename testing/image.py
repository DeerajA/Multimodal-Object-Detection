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

results = model(img1_path)

boxes = results[0].boxes.xyxy
classes = results[0].boxes.cls
names = results[0].names

for box, cls in zip(boxes, classes):
    x1, y1, x2, y2 = box
    print(f"Name: {names[int(cls)]}, Location: {x1.item(), y1.item(), x2.item(), y2.item()}")
    
height, width = results[0].orig_shape
print(width, height)

counts = {}
for cls in classes:
    name = names[int(cls)]
    counts[name] = counts.get(name, 0) + 1

print(counts)
"""
# Plot and log the results

fig, ax = plt.subplots(figsize=(12, 8))
ax.imshow(cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB))
ax.axis("off")


# Upload the image to Neptune
run["predictions/sample"].upload(neptune.types.File.as_image(fig))

plt.show()
"""



run.stop()