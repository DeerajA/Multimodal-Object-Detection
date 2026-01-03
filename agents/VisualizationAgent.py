import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
from container.state import state

MODEL_NAME = "yolo11n.pt"
model = YOLO(MODEL_NAME)


def show_plot(img_path):
    state.completedAgents.append("VisualizationAgent")

    results = model(img_path)
    # Plot and log the results
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB))
    ax.axis("off")
    plt.show()