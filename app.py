from state import state
from ParentAgent import get_agents


state: dict = {}
state["question"] = "How many people are in the image?"
state["image_path"] = "images/image2.png"
result = get_agents(state)

print(result)