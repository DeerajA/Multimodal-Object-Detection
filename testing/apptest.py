from container.state import state
from ParentAgent import choose_agent

state: dict = {}
state["question"] = "Where is the school bus located?"
state["image_path"] = "images/image2.png"




choose_agent(state=state)

