from container.state import state
from ParentAgent import get_agents_required

from agents.DectectionAgent import getCount
from agents.LocationAgent import getLocations
from agents.SizeAgent import getSize
from agents.VisualizationAgent import show_plot

from langgraph.graph import StateGraph

state: dict = {}
state["question"] = "How many people are in the image?"
state["image_path"] = "images/image2.png"

graph = StateGraph(state)

graph.add_node("get_agents_required", get_agents_required)

graph.add_node("getCount", getCount)
graph.add_node("getLocations", getLocations)
graph.add_node("getSize", getSize)
graph.add_node("show_plot", show_plot)

graph.set_entry_point("get_agents_required")

def get_next(state):
    print(state['requiredAgents'])


graph.add_conditional_edges(
    "get_agents_required",
    get_next,
    {
        "getCount":"getCount",
        "getLocations":"getLocations",
        "getSize":"getSize",
        "show_plot":"show_plot"
    }
)

