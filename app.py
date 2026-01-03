from container.state import state
from ParentAgent import choose_agent

from agents.DectectionAgent import getCount
from agents.LocationAgent import getLocations
from agents.SizeAgent import getSize
from agents.VisualizationAgent import show_plot
from agents.ResponseAgent import LastAgent

from langgraph.graph import StateGraph

state: dict = {}
state["question"] = "How many people are in the image?"
state["image_path"] = "images/image2.png"

graph = StateGraph(state)

graph.add_node("choose_agent", choose_agent)

graph.add_node("DectectionAgent", getCount)
graph.add_node("LocationAgent", getLocations)
graph.add_node("SizeAgent", getSize)
graph.add_node("VisualizationAgent", show_plot)
graph.add_node("LastAgent", LastAgent)


graph.set_entry_point("choose_agent")

def get_next(state):
    print(state['requiredAgents'])


graph.add_conditional_edges(
    "choose_agent",
    lambda next: next,
    {
        "DectectionAgent":"DectectionAgent",
        "LocationAgent":"LocationAgent",
        "SizeAgent":"SizeAgent",
        "VisualizationAgent":"VisualizationAgent",
        "LastAgent":"LastAgent"
    }
)

graph.add_edge("DectectionAgent", "choose_agent")
graph.add_edge("LocationAgent", "choose_agent")
graph.add_edge("SizeAgent", "choose_agent")
graph.add_edge("VisualizationAgent", "choose_agent")
graph.add_edge("LastAgent", "__end__")

