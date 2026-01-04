import os   
import asyncio

from container.state import state
from ParentAgent import choose_agent
from ParentAgent import set_agents

from agents.DetectionAgent import getCount
from agents.LocationAgent import getLocations
from agents.SizeAgent import getSize
from agents.VisualizationAgent import show_plot
from agents.ResponseAgent import LastAgent

from langgraph.graph import StateGraph

graph = StateGraph(state)

graph.add_node("set_agents", set_agents)
graph.add_node("choose_agent", choose_agent)

graph.add_node("DetectionAgent", getCount)
graph.add_node("LocationAgent", getLocations)
graph.add_node("SizeAgent", getSize)
graph.add_node("VisualizationAgent", show_plot)
graph.add_node("LastAgent", LastAgent)

graph.set_entry_point("set_agents")

graph.add_edge("set_agents", "choose_agent")

graph.add_conditional_edges(
    "choose_agent",
    lambda state: state['nextAgent'],
    {
        "DetectionAgent":"DetectionAgent",
        "LocationAgent":"LocationAgent",
        "SizeAgent":"SizeAgent",
        "VisualizationAgent":"VisualizationAgent",
        "LastAgent":"LastAgent"
    }
)

graph.add_edge("DetectionAgent", "choose_agent")
graph.add_edge("LocationAgent", "choose_agent")
graph.add_edge("SizeAgent", "choose_agent")
graph.add_edge("VisualizationAgent", "choose_agent")
graph.add_edge("LastAgent", "__end__")

app = graph.compile()


async def run():
    inputs = {
        "question": "How many cars are there?",
        "image_path": os.path.abspath("images/image1.png")
    }
    async for s in app.astream(inputs, stream_mode="values"):
        if 'response' in s and s['response']:
            print(f'Final response: {s['response']}')


if __name__ == "__main__":
    asyncio.run(run())

