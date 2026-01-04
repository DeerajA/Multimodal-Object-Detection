from langchain_core.prompts import ChatPromptTemplate
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from pydantic import BaseModel, Field
from typing import List
from langchain_core.output_parsers import PydanticOutputParser

#export NVIDIA_API_KEY="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"
NVIDIA = ChatNVIDIA(model="meta/llama-3.1-8b-instruct", temperature=0)

class template(BaseModel):
    requiredAgents: List[str] = Field(description = "Choose from [DetectionAgent, VisualizationAgent, LocationAgent, SizeAgent]")

parser = PydanticOutputParser(pydantic_object=template)



planner_prompt = ChatPromptTemplate.from_template(
   """
    You must return ONLY valid JSON.
    Do not add text, explanations, or formatting.
    Your role is to determine which agents are required to answer the question,
    Analyze the question and select all relevant agents needed to complete the task.
    The answer should always **end with "LastAgent"** as the final agent in the list.

    Available agents include (but are not limited to):
        DetectionAgent: detects all objects in the image and returns a dictionary of object names and their counts
        VisualizationAgent: returns an image showing how the YOLO model detected objects (with bounding boxes)
        LocationAgent: returns each detected object with its bounding box coordinates (x1, y1, x2, y2)
        SizeAgent: returns the total size of the image (width and height), meant to help you if they ask where something is in relation to the image or other objects

    Schema:
    {format_instructions}

    Example:
        Question: "What objects are in this image and where are they located?"
        Answer: {{"requiredAgents": ["DetectionAgent", "LocationAgent", "SizeAgent", "LastAgent"]}}

    Question:
    {question}
    """,
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

planner = planner_prompt | NVIDIA | parser

from typing import List, Optional, TypedDict, Dict, Tuple
class state(TypedDict, total=False):
    question= str
    image_path: str
    requiredAgents = List[str]
    completedAgents: List[str]

    DetectionAgent: Dict[str, int]
    LocationAgent: List[str]
    SizeAgent: Tuple[int, int]

state_instance: state = {
    "question": "What objects are in this image and where are they located?",
    "image_path": "",
    "requiredAgents": ["DetectionAgent", "LocationAgent", "SizeAgent", "LastAgent"],
    "completedAgents": [],
    "DetectionAgent": {},
    "LocationAgent": [],
    "SizeAgent": (0, 0)
}

if "requiredAgents" not in state_instance:
    result = planner.invoke({"question": state_instance["question"]})
    state_instance["requiredAgents"] = result.requiredAgents
    state_instance["completedAgents"] = []

for agent in state_instance["requiredAgents"]:
    if agent not in state_instance["completedAgents"]:
        print(agent)

