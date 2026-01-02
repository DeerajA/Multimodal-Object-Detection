from langchain_core.prompts import ChatPromptTemplate
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from pydantic import BaseModel, Field
from typing import List
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import HumanMessage

#export NVIDIA_API_KEY="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"
NVIDIA = ChatNVIDIA(model="deepseek-ai/deepseek-r1-0528")

class template(BaseModel):
    POV: List[str] = Field(description = "Choose from [First, Second, Third]")

parser = PydanticOutputParser(pydantic_object=template)



planner_prompt = ChatPromptTemplate.from_template(
    """
    Your role is to determine the point of view of the question.
    e.g., first person, second person, or third person

    {format_instructions}
    
    Example:
    Question: "I like the color red"  
    Answer: {{"POV": ["First"]}}
    
    {question}
    """,
    partial_variables={"format_instructions": parser.get_format_instructions()},
)




prompt_text = planner_prompt.format(
    question="he saw a bird"
)

messages = [HumanMessage(content=prompt_text)]
llm_output = NVIDIA.generate([messages])
print(llm_output)