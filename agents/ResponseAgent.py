from container.state import state
from langchain_core.prompts import ChatPromptTemplate
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

#export NVIDIA_API_KEY="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"
NVIDIA = ChatNVIDIA(model="meta/llama-3.1-8b-instruct", temperature=0)

prompt = ChatPromptTemplate.from_template(
    """
    You are given information collected by multiple agents in the variable called `final`.

    Your task:
        - Use ONLY the information contained in `final`
        - Answer the user's question accurately and directly
        - Do NOT invent, assume, or infer anything not present in `final`
        - If the answer cannot be determined from `final`, say so clearly
        - Talk in a short conversational way, do not talk for too long unless it is specifically asked of you to provide details 
        - Don't give coordinates for location based questions use the total size of the image to calculate where the object is in relation to everything else
        - Understand that the locationAgent returns (x1, y1, x2, y2), where (0,0) is top-left, same for the sizeAgent
    Question:
    {question}

    Information:
    {final}

    Answer:
    """
)


def LastAgent(state):
    final = ""
    for s in state['completedAgents']:
        if data := state.get(s):
            final += str(data) + "\n"

    model = prompt | NVIDIA

    results = model.invoke({"question":state['question'],
                            "final": final})
    return {
            "response": results.content,
        }
