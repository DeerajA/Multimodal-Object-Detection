from typing import List, Optional, TypedDict

class state:
    question: str
    image_path: str

    requiredAgents: List[str]
    completedAgents: List[str]
