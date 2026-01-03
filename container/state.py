from typing import List, Optional, TypedDict, Dict, Tuple

class state:
    question: str
    image_path: str

    requiredAgents: List[str]
    completedAgents: List[str]

    DetectionAgent: Dict[str, int]
    LocationAgent: List[str]
    SizeAgent: Tuple[int, int]