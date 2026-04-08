from pydantic import BaseModel
from typing import List


# -----------------------------
# Region Model
# -----------------------------
class Region(BaseModel):
    name: str
    severity: float   # 0.0 to 1.0
    population: int
    helped: bool = False


# -----------------------------
# Observation (what agent sees)
# -----------------------------
class Observation(BaseModel):
    regions: List[Region]
    ambulances: int
    reward: float = 0.0
    done: bool = False
    score: float = 0.0        # top-level score for deterministic grader
    metadata: dict = {}


# -----------------------------
# Action (what agent does)
# -----------------------------
class Action(BaseModel):
    action_type: str   # "allocate"
    region_name: str