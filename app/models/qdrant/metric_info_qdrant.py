from dataclasses import dataclass
from typing import TypedDict

@dataclass
class MetricInfoQdrant(TypedDict):
    id:str
    name:str
    description:str
    relevant_columns: list
    alias:list