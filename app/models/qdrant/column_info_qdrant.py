from dataclasses import dataclass
from typing import TypedDict

@dataclass
class ColumnInfoQdrant(TypedDict):
    id: str
    name: str
    type: str
    role: str
    examples: list
    description: str
    alias: list
    table_id: str