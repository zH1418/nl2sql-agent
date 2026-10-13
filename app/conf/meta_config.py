from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from omegaconf import OmegaConf


@dataclass
class ColumnConfig:
    name: str
    role: str
    description: str
    alias: list[str]
    sync: bool


@dataclass
class TableConfig:
    name: str
    role: str
    description: str
    columns: list[ColumnConfig]


@dataclass
class MetricConfig:
    name: str
    description: str
    relevant_columns: list[str]
    alias: list[str]


@dataclass
class MetaConfig:
    tables: list[TableConfig]
    metrics: list[MetricConfig]
