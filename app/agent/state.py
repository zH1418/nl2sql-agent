# 列信息封装实体
from typing import TypedDict

from app.models.es.value_info_es import ValueInfoEs
from app.models.qdrant.column_info_qdrant import ColumnInfoQdrant
from app.models.qdrant.metric_info_qdrant import MetricInfoQdrant


class ColumnInfoState(TypedDict):
    """列信息封装实体"""
    name: str                # 列名
    type: str                # 数据类型（如 string, int, date 等）
    role: str                # 列角色（如维度、度量、时间等）
    examples: list           # 示例值列表
    description: str         # 列描述信息
    alias: list[str]         # 列的别名列表


class TableInfoState(TypedDict):
    """表信息封装实体"""
    name: str                # 表名
    role: str                # 表角色（如事实表、维度表等）
    description: str         # 表描述信息
    columns: list[ColumnInfoState]  # 该表包含的列信息列表


class MetricInfoState(TypedDict):
    """指标信息封装实体"""
    name: str                # 指标名称
    description: str         # 指标描述
    relevant_columns: list[str]  # 该指标相关的列名列表
    alias: list[str]         # 指标的别名列表


class DateInfoState(TypedDict):
    """日期信息实体"""
    date: str                # 日期字符串（如 "2026-07-04"）
    weekday: str             # 星期几（如 "Saturday"）
    quarter: str             # 季度（如 "Q3"）


class DBInfoState(TypedDict):
    """数据库信息实体"""
    version: str             # 数据库版本号
    dialect: str             # SQL 方言（如 "mysql", "postgresql"）
