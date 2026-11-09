import dataclasses
from typing import Type, TypeVar, Set, Any, get_args, get_origin
from sqlalchemy.orm import DeclarativeBase

S = TypeVar("S")  # 源对象（SQLAlchemy Model / dataclass / 普通对象）
D = TypeVar("D")  # 目标类（SQLAlchemy Model / dataclass）


class EntityMapperUtil:

    @staticmethod
    def _get_target_field_names(target_cls: Type[Any]) -> set[str]:
        """内部工具：自动识别目标类型，提取所有字段名集合"""
        if issubclass(target_cls, DeclarativeBase):
            return {col.key for col in target_cls.__mapper__.columns}
        if dataclasses.is_dataclass(target_cls):
            return {f.name for f in dataclasses.fields(target_cls)}
        raise TypeError(
            f"不支持的目标类型 {target_cls.__name__}，仅支持 SQLAlchemy DeclarativeBase Model / dataclass"
        )

    @staticmethod
    def map_fields(source: S, target_cls: Type[D], exclude: Set[str] = None) -> D:
        exclude = exclude or set()
        target_all_fields = EntityMapperUtil._get_target_field_names(target_cls)
        target_need_fields = target_all_fields - exclude

        init_kwargs = {}
        for field_name in target_need_fields:
            if hasattr(source, field_name):
                init_kwargs[field_name] = getattr(source, field_name)

        return target_cls(**init_kwargs)