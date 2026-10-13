import dataclasses
from typing import Type, TypeVar, Set, Any, get_args, get_origin
from sqlalchemy.orm import DeclarativeBase

S = TypeVar("S")  # 源对象（SQLAlchemy Model / dataclass / 普通对象）
D = TypeVar("D")  # 目标类（SQLAlchemy Model / dataclass）


class EntityMapperUtil:
    """
    通用实体字段映射工具
    同时兼容：SQLAlchemy ORM Model、标准 dataclass、普通自定义对象
    自动拷贝同名字段，支持排除指定字段，替代手写逐字段赋值样板代码
    """

    @staticmethod
    def _get_target_field_names(target_cls: Type[Any]) -> set[str]:
        """内部工具：自动识别目标类型，提取所有字段名集合"""
        # 场景1：SQLAlchemy 模型
        if issubclass(target_cls, DeclarativeBase):
            return {col.key for col in target_cls.__mapper__.columns}
        # 场景2：标准 dataclass
        if dataclasses.is_dataclass(target_cls):
            return {f.name for f in dataclasses.fields(target_cls)}
        # 场景3：普通自定义类（无内置字段反射，无法自动解析，强制抛提示）
        raise TypeError(
            f"不支持的目标类型 {target_cls.__name__}，仅支持 SQLAlchemy DeclarativeBase Model / dataclass"
        )

    @staticmethod
    def map_fields(source: S, target_cls: Type[D], exclude: Set[str] = None) -> D:
        """
        将源对象同名字段映射生成目标实例
        :param source: 源实体（SQLAlchemy Model / dataclass / 普通对象）
        :param target_cls: 目标实体类（SQLAlchemy Model / dataclass）
        :param exclude: 需要排除的字段名集合
        :return: 填充完成的目标实体实例
        """
        exclude = exclude or set()
        # 获取目标类全部合法字段
        target_all_fields = EntityMapperUtil._get_target_field_names(target_cls)
        target_need_fields = target_all_fields - exclude

        # 遍历目标字段，从source中取值，仅保留source存在的属性
        init_kwargs = {}
        for field_name in target_need_fields:
            if hasattr(source, field_name):
                init_kwargs[field_name] = getattr(source, field_name)

        return target_cls(**init_kwargs)