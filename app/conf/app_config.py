from dataclasses import dataclass
from pathlib import Path
from omegaconf import OmegaConf


# ====== 日志配置 ======
@dataclass
class File:
    enable: bool
    level: str
    path: str
    rotation: str
    retention: str


@dataclass
class Console:
    enable: bool
    level: str


@dataclass
class LoggingConfig:
    file: File
    console: Console


# ====== 数据库配置 ======
@dataclass
class DBConfig:
    host: str
    port: int
    user: str
    password: str
    database: str


# ====== Qdrant 向量库 ======
@dataclass
class QdrantConfig:
    host: str
    port: int
    embedding_size: int


# ====== Embedding 服务 ======
@dataclass
class EmbeddingConfig:
    # mode: local=本地 ONNX 进程内编码；endpoint=远程 TEI 服务
    mode: str = "endpoint"
    model_dir: str = ""   # local 模式：ONNX 产物目录（绝对路径）
    dim: int = 512        # local 模式：向量维度
    host: str = ""
    port: int = 8081
    model: str = ""


# ====== ES 全文检索 ======
@dataclass
class ESConfig:
    host: str
    port: int
    index_name: str


# ====== LLM ======
@dataclass
class LLMConfig:
    model_name: str
    api_key: str
    base_url: str


# ====== 顶层聚合 ======
@dataclass
class AppConfig:
    logging: LoggingConfig
    db_meta: DBConfig
    db_dw: DBConfig
    qdrant: QdrantConfig
    embedding: EmbeddingConfig
    es: ESConfig
    llm: LLMConfig


# ====== 加载配置 ======
config_file = Path(__file__).parents[2] / 'conf' / 'app_config.yaml'
context = OmegaConf.load(config_file)
schema = OmegaConf.structured(AppConfig)
app_config: AppConfig = OmegaConf.to_object(OmegaConf.merge(schema, context))
