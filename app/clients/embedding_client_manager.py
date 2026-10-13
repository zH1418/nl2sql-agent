import time
from typing import Optional

import requests
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from loguru import logger

from app.conf.app_config import EmbeddingConfig, app_config


class EmbeddedClientManager:
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.client: Optional[HuggingFaceEndpointEmbeddings] = None

    def _get_url(self):
        return f"http://{self.config.host}:{self.config.port}"

    def warmup(self, max_retries: int = 10, retry_interval: float = 3.0) -> bool:
        """预热 TEI embedding 服务：虚拟机启动后需要首次请求来加载模型到内存。"""
        health_url = f"{self._get_url()}/health"
        logger.info(f"正在预热 embedding 服务: {health_url}")

        for i in range(max_retries):
            try:
                resp = requests.get(health_url, timeout=5)
                if resp.status_code == 200:
                    logger.info(f"embedding 服务预热成功 (尝试 {i + 1}/{max_retries})")
                    return True
                logger.warning(f"embedding 服务返回异常状态码: {resp.status_code}, 重试 {i + 1}/{max_retries}")
            except requests.RequestException as e:
                logger.warning(f"embedding 服务未就绪: {e}, 重试 {i + 1}/{max_retries}")

            if i < max_retries - 1:
                time.sleep(retry_interval)

        logger.error(f"embedding 服务预热失败，已重试 {max_retries} 次")
        return False

    def init(self):
        """初始化 embedding 客户端。

        - mode=local    ：本地 ONNX 进程内编码（无需 TEI 服务，无需联网）
        - mode=endpoint ：远程 TEI 服务（先预热再建客户端）
        """
        if getattr(self.config, "mode", "endpoint") == "local":
            from app.clients.local_embedding import LocalOnnxEmbeddings

            self.client = LocalOnnxEmbeddings(
                self.config.model_dir, dim=self.config.dim)
            logger.info(
                f"本地 ONNX embedding 就绪: dir={self.config.model_dir} "
                f"dim={self.config.dim} mock={self.client.is_mock} "
                f"err={self.client.load_error}"
            )
            return

        self.warmup()
        self.client = HuggingFaceEndpointEmbeddings(model=self._get_url())


embedding_client_manager = EmbeddedClientManager(app_config.embedding)

if __name__ == "__main__":
    manager = EmbeddedClientManager(app_config.embedding)
    manager.init()
    query = manager.client.embed_query("hello world")
    print(query)
    print(len(query))