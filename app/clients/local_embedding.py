"""本地 ONNX 向量编码器 —— `HuggingFaceEndpointEmbeddings` 的本地替代。

口径对齐 `customer-service-backend/agent/small_model.py`（读声明得出，非猜）：
    ONNX 后端（onnxruntime + tokenizers）→ `last_hidden_state[:, 0, :]`（cls pooling）
    → L2 归一化，max_seq_len=512。

对外接口与 `HuggingFaceEndpointEmbeddings` 兼容（本项目只用到）：
    await aembed_documents(texts) -> list[list[float]]
    await aembed_query(text)      -> list[float]
    embed_query(text)             -> list[float]

★ mock 兜底：onnxruntime / 模型产物缺失时退化为**确定性伪向量**，
  让链路在「无模型」环境也能跑通（仅用于链路验证，召回质量无意义）。
"""
from __future__ import annotations

import asyncio
import hashlib
import math
import threading
from pathlib import Path
from typing import List, Sequence

_MAX_SEQ_LEN = 512
_ONNX_FILE = "model.onnx"
_TOKENIZER_FILE = "tokenizer.json"
_OUTPUT_NAME = "last_hidden_state"


class LocalOnnxEmbeddings:
    def __init__(self, model_dir: str, dim: int = 512, mock_fallback: bool = True):
        self.model_dir = Path(model_dir)
        self.dim = dim
        self.mock_fallback = mock_fallback
        self._session = None
        self._tokenizer = None
        self._lock = threading.Lock()
        self.load_error: str | None = None
        self.is_mock = False

    # ---------- 加载 ----------
    def _ensure_loaded(self) -> None:
        if self._session is not None or self.is_mock:
            return
        with self._lock:
            if self._session is not None or self.is_mock:
                return
            try:
                import onnxruntime as ort
                from tokenizers import Tokenizer

                onnx_path = self.model_dir / _ONNX_FILE
                tok_path = self.model_dir / _TOKENIZER_FILE
                missing = [str(p) for p in (onnx_path, tok_path) if not p.is_file()]
                if missing:
                    raise FileNotFoundError(f"缺少 ONNX 产物: {missing}")

                self._session = ort.InferenceSession(
                    str(onnx_path), providers=["CPUExecutionProvider"])
                self._tokenizer = Tokenizer.from_file(str(tok_path))
                self._tokenizer.enable_truncation(max_length=_MAX_SEQ_LEN)
                self._tokenizer.enable_padding()
            except Exception as e:  # noqa: BLE001 —— 任何加载失败都走兜底
                self.load_error = f"{type(e).__name__}: {e}"
                if self.mock_fallback:
                    self.is_mock = True
                else:
                    raise

    # ---------- 编码 ----------
    def _mock_vec(self, text: str) -> List[float]:
        h = hashlib.sha256(text.encode("utf-8")).digest()
        vals: List[float] = []
        while len(vals) < self.dim:
            for b in h:
                if len(vals) >= self.dim:
                    break
                vals.append((b / 255.0) * 2.0 - 1.0)
            h = hashlib.sha256(h).digest()
        norm = math.sqrt(sum(v * v for v in vals)) or 1.0
        return [v / norm for v in vals]

    def _encode_sync(self, text: str) -> List[float]:
        self._ensure_loaded()
        if self.is_mock:
            return self._mock_vec(text)

        import numpy as np

        enc = self._tokenizer.encode(text)
        feed_all = {
            "input_ids": np.array([enc.ids], dtype=np.int64),
            "attention_mask": np.array([enc.attention_mask], dtype=np.int64),
            "token_type_ids": np.array([enc.type_ids], dtype=np.int64),
        }
        want = {i.name for i in self._session.get_inputs()}
        feed = {k: v for k, v in feed_all.items() if k in want}

        outs = [o.name for o in self._session.get_outputs()]
        target = _OUTPUT_NAME if _OUTPUT_NAME in outs else outs[0]
        hidden = self._session.run([target], feed)[0]      # (1, seq, hidden)
        vec = hidden[0, 0, :].astype("float32")            # cls pooling
        norm = float(np.linalg.norm(vec)) or 1.0
        return (vec / norm).tolist()

    # ---------- 对外接口 ----------
    async def aembed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        return await asyncio.to_thread(lambda: [self._encode_sync(t) for t in texts])

    async def aembed_query(self, text: str) -> List[float]:
        return await asyncio.to_thread(self._encode_sync, text)

    def embed_query(self, text: str) -> List[float]:
        return self._encode_sync(text)

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        return [self._encode_sync(t) for t in texts]


if __name__ == "__main__":
    import sys

    import os

    d = sys.argv[1] if len(sys.argv) > 1 else os.environ.get(
        "LOCAL_EMBEDDING_MODEL_DIR", "models/bge-small-zh-v1.5-onnx"
    )
    emb = LocalOnnxEmbeddings(d, dim=512)
    vs = emb.embed_documents(["华北地区的销售额", "手机数码品类", "2025年第一季度"])
    print(f"model_dir = {d}")
    print(f"is_mock   = {emb.is_mock}  load_error = {emb.load_error}")
    print(f"dim       = {len(vs[0])}")
    # 语义自检：同类应更接近
    import math

    def cos(a, b):
        return sum(x * y for x, y in zip(a, b))

    q = emb.embed_query("华北地区的销售额")
    print(f"cos(华北销售额, 华北地区的销售额) = {cos(q, vs[0]):.4f}")
    print(f"cos(华北销售额, 手机数码品类)     = {cos(q, vs[1]):.4f}")
