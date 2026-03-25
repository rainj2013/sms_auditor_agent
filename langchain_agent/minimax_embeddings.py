#!/usr/bin/env python3
"""
MiniMax Embeddings 封装 — LangChain 版本
基于 OpenAI SDK，自己实现不依赖 vanilla
"""

import os
from dataclasses import dataclass

from langchain_core.embeddings import Embeddings
from openai import OpenAI


def _find_root():
    """向上查找项目根目录"""
    path = os.path.dirname(os.path.dirname(__file__))
    while path != os.path.dirname(path):
        if os.path.exists(os.path.join(path, "llm_config.json")):
            return path
        path = os.path.dirname(path)
    return os.path.dirname(os.path.dirname(__file__))


def _load_config():
    """加载 llm_config.json"""
    import json
    config_path = os.path.join(_find_root(), "llm_config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


@dataclass
class EmbeddingResult:
    embedding: list[float]
    model: str


class MiniMaxEmbeddingClient:
    """MiniMax Embeddings 客户端"""

    def __init__(self):
        config = _load_config()
        cfg = config.get("embedding", {})

        self.api_key = (
            os.environ.get("EMBEDDING_API_KEY", "")
            or os.environ.get("MINIMAX_API_KEY", "")
        )
        if not self.api_key:
            raise ValueError("未设置 Embedding API Key，请设置 EMBEDDING_API_KEY 或 MINIMAX_API_KEY 环境变量")

        self.base_url = (
            os.environ.get("EMBEDDING_BASE_URL", "")
            or cfg.get("base_url", "https://api.minimaxi.com/v1")
        )
        self.model = (
            os.environ.get("EMBEDDING_MODEL", "")
            or cfg.get("model", "embo-01")
        )

        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url.rstrip("/"),
            timeout=120,
        )

    def embed(self, text: str) -> EmbeddingResult:
        """获取单条文本的 embedding 向量"""
        raw = self._client.post(
            "/embeddings",
            body={"model": self.model, "texts": [text], "type": "db"},
            cast_to=object,
        )
        vectors = raw.get("vectors", [])
        if not vectors:
            raise RuntimeError(f"Embeddings 返回为空：{raw}")
        embedding = vectors[0] if isinstance(vectors[0], list) else vectors
        return EmbeddingResult(embedding=embedding, model=raw.get("model", self.model))

    def embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        """批量获取文本 embedding"""
        raw = self._client.post(
            "/embeddings",
            body={"model": self.model, "texts": texts, "type": "db"},
            cast_to=object,
        )
        vectors = raw.get("vectors", [])
        if not vectors:
            raise RuntimeError(f"Embeddings 返回为空：{raw}")

        results = []
        for vec in vectors:
            embedding = vec if isinstance(vec, list) else vec.get("embedding")
            results.append(EmbeddingResult(
                embedding=embedding,
                model=raw.get("model", self.model)
            ))
        return results


class MiniMaxEmbeddings(Embeddings):
    """
    LangChain Embeddings 抽象类实现
    封装 MiniMax Embeddings 客户端
    """

    def __init__(self):
        self._client = MiniMaxEmbeddingClient()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量嵌入文档"""
        results = self._client.embed_batch(texts)
        return [r.embedding for r in results]

    def embed_query(self, text: str) -> list[float]:
        """嵌入单条查询"""
        return self._client.embed(text).embedding