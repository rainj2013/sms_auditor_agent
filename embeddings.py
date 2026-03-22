#!/usr/bin/env python3
"""
MiniMax Embeddings 封装 — 用于规则向量检索
"""

import json
import os
from dataclasses import dataclass
from typing import Literal

import urllib.request
import urllib.error

from llm_providers import load_config


@dataclass
class EmbeddingResult:
    embedding: list[float]
    model: str


def get_embedding_client() -> "EmbeddingClient":
    return EmbeddingClient()


class EmbeddingClient:
    """
    MiniMax Embeddings 客户端（OpenAI 兼容格式）
    """

    def __init__(self, api_key: str | None = None, model: str | None = None, base_url: str | None = None):
        config = load_config()
        cfg = config.get("embedding", {})

        self.api_key = (
            api_key
            or os.environ.get("EMBEDDING_API_KEY", "")
            or os.environ.get("MINIMAX_API_KEY", "")
        )
        if not self.api_key:
            raise ValueError("未设置 Embedding API Key，请设置 EMBEDDING_API_KEY 环境变量")

        self.base_url = (
            base_url
            or os.environ.get("EMBEDDING_BASE_URL", "")
            or cfg.get("base_url", "https://api.minimaxi.com/v1")
        )
        self.model = (
            model
            or os.environ.get("EMBEDDING_MODEL", "")
            or cfg.get("model", "embo-01")
        )

    def embed(self, text: str) -> EmbeddingResult:
        """获取单条文本的 embedding 向量"""
        url = f"{self.base_url}/embeddings"

        payload = {
            "model": self.model,
            "texts": [text],
            "type": "db",  # MiniMax embeddings 必填：db=数据库向量化
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST"
        )

        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                result = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8")
            raise RuntimeError(f"Embeddings API 错误 {e.code}：{error_body}")

        vectors = result.get("vectors", [])
        if not vectors:
            raise RuntimeError(f"Embeddings 返回为空：{result}")
        # vectors 是 float 数组的列表
        embedding = vectors[0] if isinstance(vectors[0], list) else vectors

        return EmbeddingResult(embedding=embedding, model=result.get("model", self.model))

    def embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        """批量获取文本 embedding"""
        url = f"{self.base_url}/embeddings"

        payload = {
            "model": self.model,
            "texts": texts,
            "type": "db",
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST"
        )

        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8")
            raise RuntimeError(f"Embeddings API 错误 {e.code}：{error_body}")

        vectors = result.get("vectors", [])
        if not vectors:
            raise RuntimeError(f"Embeddings 返回为空：{result}")

        results = []
        for vec in vectors:
            embedding = vec if isinstance(vec, list) else vec.get("embedding")
            results.append(EmbeddingResult(
                embedding=embedding,
                model=result.get("model", self.model)
            ))
        return results


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """计算两个向量的余弦相似度"""
    import math
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b)


if __name__ == "__main__":
    # 快速测试
    client = get_embedding_client()
    text = "【XX银行】您已逾期30天，欠款5000元"
    result = client.embed(text)
    print(f"Model: {result.model}")
    print(f"Embedding 维度: {len(result.embedding)}")
    print(f"前5维: {result.embedding[:5]}")
