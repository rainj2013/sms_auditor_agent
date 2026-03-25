#!/usr/bin/env python3
"""
MiniMax Embeddings 封装 — 基于 OpenAI SDK
用于规则向量检索
"""

import os
from dataclasses import dataclass

from openai import OpenAI

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
    使用 OpenAI SDK 调用
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

        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url.rstrip("/"),
            timeout=120,
        )

    def embed(self, text: str) -> EmbeddingResult:
        """获取单条文本的 embedding 向量"""
        response = self._client.embeddings.create(
            model=self.model,
            input=[text],
            extra_body={"type": "db"},
        )
        # 适配 MiniMax 返回格式：response.data[0].embedding
        embedding_data = response.data[0].embedding
        return EmbeddingResult(embedding=embedding_data, model=response.model)

    def embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        """批量获取文本 embedding"""
        response = self._client.embeddings.create(
            model=self.model,
            input=texts,
            extra_body={"type": "db"},
        )
        results = []
        for item in response.data:
            results.append(EmbeddingResult(
                embedding=item.embedding,
                model=response.model
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