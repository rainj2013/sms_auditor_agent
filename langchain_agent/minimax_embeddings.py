#!/usr/bin/env python3
"""
MiniMax Embeddings 封装 — LangChain 版本
用于规则向量检索
"""

import os

from langchain_core.embeddings import Embeddings

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from vanilla.embeddings import EmbeddingClient as MiniMaxEmbeddingClient


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