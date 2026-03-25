#!/usr/bin/env python3
"""
规则向量检索模块 — 基于 ChromaDB
- 规则文档分块 → embedding → ChromaDB 持久化存储
- 查询时用 SMS 计算相似度 → 召回最相关的规则片段
"""

import json
import math
import os
from dataclasses import dataclass

import chromadb
from chromadb import PersistentClient

from vanilla.embeddings import get_embedding_client


# 向上查找项目根目录
def _find_root():
    path = os.path.dirname(__file__)
    while path != os.path.dirname(path):
        if os.path.exists(os.path.join(path, "rules")):
            return path
        path = os.path.dirname(path)
    return os.path.dirname(__file__)

RULES_DIR = os.path.join(_find_root(), "rules")
CHROMA_DIR = os.path.join(_find_root(), ".chroma_db")


@dataclass
class RuleChunk:
    content: str
    source: str       # 文件名
    category: str     # 规则分类
    section: str      # 章节标题


# ─────────────────────────────────────────────
# 规则分块
# ─────────────────────────────────────────────

def split_rules_into_chunks() -> list[RuleChunk]:
    """按章节/条目拆分规则文档为小块"""
    rule_files = {
        "00_短信合规总纲.md": "通用规则",
        "01_验证码短信规范.md": "验证码",
        "02_营销短信规范.md": "营销",
        "03_催收短信规范.md": "催收",
        "04_权益通知短信规范.md": "权益通知",
    }

    chunks = []

    for filename, category in rule_files.items():
        filepath = os.path.join(RULES_DIR, filename)
        if not os.path.exists(filepath):
            continue

        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        lines = content.split("\n")
        current_section = "全文"
        current_lines = []

        for line in lines:
            if line.startswith("## ") or line.startswith("### "):
                if current_lines:
                    chunk_text = "\n".join(current_lines).strip()
                    if chunk_text:
                        chunks.append(RuleChunk(
                            content=chunk_text,
                            source=filename,
                            category=category,
                            section=current_section,
                        ))
                current_section = line.lstrip("#").strip()
                current_lines = [line]
            else:
                current_lines.append(line)

        if current_lines:
            chunk_text = "\n".join(current_lines).strip()
            if chunk_text:
                chunks.append(RuleChunk(
                    content=chunk_text,
                    source=filename,
                    category=category,
                    section=current_section,
                ))

    return chunks


# ─────────────────────────────────────────────
# ChromaDB 检索器
# ─────────────────────────────────────────────

class RuleRetriever:
    """
    规则向量检索器（ChromaDB 持久化）
    - build_index()   : 分块 + 嵌入 + 存入 ChromaDB
    - search(query, k): 返回 top-K 最相关规则片段
    """

    COLLECTION_NAME = "sms_rules"

    def __init__(self):
        self.client = PersistentClient(
            path=CHROMA_DIR,
            settings=chromadb.Settings(anonymized_telemetry=False),
        )
        self._ensure_collection()

    def _ensure_collection(self):
        """确保 collection 存在"""
        try:
            self.client.get_collection(self.COLLECTION_NAME)
        except Exception:
            self.client.create_collection(
                name=self.COLLECTION_NAME,
                metadata={"description": "短信合规规则向量库"}
            )

    def build_index(self, force: bool = False) -> int:
        """构建规则索引，返回 chunk 数量"""
        collection = self.client.get_collection(self.COLLECTION_NAME)

        if collection.count() > 0 and not force:
            return collection.count()

        chunks = split_rules_into_chunks()
        client = get_embedding_client()

        # 批量嵌入
        texts = [c.content for c in chunks]
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        metadatas = [
            {"source": c.source, "category": c.category, "section": c.section}
            for c in chunks
        ]

        batch_size = 16
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            results = client.embed_batch(batch)
            all_embeddings.extend([r.embedding for r in results])

        collection.add(
            embeddings=all_embeddings,
            documents=texts,
            ids=ids,
            metadatas=metadatas,
        )

        return len(chunks)

    def search(self, query: str, k: int = 5, sms_type: str | None = None) -> list[tuple[RuleChunk, float]]:
        """
        检索与 query 最相关的 k 个规则片段
        - sms_type 不为空时，同类型规则优先
        """
        collection = self.client.get_collection(self.COLLECTION_NAME)

        query_emb = get_embedding_client().embed(query).embedding

        # 先查 k * 3 个，增加召回范围以便过滤
        results = collection.query(
            query_embeddings=[query_emb],
            n_results=k * 3,
            include=["documents", "metadatas", "distances"],
        )

        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

        scored: list[tuple[RuleChunk, float]] = []
        for doc, meta, dist in zip(docs, metas, dists):
            if not doc:
                continue
            chunk = RuleChunk(
                content=doc,
                source=meta.get("source", ""),
                category=meta.get("category", ""),
                section=meta.get("section", ""),
            )
            # ChromaDB distance 转相似度（距离越小越相似）
            sim = 1.0 - dist if dist else 0.0
            # 同类型加权
            if sms_type and chunk.category == sms_type:
                sim += 0.1
            scored.append((chunk, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]


# ─────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="规则向量检索")
    parser.add_argument("--build", action="store_true", help="重建索引")
    parser.add_argument("--query", type=str, help="查询文本")
    parser.add_argument("-k", type=int, default=5, help="返回数量")
    parser.add_argument("-t", "--type", dest="sms_type", help="短信类型过滤")
    args = parser.parse_args()

    retriever = RuleRetriever()

    if args.build:
        n = retriever.build_index(force=True)
        print(f"✅ 索引构建完成，共 {n} 个规则片段")

    elif args.query:
        results = retriever.search(args.query, k=args.k, sms_type=args.sms_type)
        print(f"\n🔍 查询：{args.query}")
        print(f"   类型：{args.sms_type or '不限'}\n")
        for i, (chunk, score) in enumerate(results, 1):
            print(f"{i}. [{chunk.category}] {chunk.section} (相似度: {score:.4f})")
            print(f"   {chunk.content[:120]}...")
            print()
