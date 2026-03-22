#!/usr/bin/env python3
"""
规则向量检索模块
- 规则文档分块 → embedding → 本地存储
- 查询时用 SMS 计算相似度 → 召回最相关的规则片段
"""

import json
import math
import os
from dataclasses import dataclass, field

from embeddings import get_embedding_client


RULES_DIR = os.path.join(os.path.dirname(__file__), "rules")
RULE_CHUNKS_FILE = os.path.join(os.path.dirname(__file__), ".rule_chunks.json")


@dataclass
class RuleChunk:
    content: str
    source: str       # 文件名
    category: str     # 规则分类
    section: str      # 章节标题


@dataclass
class IndexedChunk:
    chunk: RuleChunk
    embedding: list[float]


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

        # 按 ## 或 ### 标题拆分
        lines = content.split("\n")
        current_section = "全文"
        current_lines = []

        for line in lines:
            if line.startswith("## ") or line.startswith("### "):
                # 保存上一个块
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

        # 最后一个块
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
# 向量检索
# ─────────────────────────────────────────────

def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b)


class RuleRetriever:
    """
    规则向量检索器
    - build_index()   : 分块 + 嵌入 + 保存本地
    - search(query, k): 返回 top-K 最相关规则片段
    """

    def __init__(self):
        self.chunks: list[RuleChunk] = []
        self.embeddings: list[list[float]] = []
        self._loaded = False

    def build_index(self, force: bool = False) -> int:
        """构建规则索引，返回 chunk 数量"""
        if os.path.exists(RULE_CHUNKS_FILE) and not force:
            self._load()
            return len(self.chunks)

        chunks = split_rules_into_chunks()
        client = get_embedding_client()

        # 批量嵌入
        texts = [c.content for c in chunks]
        batch_size = 16
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            results = client.embed_batch(batch)
            all_embeddings.extend([r.embedding for r in results])

        self.chunks = chunks
        self.embeddings = all_embeddings
        self._save()
        self._loaded = True
        return len(chunks)

    def search(self, query: str, k: int = 5, sms_type: str | None = None) -> list[tuple[RuleChunk, float]]:
        """
        检索与 query 最相关的 k 个规则片段
        - sms_type 不为空时，同类型规则优先
        """
        if not self._loaded:
            self._load()

        client = get_embedding_client()
        query_emb = client.embed(query).embedding

        scored: list[tuple[RuleChunk, float]] = []
        for chunk, emb in zip(self.chunks, self.embeddings):
            sim = cosine_similarity(query_emb, emb)
            # 同类型加权重
            if sms_type and chunk.category == sms_type:
                sim += 0.1
            scored.append((chunk, sim))

        # 排序取 top-K
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    def _load(self):
        """从本地文件加载索引"""
        if not os.path.exists(RULE_CHUNKS_FILE):
            raise FileNotFoundError("规则索引不存在，请先调用 build_index()")

        with open(RULE_CHUNKS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.chunks = [RuleChunk(**c) for c in data["chunks"]]
        self.embeddings = data["embeddings"]
        self._loaded = True

    def _save(self):
        """保存索引到本地文件"""
        data = {
            "chunks": [
                {"content": c.content, "source": c.source, "category": c.category, "section": c.section}
                for c in self.chunks
            ],
            "embeddings": self.embeddings,
        }
        with open(RULE_CHUNKS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


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
