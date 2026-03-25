#!/usr/bin/env python3
"""
规则向量检索模块 — LangChain 版本
- 使用 langchain_chroma.Chroma 封装 ChromaDB
- 使用 langchain_core.embeddings 抽象
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import os
from dataclasses import dataclass

from langchain_chroma import Chroma
from langchain_core.documents import Document

from vanilla.rule_retriever import split_rules_into_chunks, RuleChunk

RULES_DIR = os.path.join(os.path.dirname(__file__), "..", "rules")
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "..", ".chroma_db")


def get_vectorstore(force_rebuild: bool = False) -> Chroma:
    """
    获取 Chroma 向量存储，使用 MiniMax Embeddings
    """
    from langchain.minimax_embeddings import MiniMaxEmbeddings

    embeddings = MiniMaxEmbeddings()

    vectorstore = Chroma(
        collection_name="sms_rules",
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )

    if force_rebuild or vectorstore._collection.count() == 0:
        _build_index(vectorstore, embeddings)

    return vectorstore


def _build_index(vectorstore: Chroma, embeddings) -> int:
    """构建规则索引"""
    chunks = split_rules_into_chunks()

    docs = []
    ids = []
    for i, chunk in enumerate(chunks):
        doc = Document(
            page_content=chunk.content,
            metadata={
                "source": chunk.source,
                "category": chunk.category,
                "section": chunk.section,
            }
        )
        docs.append(doc)
        ids.append(f"chunk_{i}")

    vectorstore.add_documents(documents=docs, ids=ids)
    return len(chunks)


def search_rules(query: str, k: int = 5, sms_type: str | None = None) -> list[tuple[Document, float]]:
    """
    检索与 query 最相关的 k 个规则片段

    Args:
        query: 查询文本
        k: 返回数量
        sms_type: 短信类型过滤（目前通过后处理实现）

    Returns:
        (Document, 相似度分数) 列表
    """
    vectorstore = get_vectorstore()

    # 检索更多结果以便过滤
    results = vectorstore.similarity_search_with_score(query, k=k * 3)

    # 过滤和排序
    scored = []
    for doc, score in results:
        # 相似度分数（Chroma 返回的是距离，需要转换）
        similarity = 1.0 - (score / 100.0) if score > 1 else score

        # 同类型加权
        if sms_type and doc.metadata.get("category") == sms_type:
            similarity += 0.1

        scored.append((doc, similarity))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="规则向量检索 (LangChain 版本)")
    parser.add_argument("--build", action="store_true", help="重建索引")
    parser.add_argument("--query", type=str, help="查询文本")
    parser.add_argument("-k", type=int, default=5, help="返回数量")
    parser.add_argument("-t", "--type", dest="sms_type", help="短信类型过滤")
    args = parser.parse_args()

    if args.build:
        vectorstore = get_vectorstore(force_rebuild=True)
        print(f"✅ 索引构建完成，共 {vectorstore._collection.count()} 个规则片段")

    elif args.query:
        results = search_rules(args.query, k=args.k, sms_type=args.sms_type)
        print(f"\n🔍 查询：{args.query}")
        print(f"   类型：{args.sms_type or '不限'}\n")
        for i, (doc, score) in enumerate(results, 1):
            print(f"{i}. [{doc.metadata.get('category')}] {doc.metadata.get('section')} (相似度: {score:.4f})")
            print(f"   {doc.page_content[:120]}...")
            print()