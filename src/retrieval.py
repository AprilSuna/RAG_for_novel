"""
检索模块（numpy版本）
====================
使用余弦相似度从向量索引中检索与查询最相似的八月长安风格段落，
支持按 category 和 tags 过滤。
无需 chromadb，使用 numpy 计算余弦相似度。
"""

from typing import Optional

import numpy as np
from openai import OpenAI

from .embedding import (
    EMBEDDING_MODEL,
    ZHIPU_API_BASE,
    _get_api_key,
    build_index,
    load_index,
)


def _ensure_index(api_key: str = None) -> dict:
    """确保索引存在，不存在则自动构建"""
    index = load_index()
    if index is None:
        print("[retrieval] 索引不存在，正在自动构建 ...")
        index = build_index(api_key=api_key, force_rebuild=True)
    return index


def _get_query_embedding(query: str, api_key: str) -> np.ndarray:
    """获取查询文本的 embedding 向量"""
    client = OpenAI(api_key=api_key, base_url=ZHIPU_API_BASE)
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[query],
    )
    return np.array(response.data[0].embedding)


def _cosine_similarity_matrix(query_vec: np.ndarray, doc_matrix: np.ndarray) -> np.ndarray:
    """计算查询向量与文档矩阵的余弦相似度"""
    # query_vec: (dim,), doc_matrix: (n, dim)
    dot_products = doc_matrix @ query_vec
    query_norm = np.linalg.norm(query_vec)
    doc_norms = np.linalg.norm(doc_matrix, axis=1)
    denominator = doc_norms * query_norm
    # 避免除零
    denominator[denominator == 0] = 1e-10
    return dot_products / denominator


def search(
    query: str,
    category_filter: Optional[str] = None,
    tags_filter: Optional[str] = None,
    emotion_filter: Optional[str] = None,
    top_k: int = 3,
    api_key: str = None,
) -> list[dict]:
    """
    检索与查询最相似的段落

    Parameters
    ----------
    query : str
        查询文本
    category_filter : str, optional
        按分类过滤
    tags_filter : str, optional
        按标签过滤（精确匹配某个标签）
    top_k : int
        返回最相似的 top_k 条结果
    api_key : str, optional
        智谱 API Key

    Returns
    -------
    list[dict]
        检索结果列表
    """
    key = api_key or _get_api_key()
    if not key:
        raise ValueError("缺少智谱 API Key")

    index = _ensure_index(api_key=key)
    excerpts = index["excerpts"]

    # 按分类和标签过滤
    filtered_indices = []
    filtered_items = []
    for i, item in enumerate(excerpts):
        if category_filter and item.get("category") != category_filter:
            continue
        if tags_filter:
            item_tags = item.get("tags", [])
            if tags_filter not in item_tags:
                continue
        if emotion_filter:
            if item.get("emotion") != emotion_filter:
                continue
        filtered_indices.append(i)
        filtered_items.append(item)

    if not filtered_items:
        print(f"[retrieval] 过滤后无结果")
        return []

    # 构建文档向量矩阵
    doc_matrix = np.array(
        [filtered_items[j]["embedding"] for j in range(len(filtered_items))]
    )

    # 获取查询向量
    query_vec = _get_query_embedding(query, key)

    # 计算余弦相似度
    similarities = _cosine_similarity_matrix(query_vec, doc_matrix)

    # 排序取 top_k
    ranked_indices = np.argsort(similarities)[::-1][: min(top_k, len(filtered_items))]

    parsed = []
    for idx in ranked_indices:
        item = filtered_items[idx]
        sim = float(similarities[idx])
        parsed.append(
            {
                "id": str(item["id"]),
                "text": item["text"],
                "category": item.get("category", ""),
                "tags": item.get("tags", []),
                "source": item.get("source", ""),
                "scene": item.get("scene", ""),
                "highlight": item.get("highlight", ""),
                "emotion": item.get("emotion", ""),
                "distance": 1.0 - sim,  # distance = 1 - similarity，保持与旧接口兼容
            }
        )

    print(f"[retrieval] 查询 '{query}' 返回 {len(parsed)} 条结果")
    return parsed


def list_categories(api_key: str = None) -> list[str]:
    """列出所有可用的分类"""
    index = _ensure_index(api_key=api_key)
    categories = set()
    for item in index["excerpts"]:
        if item.get("category"):
            categories.add(item["category"])
    return sorted(categories)


def list_all_tags(api_key: str = None) -> list[str]:
    """列出所有可用的标签"""
    index = _ensure_index(api_key=api_key)
    tags = set()
    for item in index["excerpts"]:
        for t in item.get("tags", []):
            tags.add(t)
    return sorted(tags)


if __name__ == "__main__":
    print("=== 可用分类 ===")
    for cat in list_categories():
        print(f"  - {cat}")

    print("\n=== 可用标签 ===")
    for tag in list_all_tags():
        print(f"  - {tag}")

    print("\n=== 测试检索 ===")
    results = search("暗恋的克制与隐忍", top_k=2)
    for r in results:
        print(f"\n[ID={r['id']}] 距离={r['distance']:.4f}")
        print(f"  分类: {r['category']}")
        print(f"  标签: {', '.join(r['tags'])}")
        print(f"  原文: {r['text'][:80]}...")


def list_emotions(api_key: str = None) -> list[str]:
    """列出所有可用的情绪基调"""
    index = _ensure_index(api_key=api_key)
    emotions = set()
    for item in index["excerpts"]:
        if item.get("emotion"):
            emotions.add(item["emotion"])
    return sorted(emotions)
