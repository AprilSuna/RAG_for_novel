"""
文本向量化和索引构建模块（numpy版本）
=====================================
从 data/excerpts.json 加载八月长安作品摘录数据，
使用智谱 API (embedding-3) 将文本向量化，
存储为 JSON 文件，支持余弦相似度检索。
无需 chromadb，零编译依赖。
"""

import json
import os
from typing import Optional

from openai import OpenAI


# ─── 路径常量 ───────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
EXCERPTS_PATH = os.path.join(DATA_DIR, "excerpts.json")
INDEX_PATH = os.path.join(DATA_DIR, "vector_index.json")

# 智谱 API 配置
ZHIPU_API_BASE = "https://open.bigmodel.cn/api/paas/v4/"
EMBEDDING_MODEL = "embedding-3"


def _get_api_key() -> str:
    """获取智谱 API Key，优先从环境变量读取"""
    return os.environ.get("ZHIPU_API_KEY", "")


def _load_excerpts(filepath: str = EXCERPTS_PATH) -> list[dict]:
    """加载摘录 JSON 数据"""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"[embedding] 已加载 {len(data)} 条摘录数据")
    return data


def _get_embeddings(texts: list[str], api_key: str) -> list[list[float]]:
    """调用智谱 API 获取文本的 embedding 向量（分批处理）"""
    client = OpenAI(api_key=api_key, base_url=ZHIPU_API_BASE)

    all_embeddings = []
    batch_size = 10
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch,
        )
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)
        print(f"[embedding] 已向量化 {len(all_embeddings)}/{len(texts)} 条")

    return all_embeddings


def build_index(
    excerpts_path: str = EXCERPTS_PATH,
    index_path: str = INDEX_PATH,
    api_key: str = None,
    force_rebuild: bool = False,
) -> dict:
    """
    构建向量索引

    Parameters
    ----------
    api_key : str, optional
        智谱 API Key
    force_rebuild : bool
        是否强制重建索引

    Returns
    -------
    dict
        包含 excerpts 数据和 embeddings 的索引字典
    """
    key = api_key or _get_api_key()
    if not key:
        raise ValueError("缺少智谱 API Key")

    # 如果索引已存在且不强制重建，直接加载
    if not force_rebuild and os.path.exists(index_path):
        print(f"[embedding] 索引已存在，加载: {index_path}")
        return load_index(index_path)

    excerpts = _load_excerpts(excerpts_path)
    texts = [item["text"] for item in excerpts]

    print(f"[embedding] 正在调用智谱 API 进行向量化 ...")
    embeddings = _get_embeddings(texts, key)

    # 存储索引（JSON 格式，含向量）
    index_data = {
        "model": EMBEDDING_MODEL,
        "count": len(excerpts),
        "excerpts": [
            {
                "id": item["id"],
                "text": item["text"],
                "category": item.get("category", ""),
                "tags": item.get("tags", []),
                "source": item.get("source", ""),
                "scene": item.get("scene", ""),
                "highlight": item.get("highlight", ""),
                "emotion": item.get("emotion", ""),
                "embedding": embeddings[i],
            }
            for i, item in enumerate(excerpts)
        ],
    }

    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index_data, f, ensure_ascii=False)

    print(f"[embedding] 索引构建完成，共 {len(excerpts)} 条记录")
    return index_data


def load_index(index_path: str = INDEX_PATH) -> Optional[dict]:
    """加载已有的向量索引"""
    if not os.path.exists(index_path):
        return None
    with open(index_path, "r", encoding="utf-8") as f:
        return json.load(f)


def index_exists(index_path: str = INDEX_PATH) -> bool:
    """检查索引是否已存在"""
    return os.path.exists(index_path)


if __name__ == "__main__":
    build_index(force_rebuild=True)
