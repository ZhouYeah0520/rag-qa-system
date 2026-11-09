"""向量检索：TF-IDF 向量化 + 余弦相似度 top-k 召回。

默认使用 TF-IDF 稀疏向量（可离线运行、无第三方依赖），
接口上可替换为稠密 Embedding（如 OpenAI embedding / 本地模型）。
"""

from __future__ import annotations

import math
import re
from collections import Counter

from .chunker import Chunk

_CJK = r"\u4e00-\u9fff"


def tokenize(text: str) -> list[str]:
    """简单分词：拉丁字母/数字词 + 单个中文字符。

    生产环境建议替换为 jieba 分词或真实 Embedding 模型。
    """
    text = text.lower()
    return re.findall(rf"[a-z0-9]+|[{_CJK}]", text)


class TfidfRetriever:
    """基于 TF-IDF 与余弦相似度的稀疏向量检索器。"""

    def __init__(self):
        self.chunks: list[Chunk] = []
        self._doc_vectors: list[dict[str, float]] = []

    def index(self, chunks: list[Chunk]) -> None:
        """对分块建立索引（计算 TF-IDF 向量）。"""
        self.chunks = list(chunks)
        docs = [Counter(tokenize(c.text)) for c in chunks]
        n = max(len(docs), 1)
        df: Counter = Counter()
        for doc in docs:
            for term in doc:
                df[term] += 1

        vectors: list[dict[str, float]] = []
        for doc in docs:
            vec: dict[str, float] = {}
            norm = 0.0
            for term, tf in doc.items():
                w = tf * (math.log((1 + n) / (1 + df[term])) + 1.0)
                vec[term] = w
                norm += w * w
            norm = math.sqrt(norm) or 1.0
            vectors.append({t: w / norm for t, w in vec.items()})
        self._doc_vectors = vectors

    def _query_vector(self, query: str) -> dict[str, float]:
        q = Counter(tokenize(query))
        if not q:
            return {}
        vec: dict[str, float] = {}
        norm = 0.0
        for term, tf in q.items():
            # 与文档同权重的近似处理：仅用词频
            vec[term] = float(tf)
            norm += tf * tf
        norm = math.sqrt(norm) or 1.0
        return {t: w / norm for t, w in vec.items()}

    def search(self, query: str, top_k: int = 3) -> list[tuple[Chunk, float]]:
        """返回 (chunk, 相似度) 列表，按相似度降序，只保留分数 > 0 的。"""
        qv = self._query_vector(query)
        if not qv or not self._doc_vectors:
            return []

        scored: list[tuple[int, float]] = []
        for i, dv in enumerate(self._doc_vectors):
            score = sum(w * dv.get(t, 0.0) for t, w in qv.items())
            scored.append((i, score))

        scored.sort(key=lambda x: -x[1])
        return [
            (self.chunks[i], round(score, 6))
            for i, score in scored[:top_k]
            if score > 0
        ]
