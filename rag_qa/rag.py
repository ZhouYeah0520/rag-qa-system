"""RAG 主流程：ingest（建索引）→ ask（检索 + 生成）。"""

from __future__ import annotations

from pathlib import Path

from .chunker import Chunk, load_chunks
from .llm import BaseLLM, ExtractiveLLM
from .retrieval import TfidfRetriever


class KnowledgeBase:
    """企业知识库：负责文档入库与问答。"""

    def __init__(
        self,
        retriever: TfidfRetriever | None = None,
        llm: BaseLLM | None = None,
        chunk_size: int = 800,
        overlap: int = 100,
    ):
        self.retriever = retriever or TfidfRetriever()
        self.llm = llm or ExtractiveLLM()
        self.chunk_size = chunk_size
        self.overlap = overlap

    def ingest(self, paths: list[str | Path]) -> int:
        """加载并索引一批文档，返回总 chunk 数。"""
        chunks: list[Chunk] = []
        for p in paths:
            for c in load_chunks(p, self.chunk_size, self.overlap):
                c.id = len(chunks)
                chunks.append(c)
        self.retriever.index(chunks)
        return len(chunks)

    def ask(self, question: str, top_k: int = 3) -> dict:
        """检索 top-k 相关片段并生成回答（带来源）。"""
        hits = self.retriever.search(question, top_k=top_k)
        context = [c for c, _ in hits]
        answer = self.llm.answer(question, context)
        return {
            "question": question,
            "answer": answer,
            "sources": [c.source for c in context],
            "scores": [s for _, s in hits],
        }

    def evaluate(self, eval_path: str | Path, top_k: int = 3) -> dict:
        """用评测集计算"检索命中率"：期望答案是否出现在召回片段中。"""
        import json

        items = json.loads(Path(eval_path).read_text(encoding="utf-8"))
        hits = 0
        details = []
        for item in items:
            question = item["question"]
            expected = item["answer_contains"]
            found = any(
                expected in chunk.text
                for chunk, _ in self.retriever.search(question, top_k=top_k)
            )
            hits += int(found)
            details.append({"question": question, "hit": found})
        return {
            "total": len(items),
            "hits": hits,
            "hit_rate": round(hits / len(items), 4) if items else 0.0,
            "details": details,
        }
