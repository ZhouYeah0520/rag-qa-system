"""命令行入口：ingest / ask / eval。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from .llm import ExtractiveLLM, OpenAICompatLLM
from .rag import KnowledgeBase

KB_FILE = "kb.jsonl"


def cmd_ingest(args) -> int:
    paths = []
    directory = Path(args.directory)
    if not directory.is_dir():
        print(f"error: '{args.directory}' 不是目录", file=sys.stderr)
        return 2
    for ext in (".md", ".txt", ".pdf", ".docx"):
        paths.extend(sorted(directory.rglob(f"*{ext}")))
    if not paths:
        print("未找到可解析的文档（.md/.txt/.pdf/.docx）", file=sys.stderr)
        return 2

    kb = KnowledgeBase(chunk_size=args.chunk_size, overlap=args.overlap)
    n = kb.ingest(paths)
    _save_index(kb)
    print(f"已入库 {len(paths)} 个文档，共 {n} 个 chunk")
    return 0


def _save_index(kb: KnowledgeBase) -> None:
    """把 chunk 与索引保存为 jsonl，便于下次直接加载。"""
    with open(KB_FILE, "w", encoding="utf-8") as fh:
        for c in kb.retriever.chunks:
            fh.write(json.dumps({"id": c.id, "source": c.source, "text": c.text}, ensure_ascii=False) + "\n")


def _load_index(kb: KnowledgeBase) -> None:
    from .chunker import Chunk

    chunks = []
    with open(KB_FILE, encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            chunks.append(Chunk(id=d["id"], source=d["source"], text=d["text"]))
    kb.retriever.index(chunks)


def cmd_ask(args) -> int:
    kb = KnowledgeBase(chunk_size=args.chunk_size, overlap=args.overlap)
    if Path(KB_FILE).exists():
        _load_index(kb)

    if args.api_key:
        kb.llm = OpenAICompatLLM(args.api_key, args.base_url, args.model)

    result = kb.ask(args.question, top_k=args.top_k)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def cmd_eval(args) -> int:
    kb = KnowledgeBase(chunk_size=args.chunk_size, overlap=args.overlap)
    if Path(KB_FILE).exists():
        _load_index(kb)
    result = kb.evaluate(args.eval_set, top_k=args.top_k)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag_qa",
        description="企业知识库 RAG 问答系统：ingest 入库 / ask 问答 / eval 评测。",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="将目录下文档解析、分块并建立索引")
    p_ingest.add_argument("directory", help="文档目录")
    p_ingest.set_defaults(func=cmd_ingest)

    p_ask = sub.add_parser("ask", help="提问")
    p_ask.add_argument("question", help="问题")
    p_ask.add_argument("--api-key", default=None, help="大模型 API key（不填则用抽取式回答）")
    p_ask.add_argument("--base-url", default="https://api.openai.com/v1", help="OpenAI 兼容 API 地址")
    p_ask.add_argument("--model", default="gpt-4o-mini", help="模型名")
    p_ask.set_defaults(func=cmd_ask)

    p_eval = sub.add_parser("eval", help="用评测集计算检索命中率")
    p_eval.add_argument("eval_set", help="评测集 JSON 文件")
    p_eval.set_defaults(func=cmd_eval)

    for p in (p_ingest, p_ask, p_eval):
        p.add_argument("--chunk-size", type=int, default=800)
        p.add_argument("--overlap", type=int, default=100)
        p.add_argument("--top-k", type=int, default=3)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)
