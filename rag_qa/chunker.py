"""文档加载与分块：RAG 的 ingestion 环节。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".md", ".txt"}


@dataclass
class Chunk:
    """一个文本分块，带来源信息，便于回答溯源。"""

    id: int
    source: str  # 形如 "data/员工手册.md#2"
    text: str


def load_text(path: str | Path) -> str:
    """按扩展名提取文档文本。PDF/DOCX 需要对应依赖，未安装时给出明确提示。"""
    path = Path(path)
    ext = path.suffix.lower()

    if ext in (".md", ".txt"):
        return path.read_text(encoding="utf-8")

    if ext == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("解析 PDF 需要安装 pypdf：pip install pypdf") from exc
        reader = PdfReader(str(path))
        return "\n".join((page.extract_text() or "") for page in reader.pages)

    if ext == ".docx":
        try:
            import docx
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("解析 DOCX 需要安装 python-docx：pip install python-docx") from exc
        d = docx.Document(str(path))
        return "\n".join(p.text for p in d.paragraphs)

    raise ValueError(f"不支持的扩展名：{ext}（支持 {sorted(SUPPORTED_EXTENSIONS)}）")


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    """按固定大小切分文本，块之间保留 overlap 重叠。"""
    text = text.strip()
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        if end >= n:
            break
        start = end - overlap
    return chunks


def load_chunks(path: str | Path, chunk_size: int = 800, overlap: int = 100) -> list[Chunk]:
    """加载一个文档并切成带来源信息的 Chunk 列表。"""
    source = str(path)
    text = load_text(path)
    return [
        Chunk(id=0, source=f"{source}#{i}", text=c)
        for i, c in enumerate(chunk_text(text, chunk_size, overlap))
    ]
