import tempfile
import unittest
from pathlib import Path

from rag_qa.chunker import chunk_text, load_chunks, load_text


class ChunkTextTest(unittest.TestCase):
    def test_short_text_single_chunk(self):
        self.assertEqual(chunk_text("hello", 800, 100), ["hello"])

    def test_empty_text(self):
        self.assertEqual(chunk_text("   ", 800, 100), [])

    def test_boundaries(self):
        self.assertEqual(len(chunk_text("a" * 800, 800, 100)), 1)
        self.assertEqual(len(chunk_text("a" * 801, 800, 100)), 2)
        self.assertEqual(len(chunk_text("a" * 1600, 800, 100)), 3)

    def test_overlap(self):
        chunks = chunk_text("0123456789" * 200, 800, 100)
        self.assertEqual(chunks[0][-100:], chunks[1][:100])


class LoadTextTest(unittest.TestCase):
    def test_load_md(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "a.md"
            p.write_text("# 标题\n内容", encoding="utf-8")
            self.assertIn("内容", load_text(p))

    def test_load_txt(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "a.txt"
            p.write_text("plain", encoding="utf-8")
            self.assertEqual(load_text(p), "plain")

    def test_unsupported_extension(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "a.exe"
            p.write_text("x", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_text(p)


class LoadChunksTest(unittest.TestCase):
    def test_source_naming(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "doc.md"
            p.write_text("第一段" * 300, encoding="utf-8")
            chunks = load_chunks(p, chunk_size=100, overlap=10)
            self.assertGreater(len(chunks), 1)
            self.assertEqual(chunks[0].source, f"{p}#0")
            self.assertEqual(chunks[1].source, f"{p}#1")


if __name__ == "__main__":
    unittest.main()
