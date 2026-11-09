import json
import tempfile
import unittest
from pathlib import Path

from rag_qa.llm import ExtractiveLLM
from rag_qa.rag import KnowledgeBase


def write_doc(tmp: str) -> str:
    p = Path(tmp) / "手册.md"
    p.write_text(
        "# 手册\n请假需提前三个工作日申请。\n报销需十个工作日内提交发票。\n"
        "VPN 连续登录十二小时自动断开。\n设备故障可提交 IT 工单报修。",
        encoding="utf-8",
    )
    return str(p)


class RagTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.kb = KnowledgeBase()
        self.n = self.kb.ingest([write_doc(self.tmp.name)])

    def tearDown(self):
        self.tmp.cleanup()

    def test_ingest_builds_index(self):
        self.assertGreater(self.n, 0)

    def test_ask_returns_answer_and_sources(self):
        result = self.kb.ask("年假怎么申请？", top_k=2)
        self.assertIn("question", result)
        self.assertIn("answer", result)
        self.assertIn("sources", result)
        self.assertTrue(result["sources"])

    def test_extractive_answer_contains_context(self):
        result = self.kb.ask("设备坏了怎么报修？", top_k=3)
        self.assertIn("IT 工单", result["answer"])

    def test_extractive_answer_no_match(self):
        result = self.kb.ask("量子力学是什么？", top_k=3)
        self.assertEqual(result["answer"], "资料中未找到相关信息。")
        self.assertEqual(result["sources"], [])

    def test_evaluate_hit_rate(self):
        with tempfile.TemporaryDirectory() as tmp:
            eval_file = Path(tmp) / "eval.json"
            eval_file.write_text(
                json.dumps(
                    [
                        {"question": "年假怎么申请？", "answer_contains": "三个工作日"},
                        {"question": "报销多久？", "answer_contains": "十个工作日"},
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            result = self.kb.evaluate(eval_file, top_k=3)
        self.assertEqual(result["total"], 2)
        self.assertGreaterEqual(result["hits"], 1)
        self.assertIn("hit_rate", result)


class ExtractiveLLMTest(unittest.TestCase):
    def test_empty_context(self):
        llm = ExtractiveLLM()
        self.assertEqual(llm.answer("q", []), "资料中未找到相关信息。")


if __name__ == "__main__":
    unittest.main()
