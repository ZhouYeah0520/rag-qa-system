import unittest

from rag_qa.chunker import Chunk
from rag_qa.retrieval import TfidfRetriever, tokenize


def make_chunks():
    return [
        Chunk(id=0, source="doc1#0", text="请假需要提前三个工作日申请，由主管审批。"),
        Chunk(id=1, source="doc1#1", text="报销需要提交发票原件，十个工作日内完成。"),
        Chunk(id=2, source="doc2#0", text="VPN 连续登录十二小时会自动断开。"),
    ]


class TokenizeTest(unittest.TestCase):
    def test_latin_and_cjk(self):
        tokens = tokenize("VPN 登录 12 小时")
        self.assertIn("vpn", tokens)
        self.assertIn("12", tokens)
        self.assertIn("登", tokens)


class RetrieverTest(unittest.TestCase):
    def setUp(self):
        self.retriever = TfidfRetriever()
        self.retriever.index(make_chunks())

    def test_search_returns_most_relevant(self):
        hits = self.retriever.search("年假怎么申请", top_k=1)
        self.assertEqual(len(hits), 1)
        self.assertIn("请假", hits[0][0].text)

    def test_top_k_respected(self):
        hits = self.retriever.search("请假 报销 VPN", top_k=2)
        self.assertLessEqual(len(hits), 2)

    def test_no_match_returns_empty(self):
        hits = self.retriever.search("zzzz 不存在的词", top_k=3)
        self.assertEqual(hits, [])

    def test_scores_descending(self):
        hits = self.retriever.search("报销流程", top_k=3)
        scores = [s for _, s in hits]
        self.assertEqual(scores, sorted(scores, reverse=True))


if __name__ == "__main__":
    unittest.main()
