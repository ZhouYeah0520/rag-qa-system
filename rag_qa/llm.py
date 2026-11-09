"""大模型生成：支持 OpenAI 兼容 API，也可离线回退为"抽取式回答"。"""

from __future__ import annotations

from .chunker import Chunk

SYSTEM_PROMPT = (
    "你是企业知识库助手。请只依据下面提供的上下文回答用户问题，"
    "并给出你依据的上下文编号。如果上下文不足以回答，请明确说"
    "'资料中未找到相关信息'，不要编造。"
)


class BaseLLM:
    def answer(self, question: str, context: list[Chunk]) -> str:
        raise NotImplementedError


class ExtractiveLLM(BaseLLM):
    """无 API key 时的回退方案：直接返回检索到的原文片段作为"答案"。"""

    def answer(self, question: str, context: list[Chunk]) -> str:
        if not context:
            return "资料中未找到相关信息。"
        parts = [f"[{i + 1}] 来源 {c.source}：\n{c.text}" for i, c in enumerate(context)]
        return "\n\n".join(parts)


class OpenAICompatLLM(BaseLLM):
    """调用 OpenAI 兼容的 chat/completions 接口（OpenAI / 通义千问 / 智谱 / DeepSeek 等）。"""

    def __init__(self, api_key: str, base_url: str, model: str):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model

    def answer(self, question: str, context: list[Chunk]) -> str:
        import json
        import urllib.request

        context_text = "\n\n".join(
            f"[{i + 1}] {c.text}" for i, c in enumerate(context)
        )
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"上下文：\n{context_text}\n\n问题：{question}",
                },
            ],
            "temperature": 0.2,
        }
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"]
