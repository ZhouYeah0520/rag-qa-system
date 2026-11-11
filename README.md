# 企业知识库 RAG 问答系统

面向企业文档的检索增强生成（RAG）问答系统：文档 → 解析分块 → 向量化入库 → 相似度检索 → 拼接上下文 → 大模型生成带引用来源的回答。

## 架构

```
文档(pdf/docx/md/txt)
      │  解析、分块（chunker）
      ▼
   Chunk 列表
      │  TF-IDF 向量化 + 余弦相似度（retrieval，可替换为 Embedding）
      ▼
   检索 top-k 相关片段
      │  拼接为上下文 + 提示词（llm）
      ▼
   大模型生成回答（带来源引用）
```

- **离线可跑**：核心链路（md/txt + TF-IDF 检索 + 抽取式回答）只用 Python 标准库，无需任何 API key。
- **可接大模型**：`--api-key` 传入后走 OpenAI 兼容接口（OpenAI / 通义千问 / 智谱 / DeepSeek 等）。
- **带评测集**：`eval` 命令计算检索命中率，用于迭代分块粒度与提示词。

## 快速开始（无需 API key）

```bash
# 1. 入库：解析 data/ 目录下文档，建立索引（生成 kb.jsonl）
python -m rag_qa ingest data

# 2. 提问（离线抽取式回答，返回最相关原文片段 + 来源）
python -m rag_qa ask "年假需要提前几天申请？"
```

输出示例：

```json
{
  "question": "年假需要提前几天申请？",
  "answer": "[1] 来源 data/员工手册.md#0：\n... 年假需提前 3 个工作日提交申请 ...",
  "sources": ["data/员工手册.md#0"],
  "scores": [0.42]
}
```

## 接入大模型 API（生成式回答）

**DeepSeek**（OpenAI 兼容，国内可用）：

```bash
python -m rag_qa ask "年假需要提前几天申请？" \
  --api-key "你的DeepSeek key" \
  --base-url "https://api.deepseek.com" \
  --model "deepseek-chat"
```

**通义千问（DashScope）**：

```bash
python -m rag_qa ask "年假需要提前几天申请？" \
  --api-key "你的key" \
  --base-url "https://dashscope.aliyuncs.com/compatible-mode/v1" \
  --model "qwen-plus"
```

系统提示词会约束模型"只依据检索到的上下文作答、依据不足就说不确定"，降低幻觉。

## 评测（检索命中率）

```bash
python -m rag_qa eval eval/eval_set.json
```

输出：命中数、命中率、每题是否命中。命中率反映"检索到的片段里是否包含正确答案"，用于对比不同分块粒度/检索策略的效果。

## 运行测试

```bash
python -m unittest discover -s tests -v
```

## 目录结构

```
rag_qa/
  chunker.py    # 文档解析 + 分块（ingestion）
  retrieval.py  # TF-IDF 向量化 + 余弦相似度 top-k
  llm.py        # 大模型客户端（OpenAI 兼容）+ 抽取式回退
  rag.py        # 主流程：ingest / ask / evaluate
  cli.py        # 命令行入口
data/           # 示例文档
eval/           # 评测集
```

## 设计思路与踩坑（面试可用）

1. **分块粒度直接影响检索质量**：块太大噪声多、太小丢失上下文。本项目默认 800 字符 / 100 重叠，且通过 `eval` 量化不同粒度的命中率。
2. **回答必须带来源引用**：没有引用就无法溯源错误，因此把"来源"作为必回字段。
3. **提示词约束幻觉**：系统提示词明确要求"资料中没有就直说"，而不是让模型自由发挥。
4. **TF-IDF 是稀疏检索基线**：适合离线演示与评测；生产可替换为稠密 Embedding（如 OpenAI embedding / 本地模型），接口已按可插拔设计。

## 依赖

- 核心：Python 标准库（无需安装）
- 解析 PDF / DOCX：`pip install pypdf python-docx`（见 requirements.txt）

## 常见问题

- **Windows 终端中文显示乱码**：先执行 `chcp 65001` 或 `set PYTHONUTF8=1`，再运行命令即可正常显示中文（数据本身是 UTF-8，不受影响）。
- **提问返回"资料中未找到相关信息"**：说明该问题在已入库文档里没有对应内容，或需要先用 `ingest` 入库对应文档。
