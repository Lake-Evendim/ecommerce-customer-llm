from __future__ import annotations

from typing import Any, Dict, Iterable, List


def format_retrieved_docs(docs: Iterable[Dict[str, Any]]) -> str:
    blocks: List[str] = []
    for i, doc in enumerate(docs, start=1):
        doc_id = doc.get("doc_id", "")
        chunk_id = doc.get("chunk_id", "")
        title = doc.get("title", "")
        content = doc.get("content", "")
        score = doc.get("score", None)
        score_text = f"，score={score:.4f}" if isinstance(score, float) else ""
        blocks.append(
            f"【知识{i}】doc_id={doc_id}, chunk_id={chunk_id}{score_text}\n"
            f"标题：{title}\n"
            f"内容：{content}"
        )
    return "\n\n".join(blocks)


def build_rag_prompt(user_query: str, retrieved_docs: List[Dict[str, Any]], chat_history: str = "") -> str:
    retrieved_context = format_retrieved_docs(retrieved_docs)
    if not retrieved_context:
        retrieved_context = "未检索到可靠知识。"

    return f"""你是电商平台客服助手，请根据以下知识库内容回答用户问题。

【用户历史对话】
{chat_history or "无"}

【检索到的知识】
{retrieved_context}

【用户当前问题】
{user_query}

【回答要求】
- 使用中文回答；
- 优先基于检索到的知识回答；
- 不要编造知识库中没有的信息；
- 如果检索内容不足以支持回答，请说明需要进一步核实；
- 如涉及订单状态、赔付金额、投诉升级，请建议转人工；
- 如能解决，请给出明确操作步骤；
- 回复应先安抚，再说明规则，最后给出下一步。

请输出客服回复："""
