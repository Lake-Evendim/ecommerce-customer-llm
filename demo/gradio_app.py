from __future__ import annotations

import json
import subprocess
import sys
from typing import Any, Dict, List, Tuple

import gradio as gr


CONFIG_PATH = "configs/infer_final.yaml"


def run_infer_final(query: str, safe_mode: bool = True) -> Dict[str, Any]:
    """
    调用最终推理入口 scripts.infer_final，并解析 JSON 输出。
    """
    cmd = [
        sys.executable,
        "-m",
        "scripts.infer_final",
        "--config",
        CONFIG_PATH,
        "--query",
        query,
    ]

    if safe_mode:
        cmd.append("--safe_mode")

    result = subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        check=False,
    )

    if result.returncode != 0:
        return {
            "error": True,
            "message": result.stderr or result.stdout or "推理失败，但没有捕获到错误信息。",
        }

    stdout = result.stdout.strip()

    try:
        return json.loads(stdout)
    except json.JSONDecodeError:
        return {
            "error": True,
            "message": "推理脚本没有输出合法 JSON。",
            "raw_stdout": stdout,
            "raw_stderr": result.stderr,
        }


def format_risk_block(result: Dict[str, Any]) -> str:
    """
    格式化风控结果。
    """
    if result.get("error"):
        return result.get("message", "Unknown error")

    lines = []

    lines.append(f"model_type: {result.get('model_type')}")
    lines.append(f"safe_fallback: {result.get('safe_fallback')}")
    lines.append("")

    pre_risk = result.get("pre_risk", {})
    retrieval_status = result.get("retrieval_status", {})
    post_risk = result.get("post_risk", {})

    lines.append("[pre_risk]")
    lines.append(json.dumps(pre_risk, ensure_ascii=False, indent=2))

    lines.append("")
    lines.append("[retrieval_status]")
    lines.append(json.dumps(retrieval_status, ensure_ascii=False, indent=2))

    lines.append("")
    lines.append("[post_risk]")
    lines.append(json.dumps(post_risk, ensure_ascii=False, indent=2))

    return "\n".join(lines)


def format_retrieved_docs(result: Dict[str, Any]) -> str:
    """
    格式化 RAG 检索证据。
    """
    if result.get("error"):
        return result.get("raw_stdout", "")

    docs: List[Dict[str, Any]] = result.get("retrieved_docs", []) or []

    if not docs:
        return "未返回 retrieved_docs。请确认 infer_rag.py 是否返回检索结果。"

    blocks = []

    for i, doc in enumerate(docs, start=1):
        doc_id = doc.get("doc_id", "")
        title = doc.get("title", "")
        category = doc.get("category", "")
        doc_type = doc.get("doc_type", "")
        score = doc.get("score", "")
        content = doc.get("content", "")

        block = (
            f"【Doc {i}】\n"
            f"doc_id: {doc_id}\n"
            f"title: {title}\n"
            f"category: {category}\n"
            f"doc_type: {doc_type}\n"
            f"score: {score}\n"
            f"content:\n{content}"
        )
        blocks.append(block)

    return "\n\n" + ("=" * 80 + "\n\n").join(blocks)


def format_raw_json(result: Dict[str, Any]) -> str:
    """
    输出完整 JSON，方便调试。
    """
    return json.dumps(result, ensure_ascii=False, indent=2)


def answer_fn(query: str, safe_mode: bool) -> Tuple[str, str, str, str]:
    """
    Gradio 主函数。
    """
    query = (query or "").strip()

    if not query:
        return "请输入用户问题。", "", "", ""

    result = run_infer_final(query=query, safe_mode=safe_mode)

    if result.get("error"):
        error_msg = result.get("message", "推理失败。")
        raw = format_raw_json(result)
        return error_msg, error_msg, raw, raw

    answer = result.get("answer", "")

    risk_text = format_risk_block(result)
    docs_text = format_retrieved_docs(result)
    raw_json = format_raw_json(result)

    return answer, risk_text, docs_text, raw_json


EXAMPLES = [
    ["我的快递三天没更新了怎么办？", True],
    ["你们物流太慢了，必须赔我100块钱。", True],
    ["订单号是123456789012345，帮我查一下退款什么时候到账。", True],
    ["我要投诉你们客服，还要去平台举报。", True],
    ["这个商品支持七天无理由退货吗？", True],
    ["这款耳机支持主动降噪吗？", True],
]


with gr.Blocks(title="电商客服问答大模型 Demo") as demo:
    gr.Markdown("# 电商客服问答大模型 Demo")
    gr.Markdown(
        "系统链路：用户问题 → RAG 检索 → DPO+RAG 模型生成 → Risk Controller 风控判断 → 最终回复"
    )

    with gr.Row():
        query_input = gr.Textbox(
            label="用户问题",
            placeholder="例如：我的快递三天没更新了怎么办？",
            lines=4,
        )

    with gr.Row():
        safe_mode_input = gr.Checkbox(
            label="启用 safe_mode：命中高风险或低可信检索时使用安全兜底回复",
            value=True,
        )

    submit_btn = gr.Button("生成客服回复", variant="primary")

    with gr.Tab("模型回复"):
        answer_output = gr.Textbox(
            label="最终回复",
            lines=10,
        )

    with gr.Tab("风控结果"):
        risk_output = gr.Textbox(
            label="Risk Controller 输出",
            lines=16,
        )

    with gr.Tab("RAG 检索证据"):
        docs_output = gr.Textbox(
            label="retrieved_docs",
            lines=18,
        )

    with gr.Tab("原始 JSON"):
        raw_json_output = gr.Textbox(
            label="完整推理结果 JSON",
            lines=20,
        )

    gr.Examples(
        examples=EXAMPLES,
        inputs=[query_input, safe_mode_input],
    )

    submit_btn.click(
        fn=answer_fn,
        inputs=[query_input, safe_mode_input],
        outputs=[answer_output, risk_output, docs_output, raw_json_output],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )