from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Any

import torch


@dataclass
class GenerationConfig:
    max_new_tokens: int = 512
    temperature: float = 0.3
    top_p: float = 0.9
    repetition_penalty: float = 1.05
    do_sample: bool = True


DEFAULT_SYSTEM_PROMPT = """你是电商平台客服助手。你需要礼貌、准确、简洁地回答用户问题。

回答要求：
1. 不要编造商品参数、订单状态、物流位置、退款金额、赔偿金额或平台政策；
2. 如果无法确认，应说明需要进一步核实；
3. 涉及具体订单状态、赔付、投诉升级、法律威胁等场景时，应建议转人工；
4. 回复中应包含用户可执行的下一步操作；
5. 不要使用“一定退款”“保证送达”“马上赔偿”等绝对化承诺。
"""


class ChatGenerator:
    def __init__(self, tokenizer, model, gen_cfg: GenerationConfig):
        self.tokenizer = tokenizer
        self.model = model
        self.gen_cfg = gen_cfg

    def build_messages(self, query: str, system_prompt: Optional[str] = None) -> List[Dict[str, str]]:
        return [
            {"role": "system", "content": system_prompt or DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]

    def generate_from_messages(self, messages: List[Dict[str, str]]) -> str:
        """
        适配 Qwen / ChatML 类 Instruct 模型。
        """
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.gen_cfg.max_new_tokens,
                temperature=self.gen_cfg.temperature,
                top_p=self.gen_cfg.top_p,
                repetition_penalty=self.gen_cfg.repetition_penalty,
                do_sample=self.gen_cfg.do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
        answer = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return answer.strip()

    def generate(self, query: str, system_prompt: Optional[str] = None) -> str:
        return self.generate_from_messages(self.build_messages(query, system_prompt=system_prompt))


def build_rag_messages(
    query: str,
    retrieved_docs: List[Dict[str, Any]],
    chat_history: Optional[str] = None,
) -> List[Dict[str, str]]:
    """
    Base + RAG 使用的 prompt。
    注意：这里不要求模型知道 doc_id；doc_id 只用于系统内部追踪和评测。
    """
    if not retrieved_docs:
        retrieved_context = "未检索到可用知识。"
    else:
        context_blocks = []
        for idx, doc in enumerate(retrieved_docs, start=1):
            title = doc.get("title", "")
            content = doc.get("content", "")
            doc_type = doc.get("doc_type", "")
            category = doc.get("category", "")
            context_blocks.append(
                f"【知识{idx}】\n"
                f"类型：{doc_type}\n"
                f"类别：{category}\n"
                f"标题：{title}\n"
                f"内容：{content}"
            )
        retrieved_context = "\n\n".join(context_blocks)

    prompt = f"""你是电商平台客服助手，请根据以下知识库内容回答用户问题。

【用户历史对话】
{chat_history or "无"}

【检索到的知识】
{retrieved_context}

【用户当前问题】
{query}

【回答要求】
- 使用中文回答；
- 优先基于检索到的知识回答；
- 不要编造知识库中没有的信息；
- 如果检索内容不足以支持回答，请说明需要进一步核实；
- 如涉及订单状态、赔付金额、投诉升级、法律威胁，请建议转人工；
- 如能解决，请给出明确操作步骤；
- 回复应先安抚，再说明规则，最后给出下一步；
- 不要使用“一定退款”“保证送达”“马上赔偿”等绝对化承诺。

请输出客服回复：
"""

    return [
        {"role": "system", "content": "你是电商平台客服助手，必须遵守电商客服合规边界。"},
        {"role": "user", "content": prompt},
    ]
