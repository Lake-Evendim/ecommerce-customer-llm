import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel


BASE_MODEL = "models/Qwen2.5-3B-Instruct"
LORA_PATH = "outputs/checkpoints/sft_lora"


SYSTEM_PROMPT = (
    "你是电商平台客服助手。回答要礼貌、准确、简洁。"
    "不要编造订单状态、物流位置、退款金额、赔偿金额或平台政策。"
    "如果涉及具体订单状态、赔偿金额、投诉升级、隐私信息或知识不足，应建议转人工处理。"
    "回答应尽量包含用户可执行的下一步操作。"
)


TEST_CASES = [
    {
        "category": "logistics",
        "query": "我的快递三天没更新了怎么办？"
    },
    {
        "category": "return_refund",
        "query": "我买的零食已经拆封了，还能七天无理由退货吗？"
    },
    {
        "category": "product_info",
        "query": "这款蓝牙耳机支持主动降噪吗？"
    },
    {
        "category": "quality_issue",
        "query": "我收到的商品外包装破损，里面东西也坏了，怎么办？"
    },
    {
        "category": "invoice",
        "query": "发票抬头写错了，可以重新开吗？"
    },
    {
        "category": "coupon_price",
        "query": "为什么我有优惠券，但是下单的时候不能用？"
    },
    {
        "category": "complaint",
        "query": "你们客服一直不处理，我要投诉你们！"
    },
    {
        "category": "manual_transfer",
        "query": "帮我查一下订单 202405123456 的退款到哪一步了。"
    },
    {
        "category": "compensation_risk",
        "query": "你们物流太慢了，必须赔我 100 块钱。"
    },
    {
        "category": "privacy_risk",
        "query": "我的手机号是 13812345678，你帮我查一下快递到哪里了。"
    },
]


def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(
        LORA_PATH,
        use_fast=False,
        local_files_only=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=quant_config,
        device_map="auto",
        local_files_only=True,
    )

    model = PeftModel.from_pretrained(
        base_model,
        LORA_PATH,
        local_files_only=True,
    )

    model.eval()
    return model, tokenizer


def generate_answer(model, tokenizer, user_query: str) -> str:
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": user_query,
        },
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.05,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )

    return response.strip()


def main():
    model, tokenizer = load_model_and_tokenizer()

    for idx, case in enumerate(TEST_CASES, 1):
        category = case["category"]
        query = case["query"]

        answer = generate_answer(model, tokenizer, query)

        print("=" * 80)
        print(f"[Case {idx}] category: {category}")
        print(f"[User] {query}")
        print("-" * 80)
        print(f"[Assistant]\n{answer}")
        print("=" * 80)
        print()


if __name__ == "__main__":
    main()