import os
import json
import yaml
import torch
from datasets import Dataset
from transformers import AutoTokenizer, BitsAndBytesConfig
from peft import AutoPeftModelForCausalLM
from trl import DPOConfig, DPOTrainer


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            x = json.loads(line)

            # DPOTrainer 需要 prompt / chosen / rejected
            if "prompt" not in x or "chosen" not in x or "rejected" not in x:
                raise ValueError(f"{path}:{line_num} missing prompt/chosen/rejected")

            rows.append({
                "prompt": x["prompt"],
                "chosen": x["chosen"],
                "rejected": x["rejected"],
                # 保留元信息，便于后面 debug；DPOTrainer 会忽略无关列
                "rejected_type": x.get("rejected_type", ""),
                "preference_reason": x.get("preference_reason", ""),
            })

    return Dataset.from_list(rows)


def main():
    cfg = load_yaml("configs/dpo.yaml")

    os.makedirs(cfg["output_dir"], exist_ok=True)
    os.makedirs(cfg["log_dir"], exist_ok=True)

    sft_adapter_path = cfg["sft_adapter_path"]

    print("Loading tokenizer from:", sft_adapter_path)
    tokenizer = AutoTokenizer.from_pretrained(
        sft_adapter_path,
        trust_remote_code=True,
        use_fast=False,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    print("Loading SFT adapter as trainable policy model:", sft_adapter_path)
    model = AutoPeftModelForCausalLM.from_pretrained(
        sft_adapter_path,
        is_trainable=True,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
    )

    model.config.use_cache = False

    if cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    train_dataset = load_jsonl(cfg["train_file"])
    eval_dataset = load_jsonl(cfg["valid_file"])

    print("train size:", len(train_dataset))
    print("eval size:", len(eval_dataset))

    args = DPOConfig(
        output_dir=cfg["output_dir"],
        logging_dir=cfg["log_dir"],

        num_train_epochs=cfg["num_train_epochs"],
        max_steps=cfg.get("max_steps", -1),

        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],

        learning_rate=cfg["learning_rate"],
        warmup_ratio=cfg["warmup_ratio"],
        weight_decay=cfg["weight_decay"],
        beta=cfg["beta"],

        max_length=cfg["max_length"],

        logging_steps=cfg["logging_steps"],
        eval_strategy="steps",
        eval_steps=cfg["eval_steps"],
        save_strategy="steps",
        save_steps=cfg["save_steps"],
        save_total_limit=cfg["save_total_limit"],

        bf16=cfg.get("bf16", True),
        fp16=False,

        gradient_checkpointing=cfg.get("gradient_checkpointing", True),
        remove_unused_columns=False,
        report_to="none",
        optim="paged_adamw_8bit",
        seed=cfg.get("seed", 42),
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    trainer.train()

    trainer.save_model(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])

    print("DPO adapter saved to:", cfg["output_dir"])


if __name__ == "__main__":
    main()
