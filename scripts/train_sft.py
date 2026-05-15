import os
import json
import yaml
import torch
from dataclasses import dataclass
from typing import Dict, List, Any

from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)


IGNORE_INDEX = -100


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class SFTDataset(Dataset):
    def __init__(self, jsonl_path: str, tokenizer, max_seq_length: int):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                self.samples.append(json.loads(line))

    def __len__(self):
        return len(self.samples)

    def _build_prompt_and_answer(self, messages: List[Dict[str, str]]):
        """
        将一条 messages 样本拆成：
        prompt_messages: system + user
        answer_message: assistant
        """
        if len(messages) < 2:
            raise ValueError("messages too short")

        if messages[-1]["role"] != "assistant":
            raise ValueError("last message must be assistant")

        prompt_messages = messages[:-1]
        answer_message = messages[-1]

        return prompt_messages, answer_message

    def __getitem__(self, idx):
        item = self.samples[idx]
        messages = item["messages"]

        prompt_messages, answer_message = self._build_prompt_and_answer(messages)

        # prompt 部分：system + user，并添加 generation prompt
        prompt_text = self.tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # assistant answer 部分：只取 assistant content，再补 eos
        answer_text = answer_message["content"] + self.tokenizer.eos_token

        prompt_ids = self.tokenizer(
            prompt_text,
            add_special_tokens=False,
        )["input_ids"]

        answer_ids = self.tokenizer(
            answer_text,
            add_special_tokens=False,
        )["input_ids"]

        input_ids = prompt_ids + answer_ids

        # prompt 部分不算 loss，answer 部分算 loss
        labels = [IGNORE_INDEX] * len(prompt_ids) + answer_ids.copy()

        # 截断
        input_ids = input_ids[: self.max_seq_length]
        labels = labels[: self.max_seq_length]

        attention_mask = [1] * len(input_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def find_all_linear_names(model):
    """
    找到 4bit 量化模型中适合加 LoRA 的 Linear 层。
    Qwen 常见目标层：
    q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
    """
    target_modules = set()

    for name, module in model.named_modules():
        if any(key in name for key in [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]):
            target_modules.add(name.split(".")[-1])

    return sorted(list(target_modules))


def main():
    config_path = "configs/sft_lora.yaml"
    cfg = load_yaml(config_path)

    os.makedirs(cfg["output_dir"], exist_ok=True)
    os.makedirs(cfg["log_dir"], exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model_name_or_path"],
        trust_remote_code=True,
        use_fast=False,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name_or_path"],
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
    )

    model.config.use_cache = False

    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=cfg.get("gradient_checkpointing", True),
    )

    target_modules = find_all_linear_names(model)
    print("LoRA target modules:", target_modules)

    lora_config = LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_dataset = SFTDataset(
        cfg["train_file"],
        tokenizer,
        cfg["max_seq_length"],
    )

    eval_dataset = SFTDataset(
        cfg["valid_file"],
        tokenizer,
        cfg["max_seq_length"],
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=IGNORE_INDEX,
        return_tensors="pt",
    )

    training_args = TrainingArguments(
        output_dir=cfg["output_dir"],
        logging_dir=cfg["log_dir"],

        num_train_epochs=cfg["num_train_epochs"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],

        learning_rate=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
        warmup_ratio=cfg["warmup_ratio"],

        logging_steps=cfg["logging_steps"],
        eval_strategy="steps",
        eval_steps=cfg["eval_steps"],

        save_strategy="steps",
        save_steps=cfg["save_steps"],
        save_total_limit=cfg["save_total_limit"],

        bf16=cfg.get("bf16", True),
        fp16=False,

        gradient_checkpointing=cfg.get("gradient_checkpointing", True),
        report_to="none",

        remove_unused_columns=False,
        optim="paged_adamw_8bit",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    trainer.train()

    trainer.save_model(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])

    print(f"SFT LoRA adapter saved to: {cfg['output_dir']}")


if __name__ == "__main__":
    main()