from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftModel
except Exception:  # pragma: no cover
    PeftModel = None


@dataclass
class ModelConfig:
    model_name_or_path: str
    adapter_path: Optional[str] = None
    trust_remote_code: bool = False
    torch_dtype: str = "auto"
    device_map: str = "auto"
    local_files_only: bool = True


def _resolve_torch_dtype(dtype: str):
    if dtype in (None, "auto"):
        return "auto"
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if dtype not in mapping:
        raise ValueError(f"Unsupported torch_dtype: {dtype}")
    return mapping[dtype]


def load_tokenizer_and_model(cfg: ModelConfig) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    """
    加载 Base 模型。后续如果你训练了 LoRA，只需要传 adapter_path。
    """
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name_or_path,
        trust_remote_code=cfg.trust_remote_code,
        use_fast=False,
        local_files_only=cfg.local_files_only,
    )

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name_or_path,
        trust_remote_code=cfg.trust_remote_code,
        torch_dtype=_resolve_torch_dtype(cfg.torch_dtype),
        device_map=cfg.device_map,
        local_files_only=cfg.local_files_only,
    )

    if cfg.adapter_path:
        if PeftModel is None:
            raise ImportError("peft is required when adapter_path is provided.")
        model = PeftModel.from_pretrained(
                    model,
                    cfg.adapter_path,
                    local_files_only=cfg.local_files_only,
                )

    model.eval()
    return tokenizer, model
