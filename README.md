# Ecommerce Customer LLM

面向电商客服场景的中文大模型项目，包含数据构建、SFT 微调、DPO 偏好优化、RAG 知识库检索、风险控制、自动评测和 Gradio 演示界面。

项目默认使用本地模型路径：

- 基座模型：`models/Qwen2.5-3B-Instruct`
- 向量模型：`models/bge-small-zh-v1.5`

如模型路径不同，请先修改 `configs/*.yaml` 中的 `model_name_or_path`、`adapter_path` 和 embedding 配置。

## 功能特性

- 电商客服多轮问答数据清洗与划分
- FAQ、SOP、商品信息知识库构建
- 基于 FAISS 的向量检索
- QLoRA SFT 指令微调
- DPO 偏好优化
- Base / SFT / DPO / RAG 多链路推理
- 检索与生成效果评测
- 风控前置检查、后置检查和安全兜底回复
- Gradio 可视化 Demo

## 项目结构

```text
.
├── configs/                 # 训练、推理、RAG 配置
├── data/
│   ├── raw/                 # 原始种子数据
│   ├── interim/             # 中间数据
│   ├── processed/           # SFT / DPO 训练数据
│   ├── knowledge_base/      # RAG 知识库
│   └── eval/                # 评测集
├── demo/                    # Gradio Demo
├── docs/                    # 数据报告和质量报告
├── models/                  # 本地基座模型和 embedding 模型
├── outputs/
│   ├── checkpoints/         # LoRA adapter 输出
│   ├── eval_results/        # 评测结果
│   ├── badcases/            # badcase 样本
│   └── vector_db/           # FAISS 索引和文档库
├── scripts/                 # 数据、训练、推理、评测脚本
└── src/                     # 核心模块
```

## 环境准备

建议使用 Python 3.10+，并准备支持 CUDA 的 PyTorch 环境。

```bash
conda create -n ecommerce-llm python=3.10 -y
conda activate ecommerce-llm
```

安装常用依赖：

```bash
pip install torch transformers datasets accelerate peft trl bitsandbytes
pip install sentence-transformers faiss-cpu pyyaml gradio scikit-learn pandas tqdm
```

如果使用 GPU 版本 FAISS，可按本机 CUDA 环境安装 `faiss-gpu`。

## 数据处理

生成数据质量报告：

```bash
python -m scripts.build_data_report
python -m scripts.check_dataset
```

清洗并划分 SFT 数据：

```bash
python -m scripts.clean_sft_data
python -m scripts.split_sft_dataset
```

构建评测集和 DPO 数据时，可使用对应脚本：

```bash
python -m scripts.build_eval_set_with_llm
python -m scripts.build_dpo_data_with_llm
```

## 构建 RAG 知识库

从原始 FAQ、SOP、商品信息构建知识库：

```bash
python -m scripts.build_kb
```

构建 FAISS 向量库：

```bash
python -m scripts.build_vector_db --config configs/rag.yaml
```

测试检索：

```bash
python -m scripts.query_retrieval --query "我的快递三天没更新了怎么办？"
```

评测检索效果：

```bash
python -m scripts.eval_retrieval --config configs/rag.yaml
```

评测结果会写入：

- `outputs/eval_results/rag_retrieval_eval.jsonl`
- `outputs/eval_results/rag_retrieval_summary.md`
- `outputs/badcases/rag_retrieval_badcases.jsonl`

## SFT 训练

训练配置位于 `configs/sft_lora.yaml`。

```bash
python -m scripts.train_sft
```

默认输出目录：

```text
outputs/checkpoints/sft_lora
```

## DPO 训练

DPO 配置位于 `configs/dpo.yaml`。训练前请确认：

- `sft_adapter_path` 指向已训练好的 SFT adapter
- `data/processed/dpo_train.jsonl` 和 `data/processed/dpo_valid.jsonl` 已存在

```bash
python -m scripts.train_dpo
```

默认输出目录：

```text
outputs/checkpoints/dpo_lora
```

## 推理

Base 模型推理：

```bash
python -m scripts.infer_base --query "这个商品支持七天无理由退货吗？"
```

SFT 模型推理：

```bash
python -m scripts.infer_sft
```

DPO 模型推理：

```bash
python -m scripts.infer_dpo
```

`infer_sft.py` 和 `infer_dpo.py` 会运行脚本内置的客服测试样例。

RAG 推理：

```bash
python -m scripts.infer_rag \
  --config configs/infer_final.yaml \
  --model_type dpo_rag \
  --query "这款耳机支持主动降噪吗？"
```

最终链路推理，包含 RAG、DPO adapter 和 safe mode：

```bash
python -m scripts.infer_final \
  --config configs/infer_final.yaml \
  --query "我的快递三天没更新了怎么办？" \
  --safe_mode
```

## 评测

生成质量评测：

```bash
python -m scripts.eval_generation --config configs/infer.yaml --model_type base
python -m scripts.eval_generation --config configs/infer_sft.yaml --model_type sft
python -m scripts.eval_generation --config configs/infer_dpo.yaml --model_type dpo
python -m scripts.eval_generation --config configs/infer_final.yaml --model_type dpo_rag --safe_mode
```

对比不同模型结果：

```bash
python -m scripts.compare_generation_eval
```

主要输出目录：

```text
outputs/eval_results/
outputs/badcases/
```

## 启动 Demo

```bash
python demo/gradio_app.py
```

启动后在浏览器访问 Gradio 输出的本地地址。Demo 会调用 `scripts.infer_final`，默认配置文件为 `configs/infer_final.yaml`。

## 常用配置

- `configs/rag.yaml`：RAG 知识库、embedding、检索参数
- `configs/sft_lora.yaml`：SFT QLoRA 训练参数
- `configs/dpo.yaml`：DPO 训练参数
- `configs/infer.yaml`：Base 推理配置
- `configs/infer_sft.yaml`：SFT 推理配置
- `configs/infer_dpo.yaml`：DPO 推理配置
- `configs/infer_final.yaml`：最终 DPO + RAG + 风控推理配置

## 注意事项

- 当前配置使用本地模型路径，迁移环境后需要同步修改配置文件。
- SFT 和 DPO 训练默认启用 4bit 量化与 bf16，请确认显卡和驱动环境支持。
- `outputs/vector_db/` 中的 FAISS 索引需要与 `data/knowledge_base/kb_all.jsonl` 和 embedding 模型保持一致。
- Demo 和最终推理依赖 DPO adapter、向量库和知识库，首次运行前请先完成 SFT/DPO 训练与 RAG 向量库构建。
