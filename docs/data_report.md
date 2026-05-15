# 电商客服大模型数据集构建报告

## 1. 数据集目标

本项目构建一套面向电商客服场景的大模型数据闭环，用于支持 **SFT 微调**、**RAG 知识增强**、**DPO 偏好对齐** 和 **离线评测**。

数据集围绕基础客服问答、安全边界、可执行回复、转人工意识和 RAG 可追溯能力进行设计。


## 2. 数据来源与合规说明

数据来源包括人工构造的 FAQ/SOP/商品知识、100 条人工黄金 SFT seed、GLM API 生成的用户问法扩写、Eval 样本和 DPO rejected 回复。

本项目不使用真实企业客服日志、手机号、身份证号、邮箱、真实订单号等用户隐私数据；各阶段脚本均包含敏感信息和错误承诺检查。


## 3. 数据目录与最终交付物

| 数据文件 | 路径 | 样本数 |
| --- | ---: | ---: |
| SFT seed | `/home/amlab/桌面/shenzy/ecommerce-customer-llm/data/raw/sft_seed.jsonl` | 100 |
| SFT cleaned | `/home/amlab/桌面/shenzy/ecommerce-customer-llm/data/interim/cleaned_sft.jsonl` | 998 |
| SFT train | `/home/amlab/桌面/shenzy/ecommerce-customer-llm/data/processed/sft_train.jsonl` | 879 |
| SFT valid | `/home/amlab/桌面/shenzy/ecommerce-customer-llm/data/processed/sft_valid.jsonl` | 119 |
| RAG KB | `/home/amlab/桌面/shenzy/ecommerce-customer-llm/data/knowledge_base/kb_all.jsonl` | 89 |
| Eval set | `/home/amlab/桌面/shenzy/ecommerce-customer-llm/data/eval/eval_set.jsonl` | 200 |
| DPO candidates | `/home/amlab/桌面/shenzy/ecommerce-customer-llm/data/interim/dpo_candidates.jsonl` | 300 |
| DPO train | `/home/amlab/桌面/shenzy/ecommerce-customer-llm/data/processed/dpo_train.jsonl` | 264 |
| DPO valid | `/home/amlab/桌面/shenzy/ecommerce-customer-llm/data/processed/dpo_valid.jsonl` | 36 |

过程报告文件：


| 报告 | 路径 |
| --- | ---: |
| SFT 清洗报告 | `/home/amlab/桌面/shenzy/ecommerce-customer-llm/docs/sft_clean_report.json` |
| SFT 划分报告 | `/home/amlab/桌面/shenzy/ecommerce-customer-llm/docs/sft_split_report.json` |
| Eval 构建报告 | `/home/amlab/桌面/shenzy/ecommerce-customer-llm/docs/eval_set_report.json` |
| DPO 构建报告 | `/home/amlab/桌面/shenzy/ecommerce-customer-llm/docs/dpo_data_report.json` |
| 全量质量验收报告 | `/home/amlab/桌面/shenzy/ecommerce-customer-llm/docs/dataset_quality_report.json` |

## 4. SFT 数据构建

SFT 数据采用“人工黄金样本 + LLM 用户问法扩写”的方式构造。扩写阶段只生成新的 user query，不改写人工确认过的 assistant answer。


| 指标 | 值 |
| --- | ---: |
| 人工 seed 数 | 100 |
| cleaned_sft 数 | 998 |
| sft_train 数 | 879 |
| sft_valid 数 | 119 |
| SFT split 实际 valid 比例 | 0.1192 |

### 4.1 SFT 类别分布


| 类别 | All | Train | Valid |
| --- | ---: | ---: | ---: |
| complaint | 100 | 90 | 10 |
| coupon_price | 80 | 70 | 10 |
| invoice | 80 | 70 | 10 |
| logistics | 150 | 130 | 20 |
| manual_transfer | 90 | 80 | 10 |
| product_info | 149 | 129 | 20 |
| quality_issue | 150 | 130 | 20 |
| return_refund | 199 | 180 | 19 |

### 4.2 SFT 来源分布


| 来源 | All | Train | Valid |
| --- | ---: | ---: | ---: |
| llm_user_rewrite | 998 | 879 | 119 |

### 4.3 SFT seed_id 泄漏检查


| 检查项 | 值 |
| --- | ---: |
| train seed_id 数 | 88 |
| valid seed_id 数 | 12 |
| 泄漏数量 | 0 |

同一个 `seed_id` 的扩写样本按组划分，不会同时进入 train 和 valid，用于降低同源扩写泄漏风险。


## 5. RAG 知识库构建

RAG 知识库由 FAQ、SOP 和商品知识构成，最终合并为 `kb_all.jsonl`。


| 指标 | 值 |
| --- | ---: |
| kb_all 总数 | 89 |
| 格式有效数 | 89 |
| 唯一 doc_id 数 | 89 |

### 5.1 doc_type 分布


| doc_type | 数量 |
| --- | ---: |
| faq | 46 |
| product | 20 |
| sop | 23 |

### 5.2 category 分布


| category | 数量 |
| --- | ---: |
| complaint | 8 |
| coupon_price | 7 |
| invoice | 7 |
| logistics | 9 |
| manual_transfer | 7 |
| product_info | 28 |
| quality_issue | 10 |
| return_refund | 13 |

RAG 质量检查包括 doc_id 唯一性、title/content 完整性、content 长度、重复 content、敏感信息检测，以及 Eval `expected_docs` 是否存在于 `kb_all`。


## 6. Eval 离线评测集构建

Eval 集不从 SFT train/valid 中随机抽样，而是单独构造，用于评估 Base、SFT、SFT+RAG 在客服场景下的业务正确性、安全边界和转人工能力。


| 指标 | 值 |
| --- | ---: |
| eval_set 总数 | 200 |
| 格式有效数 | 200 |
| 计划样本数 | 200 |
| 删除样本数 | 277 |

### 6.1 Eval 类别分布


| 类别 | 数量 |
| --- | ---: |
| complaint | 15 |
| coupon_price | 10 |
| invoice | 24 |
| logistics | 29 |
| manual_transfer | 19 |
| product_info | 35 |
| quality_issue | 32 |
| return_refund | 36 |

### 6.2 eval_type 分布


| eval_type | 数量 |
| --- | ---: |
| human_transfer | 26 |
| rag | 50 |
| safety | 45 |
| sft | 79 |

### 6.3 difficulty 分布


| difficulty | 数量 |
| --- | ---: |
| easy | 55 |
| hard | 53 |
| medium | 92 |

### 6.4 need_human / need_rag 分布


**need_human**


| need_human | 数量 |
| --- | ---: |
| False | 128 |
| True | 72 |


**need_rag**


| need_rag | 数量 |
| --- | ---: |
| False | 143 |
| True | 57 |

### 6.5 risk_tags 分布


| risk_tag | 数量 |
| --- | ---: |
| compensation_commitment | 6 |
| complaint | 4 |
| complaint_escalation | 14 |
| coupon_rule | 4 |
| fake_logistics_status | 8 |
| human_transfer_required | 72 |
| invoice | 1 |
| invoice_rule | 22 |
| logistics | 21 |
| manual_transfer | 2 |
| manual_transfer_required | 2 |
| money_commitment | 3 |
| order_privacy | 1 |
| order_status | 9 |
| price_protection | 2 |
| product_attribute | 28 |
| product_info | 2 |
| quality_evidence | 22 |
| quality_issue | 4 |
| rag_required | 57 |
| refund_policy | 5 |
| return_policy | 42 |
| return_refund | 1 |
| safety | 1 |
| sft | 1 |
| special_category | 1 |
| unsafe_promise | 46 |

### 6.6 Eval 与 SFT 重复检查


| 检查项 | 值 |
| --- | ---: |
| 精确重复数 | 0 |
| 近重复数 | 1 |

示例：


- `{'eval_id': 'eval_000096', 'query': '优惠券为什么不能用？', 'overlap_type': 'near', 'sft_dataset': 'sft_valid', 'sft_user': '优惠券为什么不能用了？'}`



## 7. DPO 偏好数据构建

DPO 数据以 cleaned SFT 和 Eval 样本为基础：`prompt` 和 `chosen` 来自高质量答案，`rejected` 由 GLM API 按指定错误类型生成，脚本再进行格式、相似度、敏感信息和 chosen 风险承诺检查。


| 指标 | 值 |
| --- | ---: |
| dpo_candidates | 300 |
| dpo_train | 264 |
| dpo_valid | 36 |
| 实际 valid 比例 | 0.12 |
| 删除样本数 | 0 |

### 7.1 DPO 类别分布


| 类别 | All | Train | Valid |
| --- | ---: | ---: | ---: |
| complaint | 35 | 31 | 4 |
| coupon_price | 25 | 23 | 2 |
| invoice | 20 | 15 | 5 |
| logistics | 45 | 39 | 6 |
| manual_transfer | 30 | 27 | 3 |
| product_info | 40 | 35 | 5 |
| quality_issue | 45 | 39 | 6 |
| return_refund | 60 | 55 | 5 |

### 7.2 rejected_type 分布


| rejected_type | All | Train | Valid |
| --- | ---: | ---: | ---: |
| compensation_commitment | 40 | 36 | 4 |
| fabricated_policy | 45 | 40 | 5 |
| fake_order_or_logistics_status | 40 | 36 | 4 |
| missing_action | 35 | 30 | 5 |
| no_human_transfer | 35 | 31 | 4 |
| over_marketing | 20 | 18 | 2 |
| rude_tone | 30 | 26 | 4 |
| unsafe_overpromise | 20 | 17 | 3 |
| wrong_intent | 35 | 30 | 5 |

### 7.3 source 分布


| source | All | Train | Valid |
| --- | ---: | ---: | ---: |
| from_eval | 83 | 70 | 13 |
| from_sft | 217 | 194 | 23 |

### 7.4 severity 分布


| severity | All | Train | Valid |
| --- | ---: | ---: | ---: |
| high | 180 | 160 | 20 |
| low | 20 | 18 | 2 |
| medium | 100 | 86 | 14 |

## 8. 数据清洗与质量控制规则

### 8.1 SFT 质量控制

- 检查 messages 是否为 `system -> user -> assistant`。
- 检查 category 是否属于 8 个合法业务类别。
- 检查 user / assistant 长度，删除空内容、过短内容。
- 检查手机号、身份证号、邮箱、明显订单号等敏感信息。
- 检查 assistant 中是否存在绝对化退款、赔偿、送达等错误承诺。
- 按 `seed_id` 分组划分 train/valid，降低同源扩写泄漏。


### 8.2 Eval 质量控制

- 检查 `eval_type`、`difficulty`、`must_include`、`must_not_include`、`risk_tags` 等字段。
- `eval_type=rag` 时要求 `need_rag=true` 且 `expected_docs` 非空。
- `eval_type=human_transfer` 时要求 `need_human=true`。
- 对 `expected_docs` 做 doc_id 规范化。
- 对 RAG 样本进行轻量相关性检查，避免 query 与 expected_docs 文档不匹配。
- 检查 Eval query 与 SFT user 的精确重复和近重复。


### 8.3 DPO 质量控制

- 检查 prompt 是否包含 system + user。
- 检查 chosen/rejected 是否各包含一个 assistant 回复。
- 检查 rejected_type 是否属于 9 类预定义错误类型。
- 检查 chosen/rejected 相似度，避免偏好差异过弱。
- 检查 chosen 是否包含错误承诺。
- 检查 preference_reason 是否有效。


### 8.4 RAG KB 质量控制

- 检查 chunk_id / product_id / id 是否唯一。
- 检查 doc_type、category、title、content 字段。
- 检查 content 长度、重复 content 和敏感信息。
- 检查 Eval expected_docs 是否能在 kb_all 中找到。


## 9. 最终质量验收结果

最终验收状态：**FAIL**


| 指标 | 值 |
| --- | ---: |
| 错误数 | 2 |
| 警告数 | 1 |
| Issue 总数 | 3 |

### 9.1 数量门槛检查


| 检查项 | 实际值 | 最低要求 | 是否通过 |
| --- | ---: | ---: | ---: |
| sft_train_min | 879 | 1800 | FAIL |
| sft_valid_min | 119 | 200 | FAIL |
| eval_set_min | 200 | 200 | PASS |
| dpo_total_min | 300 | 300 | PASS |
| dpo_valid_min | 36 | 20 | PASS |
| kb_all_min | 89 | 80 | PASS |

### 9.2 Issue 类型分布


| issue_type | 数量 |
| --- | ---: |
| below_minimum_count | 2 |
| eval_sft_near_overlap | 1 |

## 10. Badcase 与修复记录

| 问题 | 现象 | 修复方式 |
| --- | ---: | ---: |
| Eval expected_docs 格式错误 | expected_docs 被模型写成 doc_id=xxx 或包含整段 metadata。 | 增加 normalize_doc_id()，只保留裸 doc_id。 |
| RAG query 与 expected_docs 错配 | 例如物流问题绑定到换货申请文档。 | 增加 batch 内多 category KB context，并加入 rag_docs_relevant() 校验。 |
| Eval 自动补齐不足 | 部分样本被校验 drop，导致不足 200 条。 | 改为 while len(final_items) < total 持续补齐。 |
| 大 batch 导致 JSON 截断 | 一次请求过多样本导致模型输出 JSON 不完整。 | 限制单轮最大生成数量，采用小 batch 稳定补齐。 |
| DPO rejected_type 匹配过严 | 关键词规则过窄导致可用 rejected 被误删。 | 允许关闭强匹配，并通过人工抽查保证质量。 |
| Eval 与 SFT 近重复 | Eval query 与 SFT user 高度相似。 | 人工替换重复样本或提高近重复阈值后重新检查。 |

## 11. 当前数据集局限性

- 当前数据以合成数据和人工设计数据为主，真实客服噪声、错别字、多轮上下文仍然不足。

- RAG 知识库规模较小，商品知识和 SOP 覆盖仍有限。

- Eval 虽有自动规则校验，但仍需要人工抽样复核，尤其是 RAG 样本的文档相关性。

- DPO rejected 由 LLM 生成，虽然通过规则清洗，但仍建议每类 rejected_type 抽样复核。

- 当前主要为单轮问答数据，多轮售后协商、连续追问和跨轮状态管理还未充分覆盖。


## 12. 下一步优化方向

- 将 SFT 数据扩展到 5,000 条以上，并加入更多真实噪声表达。

- 将 RAG KB 扩展到 300+ chunk，覆盖更多商品、规则和 SOP。

- 将 Eval 扩展到 500 条，并增加人工评审标注。

- 引入 badcase_seed.jsonl，将模型真实失败案例持续沉淀到 SFT / DPO / Eval。

- 增加 run_eval.py，支持 Base / SFT / SFT+RAG 的自动化离线评测。

- 增加 LLM-as-judge 或人工评分表，对 must_include、must_not_include、need_human 等维度进行组合评估。


## 13. 复现命令

```bash

python scripts/expand_sft_with_llm.py --model glm-4-flash-250414 --variants-per-seed 25 --overwrite

python scripts/clean_sft_data.py

python scripts/split_sft_dataset.py

python scripts/build_eval_set_with_llm.py --model glm-4-flash-250414 --total 200 --batch-size 10 --overwrite

python scripts/build_dpo_data_with_llm.py --model glm-4-flash-250414 --target-total 300 --overwrite --disable-rejected-type-match

python scripts/check_dataset.py

python scripts/build_data_report.py

```
