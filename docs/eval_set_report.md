# 离线评测集构建报告

## 1. 总览

- 计划样本数：200
- 生成保留样本数：200
- 删除样本数：277
- 输出文件：`/home/amlab/桌面/shenzy/ecommerce-customer-llm/data/eval/eval_set.jsonl`

## 2. 删除原因

- duplicate_query: 268
- need_rag_but_no_expected_docs: 6
- rag_docs_not_relevant: 2
- empty_must_not_include: 1

## 类别分布

| 项 | 数量 |
|---|---:|
| complaint | 15 |
| coupon_price | 10 |
| invoice | 24 |
| logistics | 29 |
| manual_transfer | 19 |
| product_info | 35 |
| quality_issue | 32 |
| return_refund | 36 |

## 评测类型分布

| 项 | 数量 |
|---|---:|
| human_transfer | 26 |
| rag | 50 |
| safety | 45 |
| sft | 79 |

## 难度分布

| 项 | 数量 |
|---|---:|
| easy | 55 |
| hard | 53 |
| medium | 92 |

## need_human 分布

| 项 | 数量 |
|---|---:|
| False | 128 |
| True | 72 |

## need_rag 分布

| 项 | 数量 |
|---|---:|
| False | 143 |
| True | 57 |

## 风险标签分布

| 项 | 数量 |
|---|---:|
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

