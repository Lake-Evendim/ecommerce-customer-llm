# 数据集质量验收报告

## 1. 总体结论：FAIL

- 错误数：2
- 警告数：1
- Issue 总数：3

## 2. 数量门槛检查

| 检查项 | 实际值 | 最低要求 | 是否通过 |
|---|---:|---:|---|
| sft_train_min | 879 | 1800 | FAIL |
| sft_valid_min | 119 | 200 | FAIL |
| eval_set_min | 200 | 200 | PASS |
| dpo_total_min | 300 | 300 | PASS |
| dpo_valid_min | 36 | 20 | PASS |
| kb_all_min | 89 | 80 | PASS |

## 3. SFT Train

- 样本数：879
- 格式有效数：879

### category_distribution

| 项 | 数量 |
|---|---:|
| complaint | 90 |
| coupon_price | 70 |
| invoice | 70 |
| logistics | 130 |
| manual_transfer | 80 |
| product_info | 129 |
| quality_issue | 130 |
| return_refund | 180 |

### source_distribution

| 项 | 数量 |
|---|---:|
| llm_user_rewrite | 879 |

## 4. SFT Valid

- 样本数：119
- 格式有效数：119

### category_distribution

| 项 | 数量 |
|---|---:|
| complaint | 10 |
| coupon_price | 10 |
| invoice | 10 |
| logistics | 20 |
| manual_transfer | 10 |
| product_info | 20 |
| quality_issue | 20 |
| return_refund | 19 |

### source_distribution

| 项 | 数量 |
|---|---:|
| llm_user_rewrite | 119 |

## 5. SFT Seed 泄漏检查

- train seed_id 数：88
- valid seed_id 数：12
- 泄漏数量：0

## 6. Eval Set

- 样本数：200
- 格式有效数：200

### category_distribution

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

### eval_type_distribution

| 项 | 数量 |
|---|---:|
| human_transfer | 26 |
| rag | 50 |
| safety | 45 |
| sft | 79 |

### difficulty_distribution

| 项 | 数量 |
|---|---:|
| easy | 55 |
| hard | 53 |
| medium | 92 |

### need_human_distribution

| 项 | 数量 |
|---|---:|
| False | 128 |
| True | 72 |

### need_rag_distribution

| 项 | 数量 |
|---|---:|
| False | 143 |
| True | 57 |

### risk_tag_distribution

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

## 7. DPO Candidates

- 样本数：300
- 格式有效数：300

### category_distribution

| 项 | 数量 |
|---|---:|
| complaint | 35 |
| coupon_price | 25 |
| invoice | 20 |
| logistics | 45 |
| manual_transfer | 30 |
| product_info | 40 |
| quality_issue | 45 |
| return_refund | 60 |

### source_distribution

| 项 | 数量 |
|---|---:|
| from_eval | 83 |
| from_sft | 217 |

### rejected_type_distribution

| 项 | 数量 |
|---|---:|
| compensation_commitment | 40 |
| fabricated_policy | 45 |
| fake_order_or_logistics_status | 40 |
| missing_action | 35 |
| no_human_transfer | 35 |
| over_marketing | 20 |
| rude_tone | 30 |
| unsafe_overpromise | 20 |
| wrong_intent | 35 |

### severity_distribution

| 项 | 数量 |
|---|---:|
| high | 180 |
| low | 20 |
| medium | 100 |

## 8. DPO Train

- 样本数：264
- 格式有效数：264

### category_distribution

| 项 | 数量 |
|---|---:|
| complaint | 31 |
| coupon_price | 23 |
| invoice | 15 |
| logistics | 39 |
| manual_transfer | 27 |
| product_info | 35 |
| quality_issue | 39 |
| return_refund | 55 |

### source_distribution

| 项 | 数量 |
|---|---:|
| from_eval | 70 |
| from_sft | 194 |

### rejected_type_distribution

| 项 | 数量 |
|---|---:|
| compensation_commitment | 36 |
| fabricated_policy | 40 |
| fake_order_or_logistics_status | 36 |
| missing_action | 30 |
| no_human_transfer | 31 |
| over_marketing | 18 |
| rude_tone | 26 |
| unsafe_overpromise | 17 |
| wrong_intent | 30 |

### severity_distribution

| 项 | 数量 |
|---|---:|
| high | 160 |
| low | 18 |
| medium | 86 |

## 9. DPO Valid

- 样本数：36
- 格式有效数：36

### category_distribution

| 项 | 数量 |
|---|---:|
| complaint | 4 |
| coupon_price | 2 |
| invoice | 5 |
| logistics | 6 |
| manual_transfer | 3 |
| product_info | 5 |
| quality_issue | 6 |
| return_refund | 5 |

### source_distribution

| 项 | 数量 |
|---|---:|
| from_eval | 13 |
| from_sft | 23 |

### rejected_type_distribution

| 项 | 数量 |
|---|---:|
| compensation_commitment | 4 |
| fabricated_policy | 5 |
| fake_order_or_logistics_status | 4 |
| missing_action | 5 |
| no_human_transfer | 4 |
| over_marketing | 2 |
| rude_tone | 4 |
| unsafe_overpromise | 3 |
| wrong_intent | 5 |

### severity_distribution

| 项 | 数量 |
|---|---:|
| high | 20 |
| low | 2 |
| medium | 14 |

## 10. RAG KB

- 样本数：89
- 格式有效数：89

### doc_type_distribution

| 项 | 数量 |
|---|---:|
| faq | 46 |
| product | 20 |
| sop | 23 |

### category_distribution

| 项 | 数量 |
|---|---:|
| complaint | 8 |
| coupon_price | 7 |
| invoice | 7 |
| logistics | 9 |
| manual_transfer | 7 |
| product_info | 28 |
| quality_issue | 10 |
| return_refund | 13 |

## 11. Eval 与 SFT 重复检查

- 精确重复数：0
- 近重复数：1
- 示例：
  - {'eval_id': 'eval_000096', 'query': '优惠券为什么不能用？', 'overlap_type': 'near', 'sft_dataset': 'sft_valid', 'sft_user': '优惠券为什么不能用了？'}

## 12. Issue 汇总

### 按严重程度

| 项 | 数量 |
|---|---:|
| error | 2 |
| warning | 1 |

### 按数据集

| 项 | 数量 |
|---|---:|
| cross_dataset | 1 |
| sft_train | 1 |
| sft_valid | 1 |

### 按问题类型

| 项 | 数量 |
|---|---:|
| below_minimum_count | 2 |
| eval_sft_near_overlap | 1 |

## 13. Issue 明细

| 严重程度 | 数据集 | 类型 | id | 信息 |
|---|---|---|---|---|
| warning | cross_dataset | eval_sft_near_overlap |  | 1 eval queries are near-duplicates of SFT user texts |
| error | sft_train | below_minimum_count |  | sft_train_min: actual=879, expected_min=1800 |
| error | sft_valid | below_minimum_count |  | sft_valid_min: actual=119, expected_min=200 |
