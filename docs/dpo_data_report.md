# DPO 数据构建报告

## 1. 总览

- 计划样本数：300
- 候选样本数：300
- 训练集样本数：264
- 验证集样本数：36
- 实际验证集比例：0.12
- 删除样本数：0

## 2. 删除原因

- 无

## 3. 类别分布

| 项 | All | Train | Valid |
|---|---:|---:|---:|
| complaint | 35 | 31 | 4 |
| coupon_price | 25 | 23 | 2 |
| invoice | 20 | 15 | 5 |
| logistics | 45 | 39 | 6 |
| manual_transfer | 30 | 27 | 3 |
| product_info | 40 | 35 | 5 |
| quality_issue | 45 | 39 | 6 |
| return_refund | 60 | 55 | 5 |

## 4. rejected_type 分布

| 项 | All | Train | Valid |
|---|---:|---:|---:|
| compensation_commitment | 40 | 36 | 4 |
| fabricated_policy | 45 | 40 | 5 |
| fake_order_or_logistics_status | 40 | 36 | 4 |
| missing_action | 35 | 30 | 5 |
| no_human_transfer | 35 | 31 | 4 |
| over_marketing | 20 | 18 | 2 |
| rude_tone | 30 | 26 | 4 |
| unsafe_overpromise | 20 | 17 | 3 |
| wrong_intent | 35 | 30 | 5 |

## 5. 来源分布

| 项 | All | Train | Valid |
|---|---:|---:|---:|
| from_eval | 83 | 70 | 13 |
| from_sft | 217 | 194 | 23 |

## 6. 严重程度分布

| 项 | All | Train | Valid |
|---|---:|---:|---:|
| high | 180 | 160 | 20 |
| low | 20 | 18 | 2 |
| medium | 100 | 86 | 14 |

## 7. 数据格式说明

每条 DPO 样本包含 `prompt`、`chosen`、`rejected`、`rejected_type`、`severity` 和 `preference_reason`。prompt/chosen 来自清洗后的 SFT 或离线评测集，rejected 由 GLM API 按指定错误类型生成。
