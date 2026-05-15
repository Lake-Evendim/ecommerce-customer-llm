# SFT Train / Valid 划分报告

## 1. 配置

- 输入文件：`/home/amlab/桌面/shenzy/ecommerce-customer-llm/data/interim/cleaned_sft.jsonl`
- 训练集输出：`/home/amlab/桌面/shenzy/ecommerce-customer-llm/data/processed/sft_train.jsonl`
- 验证集输出：`/home/amlab/桌面/shenzy/ecommerce-customer-llm/data/processed/sft_valid.jsonl`
- 验证集比例：0.1
- 随机种子：42
- 划分策略：`stratified_group_split_by_category_and_seed_id`

## 2. 总览

- 原始样本数：998
- 格式有效样本数：998
- 格式无效样本数：0
- 分组总数：100
- 训练集样本数：879
- 验证集样本数：119
- 训练集 group 数：88
- 验证集 group 数：12
- 实际验证集样本比例：0.1192
- 实际验证集 group 比例：0.12
- seed_id 泄漏数量：0

## 3. 样本类别分布

| 类别/来源 | All | Train | Valid |
|---|---:|---:|---:|
| complaint | 100 | 90 | 10 |
| coupon_price | 80 | 70 | 10 |
| invoice | 80 | 70 | 10 |
| logistics | 150 | 130 | 20 |
| manual_transfer | 90 | 80 | 10 |
| product_info | 149 | 129 | 20 |
| quality_issue | 150 | 130 | 20 |
| return_refund | 199 | 180 | 19 |

## 4. Group 类别分布

| 类别/来源 | All | Train | Valid |
|---|---:|---:|---:|
| complaint | 10 | 9 | 1 |
| coupon_price | 8 | 7 | 1 |
| invoice | 8 | 7 | 1 |
| logistics | 15 | 13 | 2 |
| manual_transfer | 9 | 8 | 1 |
| product_info | 15 | 13 | 2 |
| quality_issue | 15 | 13 | 2 |
| return_refund | 20 | 18 | 2 |

## 5. 来源分布

| 类别/来源 | All | Train | Valid |
|---|---:|---:|---:|
| llm_user_rewrite | 998 | 879 | 119 |

## 6. Seed 泄漏检查

- 未发现 seed_id 同时出现在 train 和 valid。

## 7. Group 内类别冲突

- 未发现 group 内类别冲突。

## 8. 无效样本

- 无。
