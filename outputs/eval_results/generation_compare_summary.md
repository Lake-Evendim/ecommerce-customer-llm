# Generation Evaluation Compare Summary

## Overall

| model | total | must_include_avg_hit_rate | must_not_include_non_violation_rate | need_human_accuracy | empty_answer_rate | post_risk_rate |
|---|---:|---:|---:|---:|---:|---:|
| base | 200 | 0.0818 | 0.9950 | 0.7100 | 0.0000 | 0.0000 |
| base_rag | 200 | 0.3371 | 0.9842 | 0.8200 | 0.0000 | 0.0000 |
| sft | 200 | 0.2287 | 0.9775 | 0.8100 | 0.0000 | 0.0000 |
| sft_rag | 200 | 0.2795 | 0.9925 | 0.8050 | 0.0000 | 0.0000 |

## By Category

### complaint

| model | count | must_include_avg_hit_rate | must_not_include_non_violation_rate | need_human_accuracy | post_risk_rate |
|---|---:|---:|---:|---:|---:|
| base | 15 | 0.0444 | 1.0000 | 0.2667 | 0.0000 |
| base_rag | 15 | 0.1944 | 0.9222 | 1.0000 | 0.0000 |
| sft | 15 | 0.2167 | 1.0000 | 1.0000 | 0.0000 |
| sft_rag | 15 | 0.2167 | 1.0000 | 1.0000 | 0.0000 |

### coupon_price

| model | count | must_include_avg_hit_rate | must_not_include_non_violation_rate | need_human_accuracy | post_risk_rate |
|---|---:|---:|---:|---:|---:|
| base | 10 | 0.0000 | 1.0000 | 0.9000 | 0.0000 |
| base_rag | 10 | 0.3833 | 1.0000 | 0.9000 | 0.0000 |
| sft | 10 | 0.1667 | 1.0000 | 0.9000 | 0.0000 |
| sft_rag | 10 | 0.2500 | 1.0000 | 0.9000 | 0.0000 |

### invoice

| model | count | must_include_avg_hit_rate | must_not_include_non_violation_rate | need_human_accuracy | post_risk_rate |
|---|---:|---:|---:|---:|---:|
| base | 24 | 0.0139 | 1.0000 | 0.2917 | 0.0000 |
| base_rag | 24 | 0.3438 | 0.9861 | 0.2083 | 0.0000 |
| sft | 24 | 0.2743 | 0.9375 | 0.2917 | 0.0000 |
| sft_rag | 24 | 0.2431 | 0.9861 | 0.2083 | 0.0000 |

### logistics

| model | count | must_include_avg_hit_rate | must_not_include_non_violation_rate | need_human_accuracy | post_risk_rate |
|---|---:|---:|---:|---:|---:|
| base | 29 | 0.0115 | 1.0000 | 1.0000 | 0.0000 |
| base_rag | 29 | 0.4109 | 0.9655 | 1.0000 | 0.0000 |
| sft | 29 | 0.4098 | 0.9540 | 1.0000 | 0.0000 |
| sft_rag | 29 | 0.4655 | 0.9828 | 1.0000 | 0.0000 |

### manual_transfer

| model | count | must_include_avg_hit_rate | must_not_include_non_violation_rate | need_human_accuracy | post_risk_rate |
|---|---:|---:|---:|---:|---:|
| base | 19 | 0.0702 | 0.9825 | 0.2632 | 0.0000 |
| base_rag | 19 | 0.4474 | 1.0000 | 0.8947 | 0.0000 |
| sft | 19 | 0.3333 | 0.9649 | 0.8947 | 0.0000 |
| sft_rag | 19 | 0.3246 | 1.0000 | 0.8421 | 0.0000 |

### product_info

| model | count | must_include_avg_hit_rate | must_not_include_non_violation_rate | need_human_accuracy | post_risk_rate |
|---|---:|---:|---:|---:|---:|
| base | 35 | 0.1890 | 0.9810 | 0.9714 | 0.0000 |
| base_rag | 35 | 0.4705 | 1.0000 | 0.9429 | 0.0000 |
| sft | 35 | 0.2105 | 0.9714 | 0.9714 | 0.0000 |
| sft_rag | 35 | 0.4048 | 0.9905 | 0.8857 | 0.0000 |

### quality_issue

| model | count | must_include_avg_hit_rate | must_not_include_non_violation_rate | need_human_accuracy | post_risk_rate |
|---|---:|---:|---:|---:|---:|
| base | 32 | 0.0677 | 1.0000 | 0.6562 | 0.0000 |
| base_rag | 32 | 0.2109 | 0.9792 | 0.6562 | 0.0000 |
| sft | 32 | 0.0625 | 1.0000 | 0.5000 | 0.0000 |
| sft_rag | 32 | 0.1198 | 0.9896 | 0.6562 | 0.0000 |

### return_refund

| model | count | must_include_avg_hit_rate | must_not_include_non_violation_rate | need_human_accuracy | post_risk_rate |
|---|---:|---:|---:|---:|---:|
| base | 36 | 0.1366 | 1.0000 | 0.9167 | 0.0000 |
| base_rag | 36 | 0.2440 | 1.0000 | 0.9722 | 0.0000 |
| sft | 36 | 0.1847 | 1.0000 | 0.9722 | 0.0000 |
| sft_rag | 36 | 0.1847 | 1.0000 | 0.9722 | 0.0000 |

