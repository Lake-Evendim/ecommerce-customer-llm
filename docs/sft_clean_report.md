# SFT 数据清洗报告

## 1. 总览

- 原始样本数：1000
- 清洗后样本数：998
- 删除样本数：2
- 删除比例：0.002
- 输出文件：`/home/amlab/桌面/shenzy/ecommerce-customer-llm/data/interim/cleaned_sft.jsonl`

## 2. 输入解析错误

- 无

## 3. 删除原因统计

- near_duplicate_same_answer: 2

## 4. 类别分布

| 类别 | 数量 |
|---|---:|
| complaint | 100 |
| coupon_price | 80 |
| invoice | 80 |
| logistics | 150 |
| manual_transfer | 90 |
| product_info | 149 |
| quality_issue | 150 |
| return_refund | 199 |

## 5. 来源分布

| 来源 | 数量 |
|---|---:|
| llm_user_rewrite | 998 |

## 6. 文本长度统计

| 字段 | 样本数 | 最短 | 最长 | 平均 |
|---|---:|---:|---:|---:|
| user | 998 | 6 | 30 | 13.98 |
| assistant | 998 | 54 | 95 | 72.98 |

## 7. Seed 扩写保留数量

| seed_id | 保留扩写数 |
|---|---:|
| sft_seed_0001 | 10 |
| sft_seed_0002 | 10 |
| sft_seed_0003 | 10 |
| sft_seed_0004 | 10 |
| sft_seed_0005 | 10 |
| sft_seed_0006 | 9 |
| sft_seed_0007 | 10 |
| sft_seed_0008 | 10 |
| sft_seed_0009 | 10 |
| sft_seed_0010 | 10 |
| sft_seed_0011 | 10 |
| sft_seed_0012 | 10 |
| sft_seed_0013 | 10 |
| sft_seed_0014 | 10 |
| sft_seed_0015 | 10 |
| sft_seed_0016 | 10 |
| sft_seed_0017 | 10 |
| sft_seed_0018 | 10 |
| sft_seed_0019 | 10 |
| sft_seed_0020 | 10 |
| sft_seed_0021 | 10 |
| sft_seed_0022 | 10 |
| sft_seed_0023 | 10 |
| sft_seed_0024 | 10 |
| sft_seed_0025 | 10 |
| sft_seed_0026 | 10 |
| sft_seed_0027 | 10 |
| sft_seed_0028 | 10 |
| sft_seed_0029 | 10 |
| sft_seed_0030 | 10 |
| sft_seed_0031 | 10 |
| sft_seed_0032 | 10 |
| sft_seed_0033 | 10 |
| sft_seed_0034 | 10 |
| sft_seed_0035 | 10 |
| sft_seed_0036 | 10 |
| sft_seed_0037 | 10 |
| sft_seed_0038 | 10 |
| sft_seed_0039 | 10 |
| sft_seed_0040 | 10 |
| sft_seed_0041 | 10 |
| sft_seed_0042 | 9 |
| sft_seed_0043 | 10 |
| sft_seed_0044 | 10 |
| sft_seed_0045 | 10 |
| sft_seed_0046 | 10 |
| sft_seed_0047 | 10 |
| sft_seed_0048 | 10 |
| sft_seed_0049 | 10 |
| sft_seed_0050 | 10 |
| sft_seed_0051 | 10 |
| sft_seed_0052 | 10 |
| sft_seed_0053 | 10 |
| sft_seed_0054 | 10 |
| sft_seed_0055 | 10 |
| sft_seed_0056 | 10 |
| sft_seed_0057 | 10 |
| sft_seed_0058 | 10 |
| sft_seed_0059 | 10 |
| sft_seed_0060 | 10 |
| sft_seed_0061 | 10 |
| sft_seed_0062 | 10 |
| sft_seed_0063 | 10 |
| sft_seed_0064 | 10 |
| sft_seed_0065 | 10 |
| sft_seed_0066 | 10 |
| sft_seed_0067 | 10 |
| sft_seed_0068 | 10 |
| sft_seed_0069 | 10 |
| sft_seed_0070 | 10 |
| sft_seed_0071 | 10 |
| sft_seed_0072 | 10 |
| sft_seed_0073 | 10 |
| sft_seed_0074 | 10 |
| sft_seed_0075 | 10 |
| sft_seed_0076 | 10 |
| sft_seed_0077 | 10 |
| sft_seed_0078 | 10 |
| sft_seed_0079 | 10 |
| sft_seed_0080 | 10 |
| sft_seed_0081 | 10 |
| sft_seed_0082 | 10 |
| sft_seed_0083 | 10 |
| sft_seed_0084 | 10 |
| sft_seed_0085 | 10 |
| sft_seed_0086 | 10 |
| sft_seed_0087 | 10 |
| sft_seed_0088 | 10 |
| sft_seed_0089 | 10 |
| sft_seed_0090 | 10 |
| sft_seed_0091 | 10 |
| sft_seed_0092 | 10 |
| sft_seed_0093 | 10 |
| sft_seed_0094 | 10 |
| sft_seed_0095 | 10 |
| sft_seed_0096 | 10 |
| sft_seed_0097 | 10 |
| sft_seed_0098 | 10 |
| sft_seed_0099 | 10 |
| sft_seed_0100 | 10 |

## 8. 清洗规则说明

本脚本执行以下确定性清洗：

1. 检查 `messages` 是否存在且包含 `system/user/assistant`；
2. 检查角色顺序是否为 `system -> user -> assistant`；
3. 检查 `category` 是否属于项目定义的 8 个类别；
4. 删除 user 或 assistant 过短、过长的样本；
5. 删除包含手机号、身份证号、邮箱、明显订单号的样本；
6. 删除 assistant 中明显承诺退款、赔偿、送达或无条件处理的高风险样本；
7. 删除完全重复的 user-answer 样本；
8. 可选删除同一 answer 下的近重复 user 问法。

## 9. 删除样例

### 1. near_duplicate_same_answer

- id: `sft_llm_000053`
- seed_id: `sft_seed_0006`
- category: `return_refund`
- source: `llm_user_rewrite`
- user: 外包装拆了但没用过，美妆能退吗？"
- assistant: 您好，美妆类商品是否支持退货需要以商品页面规则为准。若外包装已拆封，通常可能会影响七天无理由退货。若商品存在破损、漏液或质量问题，您可以上传相关凭证并在订单页申请售后。

### 2. near_duplicate_same_answer

- id: `sft_llm_000418`
- seed_id: `sft_seed_0042`
- category: `product_info`
- source: `llm_user_rewrite`
- user: 这件衣服的版型是宽松的还是紧身的？
- assistant: 您好，这款衣服为宽松版型。不同身高体重的穿着效果会有差异，建议您参考商品详情页的尺码表，并结合平时穿衣习惯选择合适尺码。

