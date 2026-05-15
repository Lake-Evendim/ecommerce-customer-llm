# RAG Retrieval Evaluation Summary

- total samples: 57
- recall@1: 0.8596
- recall@3: 0.9474
- recall@5: 0.9825
- avg top1 score: 0.7567
- badcase count: 11

## Badcase Types
- rag_category_mismatch: 10
- rag_wrong_doc: 1

## By Category
| category | count | recall@1 | recall@3 | recall@5 |
|---|---:|---:|---:|---:|
| coupon_price | 10 | 0.9000 | 1.0000 | 1.0000 |
| invoice | 1 | 1.0000 | 1.0000 | 1.0000 |
| logistics | 20 | 0.9000 | 1.0000 | 1.0000 |
| manual_transfer | 8 | 0.6250 | 0.7500 | 1.0000 |
| product_info | 14 | 0.9286 | 1.0000 | 1.0000 |
| quality_issue | 3 | 0.6667 | 0.6667 | 0.6667 |
| return_refund | 1 | 1.0000 | 1.0000 | 1.0000 |
