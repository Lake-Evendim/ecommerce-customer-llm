import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "data" / "raw"
KB_DIR = ROOT / "data" / "knowledge_base"
KB_DIR.mkdir(parents=True, exist_ok=True)


def read_jsonl(path):
    items = []
    if not path.exists():
        return items
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def write_jsonl(path, items):
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def build_faq_chunks():
    faq_items = read_jsonl(RAW_DIR / "faq_seed.jsonl")
    chunks = []

    for item in faq_items:
        content = f"问题：{item['question']}\n答案：{item['answer']}"
        chunks.append({
            "chunk_id": item["id"],
            "doc_type": "faq",
            "category": item["category"],
            "title": item["question"],
            "content": content,
            "source": item.get("source", "manual_faq")
        })

    return chunks


def build_sop_chunks():
    sop_items = read_jsonl(RAW_DIR / "sop_seed.jsonl")
    chunks = []

    for item in sop_items:
        steps = "\n".join([f"{i+1}. {s}" for i, s in enumerate(item["steps"])])
        risks = "\n".join([f"- {r}" for r in item.get("risk_points", [])])
        content = f"场景：{item['title']}\n处理步骤：\n{steps}\n风险点：\n{risks}"

        chunks.append({
            "chunk_id": item["id"],
            "doc_type": "sop",
            "category": item["category"],
            "title": item["title"],
            "content": content,
            "source": item.get("source", "manual_sop")
        })

    return chunks


def build_product_chunks():
    product_items = read_jsonl(RAW_DIR / "product_seed.jsonl")
    chunks = []

    for item in product_items:
        attrs = "\n".join([f"- {k}：{v}" for k, v in item["attributes"].items()])
        content = (
            f"商品名称：{item['title']}\n"
            f"商品类目：{item['category']}\n"
            f"商品属性：\n{attrs}\n"
            f"退换货规则：{item['return_policy']}"
        )

        chunks.append({
            "chunk_id": item["product_id"],
            "doc_type": "product",
            "category": "product_info",
            "title": item["title"],
            "content": content,
            "source": "manual_product"
        })

    return chunks


def main():
    faq_chunks = build_faq_chunks()
    sop_chunks = build_sop_chunks()
    product_chunks = build_product_chunks()

    write_jsonl(KB_DIR / "faq_chunks.jsonl", faq_chunks)
    write_jsonl(KB_DIR / "sop_chunks.jsonl", sop_chunks)
    write_jsonl(KB_DIR / "product_chunks.jsonl", product_chunks)
    write_jsonl(KB_DIR / "kb_all.jsonl", faq_chunks + sop_chunks + product_chunks)

    print(f"FAQ chunks: {len(faq_chunks)}")
    print(f"SOP chunks: {len(sop_chunks)}")
    print(f"Product chunks: {len(product_chunks)}")
    print(f"All chunks: {len(faq_chunks) + len(sop_chunks) + len(product_chunks)}")


if __name__ == "__main__":
    main()