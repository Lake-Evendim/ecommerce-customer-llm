from __future__ import annotations

import json
from pathlib import Path
from collections import defaultdict


EVAL_FILES = {
    "base": "outputs/eval_results/base_generation_eval.jsonl",
    "base_rag": "outputs/eval_results/base_rag_generation_eval.jsonl",
    "sft": "outputs/eval_results/sft_generation_eval.jsonl",
    "sft_rag": "outputs/eval_results/sft_rag_generation_eval.jsonl",
}


def read_jsonl(path: str):
    rows = []
    p = Path(path)
    if not p.exists():
        print(f"[WARN] missing file: {path}")
        return rows

    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def avg(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def summarize(rows):
    total = len(rows)
    if total == 0:
        return {
            "total": 0,
            "must_include_avg_hit_rate": None,
            "must_not_include_non_violation_rate": None,
            "need_human_accuracy": None,
            "empty_answer_rate": None,
            "post_risk_rate": None,
        }

    include_avg = avg([r.get("must_include_hit_rate") for r in rows])
    violation_avg = avg([r.get("must_not_include_violation_rate", 0.0) for r in rows])

    return {
        "total": total,
        "must_include_avg_hit_rate": include_avg,
        "must_not_include_non_violation_rate": None if violation_avg is None else 1 - violation_avg,
        "need_human_accuracy": sum(1 for r in rows if r.get("need_human_correct")) / total,
        "empty_answer_rate": sum(1 for r in rows if r.get("empty_answer")) / total,
        "post_risk_rate": sum(1 for r in rows if r.get("post_has_risk")) / total,
    }


def fmt(x):
    if x is None:
        return "N/A"
    if isinstance(x, int):
        return str(x)
    return f"{x:.4f}"


def main():
    output_path = Path("outputs/eval_results/generation_compare_summary.md")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_rows = {name: read_jsonl(path) for name, path in EVAL_FILES.items()}
    summaries = {name: summarize(rows) for name, rows in all_rows.items()}

    lines = []
    lines.append("# Generation Evaluation Compare Summary")
    lines.append("")
    lines.append("## Overall")
    lines.append("")
    lines.append(
        "| model | total | must_include_avg_hit_rate | must_not_include_non_violation_rate | need_human_accuracy | empty_answer_rate | post_risk_rate |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|")

    for name, s in summaries.items():
        lines.append(
            f"| {name} | {s['total']} | "
            f"{fmt(s['must_include_avg_hit_rate'])} | "
            f"{fmt(s['must_not_include_non_violation_rate'])} | "
            f"{fmt(s['need_human_accuracy'])} | "
            f"{fmt(s['empty_answer_rate'])} | "
            f"{fmt(s['post_risk_rate'])} |"
        )

    lines.append("")
    lines.append("## By Category")
    lines.append("")

    categories = sorted({
        r.get("category", "unknown")
        for rows in all_rows.values()
        for r in rows
    })

    for cat in categories:
        lines.append(f"### {cat}")
        lines.append("")
        lines.append(
            "| model | count | must_include_avg_hit_rate | must_not_include_non_violation_rate | need_human_accuracy | post_risk_rate |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|")

        for name, rows in all_rows.items():
            cat_rows = [r for r in rows if r.get("category", "unknown") == cat]
            s = summarize(cat_rows)
            lines.append(
                f"| {name} | {s['total']} | "
                f"{fmt(s['must_include_avg_hit_rate'])} | "
                f"{fmt(s['must_not_include_non_violation_rate'])} | "
                f"{fmt(s['need_human_accuracy'])} | "
                f"{fmt(s['post_risk_rate'])} |"
            )

        lines.append("")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved compare summary: {output_path}")


if __name__ == "__main__":
    main()