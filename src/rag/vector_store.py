from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass
class KBDocument:
    """知识库 chunk 的统一结构。

    你的 kb_all.jsonl 至少应包含 doc_id / chunk_id / doc_type / category / title / content。
    这里做了容错：如果部分字段缺失，会自动兜底为空字符串。
    """

    doc_id: str
    chunk_id: str
    doc_type: str
    category: str
    title: str
    content: str
    raw: Dict[str, Any]

    @property
    def text_for_embedding(self) -> str:
        parts = [
            f"标题：{self.title}" if self.title else "",
            f"类型：{self.doc_type}" if self.doc_type else "",
            f"类别：{self.category}" if self.category else "",
            f"内容：{self.content}" if self.content else "",
        ]
        return "\n".join(p for p in parts if p).strip()

    def to_json(self) -> Dict[str, Any]:
        data = dict(self.raw)
        data.update(
            {
                "doc_id": self.doc_id,
                "chunk_id": self.chunk_id,
                "doc_type": self.doc_type,
                "category": self.category,
                "title": self.title,
                "content": self.content,
            }
        )
        return data


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"文件不存在：{path}")

    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"JSONL 解析失败：{path}:{line_no}，错误：{e}") from e
    return rows


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_kb_documents(kb_path: str | Path) -> List[KBDocument]:
    rows = read_jsonl(kb_path)
    docs: List[KBDocument] = []

    for idx, row in enumerate(rows):
        doc_id = str(row.get("doc_id") or row.get("id") or f"doc_{idx:06d}")
        chunk_id = str(row.get("chunk_id") or doc_id)
        doc_type = str(row.get("doc_type") or row.get("type") or "")
        category = str(row.get("category") or "")
        title = str(row.get("title") or "")
        content = str(row.get("content") or row.get("text") or "")

        if not content.strip() and not title.strip():
            continue

        docs.append(
            KBDocument(
                doc_id=doc_id,
                chunk_id=chunk_id,
                doc_type=doc_type,
                category=category,
                title=title,
                content=content,
                raw=row,
            )
        )

    if not docs:
        raise ValueError(f"没有从知识库中读取到有效文档：{kb_path}")
    return docs


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


class EmbeddingModel:
    def __init__(
        self,
        model_name_or_path: str,
        device: str = "auto",
        normalize_embeddings: bool = True,
        query_instruction: str = "",
    ) -> None:
        self.device = resolve_device(device)
        self.model = SentenceTransformer(model_name_or_path, device=self.device)
        self.normalize_embeddings = normalize_embeddings
        self.query_instruction = query_instruction or ""

    def encode_documents(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        embeddings = self.model.encode(
            list(texts),
            batch_size=batch_size,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=True,
        )
        return embeddings.astype("float32")

    def encode_queries(self, queries: Sequence[str], batch_size: int = 32) -> np.ndarray:
        prepared = [self.query_instruction + q if self.query_instruction else q for q in queries]
        embeddings = self.model.encode(
            prepared,
            batch_size=batch_size,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return embeddings.astype("float32")


class FaissVectorStore:
    """FAISS 向量库封装。

    normalize_embeddings=true 时，使用 inner product 近似 cosine similarity。
    """

    def __init__(self, index: faiss.Index, documents: List[Dict[str, Any]]) -> None:
        self.index = index
        self.documents = documents

    @classmethod
    def build(cls, embeddings: np.ndarray, documents: List[Dict[str, Any]]) -> "FaissVectorStore":
        if embeddings.ndim != 2:
            raise ValueError(f"embeddings 维度错误，期望 2D，实际：{embeddings.shape}")
        if len(embeddings) != len(documents):
            raise ValueError("embeddings 数量与 documents 数量不一致")

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        return cls(index=index, documents=documents)

    @classmethod
    def load(cls, index_path: str | Path, doc_store_path: str | Path) -> "FaissVectorStore":
        index_path = Path(index_path)
        doc_store_path = Path(doc_store_path)
        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index 不存在：{index_path}")
        if not doc_store_path.exists():
            raise FileNotFoundError(f"doc store 不存在：{doc_store_path}")

        index = faiss.read_index(str(index_path))
        documents = read_jsonl(doc_store_path)
        return cls(index=index, documents=documents)

    def save(self, index_path: str | Path, doc_store_path: str | Path) -> None:
        index_path = Path(index_path)
        doc_store_path = Path(doc_store_path)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        doc_store_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_path))
        write_jsonl(doc_store_path, self.documents)

    def search(
        self,
        query_embeddings: np.ndarray,
        top_k: int = 5,
        category: Optional[str] = None,
        score_threshold: float = 0.0,
    ) -> List[List[Dict[str, Any]]]:
        if query_embeddings.ndim != 2:
            raise ValueError(f"query_embeddings 维度错误，期望 2D，实际：{query_embeddings.shape}")

        search_k = min(max(top_k * 5, top_k), len(self.documents)) if category else top_k
        scores, indices = self.index.search(query_embeddings.astype("float32"), search_k)

        batch_results: List[List[Dict[str, Any]]] = []
        for row_scores, row_indices in zip(scores, indices):
            results: List[Dict[str, Any]] = []
            for score, idx in zip(row_scores, row_indices):
                if idx < 0:
                    continue
                doc = dict(self.documents[int(idx)])
                doc["score"] = float(score)

                if category and doc.get("category") != category:
                    continue
                if float(score) < score_threshold:
                    continue

                results.append(doc)
                if len(results) >= top_k:
                    break
            batch_results.append(results)
        return batch_results
