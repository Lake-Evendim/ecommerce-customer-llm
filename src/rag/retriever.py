from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from src.rag.vector_store import EmbeddingModel, FaissVectorStore


class Retriever:
    def __init__(self, config_path: str | Path = "configs/rag.yaml") -> None:
        self.config_path = Path(config_path)
        with self.config_path.open("r", encoding="utf-8") as f:
            self.cfg: Dict[str, Any] = yaml.safe_load(f)

        paths = self.cfg["paths"]
        emb_cfg = self.cfg["embedding"]

        self.embedding_model = EmbeddingModel(
            model_name_or_path=emb_cfg["model_name_or_path"],
            device=emb_cfg.get("device", "auto"),
            normalize_embeddings=bool(emb_cfg.get("normalize_embeddings", True)),
            query_instruction=emb_cfg.get("query_instruction", ""),
        )
        self.vector_store = FaissVectorStore.load(
            index_path=paths["faiss_index_path"],
            doc_store_path=paths["doc_store_path"],
        )

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        category: Optional[str] = None,
        score_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        retrieval_cfg = self.cfg.get("retrieval", {})
        top_k = int(top_k or retrieval_cfg.get("top_k", 5))
        score_threshold = float(
            retrieval_cfg.get("score_threshold", 0.0) if score_threshold is None else score_threshold
        )

        use_category_filter = bool(retrieval_cfg.get("use_category_filter", False))
        actual_category = category if use_category_filter else None

        query_embedding = self.embedding_model.encode_queries([query])
        results = self.vector_store.search(
            query_embeddings=query_embedding,
            top_k=top_k,
            category=actual_category,
            score_threshold=score_threshold,
        )[0]
        return results
