"""Shared retrieval pipeline used by both production and evaluation."""

from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import asdict, dataclass
from functools import lru_cache

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import chromadb
import jieba
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from .config import (
    CHROMA_PATH,
    COLLECTION_NAME,
    EMBED_MODEL_NAME,
    MIN_RELEVANCE_SCORE,
    MIN_VECTOR_SCORE,
    RERANK_BATCH_SIZE,
    RERANK_CANDIDATES,
    RERANK_CANDIDATES_CPU,
    RERANK_MODEL_NAME,
    RETRIEVAL_CACHE_SIZE,
    RETRIEVAL_CANDIDATES,
    RETRIEVAL_TOP_K,
)


@dataclass(frozen=True)
class RetrievedDocument:
    doc_id: str
    text: str
    score: float
    vector_score: float
    retrieval_score: float
    score_kind: str
    metadata: dict

    def to_dict(self) -> dict:
        return asdict(self)


embed_model = SentenceTransformer(EMBED_MODEL_NAME)
device = "cuda" if torch.cuda.is_available() else "cpu"
use_fp16 = device == "cuda"
effective_rerank_candidates = (
    min(
        RERANK_CANDIDATES,
        max(RETRIEVAL_TOP_K, RERANK_CANDIDATES_CPU),
    )
    if device == "cpu"
    else RERANK_CANDIDATES
)

reranker = None
reranker_error = ""
if os.path.isdir(RERANK_MODEL_NAME):
    try:
        from FlagEmbedding import FlagReranker

        reranker = FlagReranker(
            RERANK_MODEL_NAME,
            use_fp16=use_fp16,
            max_length=256,
            devices=device,
        )
    except Exception as exc:  # noqa: BLE001  # pragma: no cover
        reranker_error = str(exc)

chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))
collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
_collection_space = str((collection.metadata or {}).get("hnsw:space", "l2"))

_all_data = collection.get(include=["documents", "metadatas"])
_ids = _all_data.get("ids", [])
_docs = _all_data.get("documents", []) or []
_metas = _all_data.get("metadatas", []) or []
_id_to_doc = dict(zip(_ids, _docs))
_id_to_meta = {
    doc_id: (metadata or {})
    for doc_id, metadata in zip(_ids, _metas)
}
_id_to_index = {doc_id: index for index, doc_id in enumerate(_ids)}
_tokenized_corpus = [list(jieba.cut(document or "")) for document in _docs]
_bm25 = BM25Okapi(_tokenized_corpus) if _tokenized_corpus else None


def _as_float_list(scores) -> list[float]:
    if isinstance(scores, (int, float)):
        return [float(scores)]
    return [float(score) for score in scores]


def _vector_candidates(query: str, limit: int) -> list[tuple[str, str, float]]:
    if not _docs:
        return []
    query_vector = embed_model.encode(
        query,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).tolist()
    result = collection.query(
        query_embeddings=[query_vector],
        n_results=max(1, min(limit, len(_docs))),
        include=["documents", "metadatas", "distances"],
    )
    ids = result.get("ids", [[]])[0]
    documents = result.get("documents", [[]])[0]
    distances = result.get("distances", [[]])[0]
    candidates = []
    for doc_id, document, distance in zip(ids, documents, distances):
        distance = float(distance)
        if _collection_space == "cosine" or _collection_space == "ip":
            similarity = 1.0 - distance
        else:
            # Chroma's L2 distance is squared Euclidean distance. Embeddings
            # are normalized, so cosine similarity = 1 - distance / 2.
            similarity = 1.0 - distance / 2.0
        similarity = max(-1.0, min(1.0, similarity))
        candidates.append((doc_id, document or "", similarity))
    return candidates


def _bm25_candidates(query: str, limit: int) -> list[str]:
    if not _bm25 or not _ids:
        return []
    tokenized_query = list(jieba.cut(query))
    scores = _bm25.get_scores(tokenized_query)
    ranked_indexes = sorted(
        range(len(scores)),
        key=lambda index: float(scores[index]),
        reverse=True,
    )
    return [
        _ids[index]
        for index in ranked_indexes[:limit]
        if float(scores[index]) > 0
    ]


def _hybrid_candidates(query: str) -> list[dict]:
    vector_hits = _vector_candidates(query, RETRIEVAL_CANDIDATES)
    vector_by_id = {
        doc_id: {"text": text, "vector_score": score}
        for doc_id, text, score in vector_hits
    }
    bm25_ids = _bm25_candidates(query, RETRIEVAL_CANDIDATES)

    retrieval_scores: dict[str, float] = defaultdict(float)
    for rank, (doc_id, _, _) in enumerate(vector_hits, start=1):
        retrieval_scores[doc_id] += 1.0 / (60 + rank)
    for rank, doc_id in enumerate(bm25_ids, start=1):
        retrieval_scores[doc_id] += 1.0 / (60 + rank)

    candidates = []
    for doc_id in sorted(
        retrieval_scores,
        key=retrieval_scores.get,
        reverse=True,
    )[:effective_rerank_candidates]:
        vector = vector_by_id.get(doc_id, {})
        candidates.append(
            {
                "doc_id": doc_id,
                "text": vector.get("text") or _id_to_doc.get(doc_id, ""),
                "vector_score": float(vector.get("vector_score", 0.0)),
                "retrieval_score": float(retrieval_scores[doc_id]),
            }
        )
    return candidates


def _build_results(
    query: str,
    candidates: list[dict],
    *,
    mode: str,
    top_k: int,
    min_score: float | None,
    include_below_threshold: bool,
) -> tuple[RetrievedDocument, ...]:
    if not candidates:
        return ()

    if mode == "hybrid" and reranker is not None:
        pairs = [[query, candidate["text"]] for candidate in candidates]
        scores = _as_float_list(
            reranker.compute_score(
                pairs,
                normalize=True,
                batch_size=RERANK_BATCH_SIZE,
            )
        )
        for candidate, score in zip(candidates, scores):
            candidate["score"] = score
        score_kind = "rerank"
        threshold = MIN_RELEVANCE_SCORE if min_score is None else min_score
    elif mode in {"hybrid", "hybrid_no_rerank"}:
        # hybrid_no_rerank is an ablation baseline. It must use the RRF score,
        # not silently fall back to the dense-vector score after fusion.
        for candidate in candidates:
            candidate["score"] = candidate["retrieval_score"]
        score_kind = "rrf"
        threshold = float("-inf") if min_score is None else min_score
    else:
        for candidate in candidates:
            candidate["score"] = candidate["vector_score"]
        score_kind = "vector"
        threshold = MIN_VECTOR_SCORE if min_score is None else min_score

    candidates.sort(key=lambda item: item["score"], reverse=True)
    if not include_below_threshold:
        candidates = [
            candidate
            for candidate in candidates
            if float(candidate["score"]) >= float(threshold)
        ]

    return tuple(
        RetrievedDocument(
            doc_id=candidate["doc_id"],
            text=candidate["text"],
            score=float(candidate["score"]),
            vector_score=float(candidate["vector_score"]),
            retrieval_score=float(candidate["retrieval_score"]),
            score_kind=score_kind,
            metadata=_id_to_meta.get(candidate["doc_id"], {}),
        )
        for candidate in candidates[: max(1, top_k)]
    )


@lru_cache(maxsize=RETRIEVAL_CACHE_SIZE)
def _retrieve_cached(
    query: str,
    top_k: int,
    mode: str,
    min_score: float | None,
    include_below_threshold: bool,
) -> tuple[RetrievedDocument, ...]:
    if not query.strip():
        return ()
    if mode == "vector":
        vector_hits = _vector_candidates(query, RETRIEVAL_CANDIDATES)
        candidates = [
            {
                "doc_id": doc_id,
                "text": text,
                "vector_score": score,
                "retrieval_score": score,
            }
            for doc_id, text, score in vector_hits
        ]
    else:
        candidates = _hybrid_candidates(query)
    return _build_results(
        query,
        candidates,
        mode=mode,
        top_k=top_k,
        min_score=min_score,
        include_below_threshold=include_below_threshold,
    )


def retrieve(
    query: str,
    *,
    top_k: int = RETRIEVAL_TOP_K,
    mode: str = "hybrid",
    min_score: float | None = None,
    include_below_threshold: bool = False,
) -> list[dict]:
    """Retrieve evidence and preserve real scores through the API boundary."""
    if mode not in {"vector", "hybrid_no_rerank", "hybrid"}:
        raise ValueError(f"unsupported retrieval mode: {mode}")
    results = _retrieve_cached(
        query.strip(),
        int(top_k),
        mode,
        None if min_score is None else float(min_score),
        bool(include_below_threshold),
    )
    return [result.to_dict() for result in results]


def get_top10_chunks(query_text: str, top_n: int = 10) -> list[str]:
    """Backward-compatible helper; new code should call retrieve()."""
    return [item["text"] for item in retrieve(query_text, top_k=min(top_n, 10))]


def retrieval_status() -> dict:
    return {
        "documents": len(_docs),
        "collection_space": _collection_space,
        "reranker_ready": reranker is not None,
        "reranker_error": bool(reranker_error),
        "device": device,
        "rerank_candidates": effective_rerank_candidates,
        "rerank_batch_size": RERANK_BATCH_SIZE,
        "min_relevance_score": MIN_RELEVANCE_SCORE,
        "min_vector_score": MIN_VECTOR_SCORE,
    }


def clear_retrieval_cache() -> None:
    _retrieve_cached.cache_clear()
