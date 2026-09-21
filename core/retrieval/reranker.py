"""
reranker.py — Cross-encoder re-ranking of retrieved chunks.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
  - 22M parameters, fast on CPU (~50ms for 15 chunks)
  - Scores (query, chunk) pairs jointly for higher precision than cosine similarity
"""

import time
from sentence_transformers import CrossEncoder

# Available models by quality/speed tradeoff
RERANKER_MODELS = {
    "fast"    : "cross-encoder/ms-marco-MiniLM-L-6-v2",   # 22M params, ~50ms
    "balanced": "cross-encoder/ms-marco-MiniLM-L-12-v2",  # 33M params, ~80ms
    "best"    : "cross-encoder/ms-marco-electra-base",     # 110M params, ~200ms
}

_reranker_instance   = None
_reranker_model_name = None


def get_reranker(model_key: str = "fast") -> CrossEncoder:
    """Returns the cross-encoder singleton (loaded once into memory)."""
    global _reranker_instance, _reranker_model_name
    model_name = RERANKER_MODELS.get(model_key, RERANKER_MODELS["fast"])
    if _reranker_instance is None or _reranker_model_name != model_name:
        print(f"  Loading re-ranker: {model_name}...")
        t = time.time()
        _reranker_instance   = CrossEncoder(model_name, max_length=512)
        _reranker_model_name = model_name
        print(f"  Re-ranker loaded in {time.time()-t:.1f}s")
    return _reranker_instance


def rerank_chunks(
    query    : str,
    chunks   : list,
    top_k    : int = 5,
    model_key: str = "fast"
) -> list:
    """
    Re-ranks retrieved Qdrant chunks by relevance using a cross-encoder.
    More accurate than cosine similarity: scores (query, chunk) jointly.

    Args:
        query     : user clinical question
        chunks    : list of Qdrant points
        top_k     : number of chunks to keep after re-ranking
        model_key : "fast" | "balanced" | "best"

    Returns:
        top_k most relevant chunks, sorted by descending score
    """
    if not chunks or len(chunks) <= top_k:
        return chunks

    reranker = get_reranker(model_key)
    pairs    = [(query, p.payload.get("text", "")[:512]) for p in chunks]

    t      = time.time()
    scores = reranker.predict(pairs)
    elapsed = time.time() - t

    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    print(f"  Re-ranker: {len(chunks)} → {top_k} chunks ({elapsed*1000:.0f}ms)")
    return [chunk for chunk, _ in ranked[:top_k]]


def rerank_with_scores(
    query    : str,
    chunks   : list,
    top_k    : int = 5,
    model_key: str = "fast"
) -> list:
    """Same as rerank_chunks but returns (chunk, score) tuples."""
    if not chunks:
        return []
    reranker = get_reranker(model_key)
    pairs    = [(query, p.payload.get("text", "")[:512]) for p in chunks]
    scores   = reranker.predict(pairs)
    return sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)[:top_k]