"""
hybrid_search.py — Hybrid BM25 + Dense search with RRF fusion.
"""

import os
import re
import time
import numpy as np
import requests
from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient
from dotenv import load_dotenv

load_dotenv()

QDRANT_URL    = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_KEY    = os.getenv("QDRANT_API_KEY") or None
COLLECTION_EN = os.getenv("QDRANT_COLLECTION_EN", "medical_docs_en")
COLLECTION_ES = os.getenv("QDRANT_COLLECTION_ES", "medical_docs_es")

qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_KEY)


# ── BM25 Index ─────────────────────────────────────────────────────────────

class BM25Index:
    """In-memory BM25 index built from a Qdrant collection."""

    def __init__(self, collection: str, max_chunks: int = 15000):
        self.collection = collection
        self.points     = []
        self.texts      = []
        self.bm25       = None
        self._built     = False
        self.max_chunks = max_chunks

    def _tokenize(self, text: str) -> list:
        return [w for w in re.split(r'\s+', text.lower()) if len(w) >= 2]

    def build(self, verbose: bool = True):
        if self._built:
            return
        if verbose:
            print(f"  Building BM25 index for '{self.collection}'...")
        t = time.time()
        try:
            if not qdrant.collection_exists(self.collection):
                if verbose:
                    print(f"  Collection '{self.collection}' not found")
                return

            all_points, offset = [], None
            while True:
                result, next_offset = qdrant.scroll(
                    collection_name=self.collection,
                    limit=1000, offset=offset,
                    with_payload=True, with_vectors=False,
                )
                all_points.extend(result)
                if next_offset is None or len(all_points) >= self.max_chunks:
                    break
                offset = next_offset

            self.points    = all_points[:self.max_chunks]
            self.texts     = [p.payload.get("text", "") for p in self.points]
            tokenized      = [self._tokenize(t) for t in self.texts if t]
            if not tokenized:
                if verbose:
                    print(f"  BM25 {self.collection}: no valid tokens — skipping")
                return
            self.bm25   = BM25Okapi(tokenized)
            self._built = True
            if verbose:
                print(f"  BM25 index: {len(self.points)} chunks in {time.time()-t:.1f}s")
        except Exception as e:
            if verbose:
                print(f"  BM25 build error: {e}")

    def search(self, query: str, limit: int = 10) -> list:
        if not self._built or self.bm25 is None:
            return []
        scores  = self.bm25.get_scores(self._tokenize(query))
        top_idx = np.argsort(scores)[::-1][:limit]
        return [(self.points[i], float(scores[i])) for i in top_idx if scores[i] > 0]

    @property
    def is_ready(self) -> bool:
        return self._built and self.bm25 is not None


# ── BM25 Singletons ────────────────────────────────────────────────────────

_bm25_indices: dict = {}

def get_bm25_index(collection: str) -> BM25Index:
    if collection not in _bm25_indices:
        idx = BM25Index(collection)
        idx.build()
        _bm25_indices[collection] = idx
    return _bm25_indices[collection]


# ── Embedding ──────────────────────────────────────────────────────────────

def get_embedding(text: str) -> list:
    """Returns a BGE-M3 embedding vector via Ollama."""
    try:
        resp = requests.post(
            "http://localhost:11434/api/embed",
            json={"model": "bge-m3", "input": text[:8000]},
            timeout=60
        )
        return resp.json().get("embeddings", [[0.0] * 1024])[0]
    except Exception:
        return [0.0] * 1024


# ── RRF Fusion ─────────────────────────────────────────────────────────────

def reciprocal_rank_fusion(
    dense_results : list,
    sparse_results: list,
    k             : int   = 60,
    dense_weight  : float = 0.7,
    sparse_weight : float = 0.3,
) -> list:
    """Fuses dense and BM25 rankings using Reciprocal Rank Fusion (RRF)."""
    scores, points = {}, {}

    for rank, point in enumerate(dense_results, start=1):
        pid         = str(point.id)
        scores[pid] = scores.get(pid, 0) + dense_weight / (k + rank)
        points[pid] = point

    for rank, (point, _) in enumerate(sparse_results, start=1):
        pid         = str(point.id)
        scores[pid] = scores.get(pid, 0) + sparse_weight / (k + rank)
        if pid not in points:
            points[pid] = point

    return [points[pid] for pid in sorted(scores, key=lambda p: scores[p], reverse=True)]


# ── Hybrid Search ──────────────────────────────────────────────────────────

def hybrid_search(
    query        : str,
    collection   : str,
    limit        : int   = 10,
    dense_weight : float = 0.7,
    sparse_weight: float = 0.3,
    use_hyde     : bool  = False,
    hyde_doc     : str   = None,
) -> dict:
    """
    Runs hybrid BM25 + dense search and fuses results with RRF.
    Optionally averages query and HyDE document embeddings.
    """
    prefix = "Represent this sentence for searching relevant passages: "

    if use_hyde and hyde_doc:
        q_emb = get_embedding(prefix + query)
        h_emb = get_embedding(prefix + hyde_doc)
        emb   = [(q + h) / 2 for q, h in zip(q_emb, h_emb)]
    else:
        emb = get_embedding(prefix + query)

    # Dense search
    dense_results = []
    try:
        if qdrant.collection_exists(collection):
            raw  = qdrant.query_points(
                collection_name=collection,
                query=emb, with_payload=True, limit=limit * 2,
            ).points
            seen, unique = set(), []
            for p in raw:
                key = p.payload.get("text", "")[:100]
                if key not in seen:
                    seen.add(key)
                    unique.append(p)
            dense_results = unique
    except Exception as e:
        print(f"  Dense search error ({collection}): {e}")

    # BM25 search
    sparse_results = []
    try:
        bm25_idx = get_bm25_index(collection)
        if bm25_idx.is_ready:
            sparse_results = bm25_idx.search(query, limit=limit * 2)
    except Exception as e:
        print(f"  BM25 search error ({collection}): {e}")

    # Fuse results
    if dense_results and sparse_results:
        fused = reciprocal_rank_fusion(
            dense_results, sparse_results,
            dense_weight=dense_weight, sparse_weight=sparse_weight,
        )
    elif dense_results:
        fused = dense_results
    else:
        fused = [p for p, _ in sparse_results]

    return {
        "results"      : fused[:limit],
        "dense_count"  : len(dense_results),
        "sparse_count" : len(sparse_results),
    }


def hybrid_search_simple(
    query     : str,
    collection: str,
    limit     : int  = 10,
    use_hyde  : bool = False,
    hyde_doc  : str  = None,
) -> list:
    """Simplified wrapper — returns results list directly."""
    return hybrid_search(
        query=query, collection=collection,
        limit=limit, use_hyde=use_hyde, hyde_doc=hyde_doc,
    )["results"]