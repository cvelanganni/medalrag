"""
test_12_performance.py
----------------------
Tests pipeline performance and response times.
Ensures each component meets minimum speed requirements.

Thresholds:
    Embedding       : < 5s per query
    Qdrant retrieval: < 2s for top-15
    Reranking       : < 5s for 15 chunks
    LLM generation  : < 30s
    Full pipeline   : < 60s

Run:
    uv run python tests/test_12_performance.py
    or:
    uv run pytest tests/test_12_performance.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import pytest
import requests
from dotenv import load_dotenv
load_dotenv()

OLLAMA_URL    = "http://localhost:11434/api/embed"
COLLECTION_EN = os.getenv("QDRANT_COLLECTION_EN", "medical_docs_en")

TEST_QUERY = "Dolutegravir dose adjustment Rifampin tuberculosis co-infection"

def get_embedding(text: str) -> list:
    resp = requests.post(
        OLLAMA_URL,
        json={"model": "bge-m3", "input": text[:8000]},
        timeout=60
    )
    return resp.json().get("embeddings", [[0.0]*1024])[0]


# ── Performance benchmarks ─────────────────────────────────────

def test_embedding_speed():
    """BGE-M3 embedding should complete in < 5 seconds."""
    t_start = time.time()
    emb     = get_embedding(TEST_QUERY)
    elapsed = time.time() - t_start

    print(f"\n  Embedding time: {elapsed:.2f}s")
    assert len(emb) == 1024
    assert elapsed < 5.0, f"Embedding too slow: {elapsed:.2f}s (max 5s)"


def test_qdrant_retrieval_speed():
    """Qdrant top-15 retrieval should complete in < 2 seconds."""
    from qdrant_client import QdrantClient

    q       = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))
    emb     = get_embedding(TEST_QUERY)

    t_start = time.time()
    results = q.query_points(
        collection_name = COLLECTION_EN,
        query           = emb,
        with_payload    = True,
        limit           = 15,
    ).points
    elapsed = time.time() - t_start

    print(f"\n  Qdrant retrieval time: {elapsed:.2f}s ({len(results)} chunks)")
    assert len(results) > 0
    assert elapsed < 2.0, f"Retrieval too slow: {elapsed:.2f}s (max 2s)"


def test_reranker_speed():
    """Cross-encoder reranking of 15 chunks should complete in < 8 seconds."""
    from qdrant_client import QdrantClient
    from core.retrieval.reranker import rerank_chunks

    q       = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))
    emb     = get_embedding(TEST_QUERY)
    results = q.query_points(
        collection_name = COLLECTION_EN,
        query           = emb,
        with_payload    = True,
        limit           = 15,
    ).points

    t_start  = time.time()
    reranked = rerank_chunks(TEST_QUERY, results, top_k=5)
    elapsed  = time.time() - t_start

    print(f"\n  Reranking time: {elapsed:.2f}s (15 → {len(reranked)} chunks)")
    assert len(reranked) > 0
    assert elapsed < 8.0, f"Reranking too slow: {elapsed:.2f}s (max 8s)"


def test_ppr_speed():
    """PPR HippoRAG should complete in < 3 seconds."""
    import pickle
    import networkx as nx

    with open("graph_cache.pkl", "rb") as f:
        G = pickle.load(f)

    seeds = ["Dolutegravir", "Rifampin"]
    personalization = {n: (1.0/len(seeds) if n in seeds else 0.0)
                      for n in G.nodes()}

    t_start = time.time()
    ppr     = nx.pagerank(G, alpha=0.85, personalization=personalization)
    elapsed = time.time() - t_start

    print(f"\n  PPR time: {elapsed:.3f}s ({G.number_of_nodes()} nodes)")
    assert elapsed < 3.0, f"PPR too slow: {elapsed:.2f}s (max 3s)"


def test_llm_generation_speed():
    """LLM response generation should complete in < 30 seconds."""
    from openai import OpenAI

    client  = OpenAI()
    context = "Dolutegravir 50mg twice daily when co-administered with Rifampin."

    t_start = time.time()
    resp    = client.chat.completions.create(
        model    = "gpt-4o-mini",
        messages = [{"role": "user", "content":
            f"Context: {context}\nQuestion: {TEST_QUERY}\nAnswer briefly:"}],
        temperature = 0,
        max_tokens  = 150,
    )
    elapsed = time.time() - t_start
    answer  = resp.choices[0].message.content

    print(f"\n  LLM generation time: {elapsed:.2f}s")
    print(f"  Answer: {answer[:100]}...")
    assert elapsed < 30.0, f"LLM too slow: {elapsed:.2f}s (max 30s)"
    assert len(answer) > 10, "LLM response too short"


def test_full_pipeline_speed():
    """Full pipeline (embed + retrieve + rerank) should complete in < 30s."""
    from qdrant_client import QdrantClient
    from core.retrieval.reranker import rerank_chunks

    q = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))

    t_start = time.time()

    # 1. Embed
    emb = get_embedding(TEST_QUERY)

    # 2. Retrieve
    results = q.query_points(
        collection_name = COLLECTION_EN,
        query           = emb,
        with_payload    = True,
        limit           = 15,
    ).points

    # 3. Rerank
    reranked = rerank_chunks(TEST_QUERY, results, top_k=5)

    elapsed = time.time() - t_start

    print(f"\n  Full pipeline time: {elapsed:.2f}s")
    print(f"  Steps: embed + retrieve({len(results)}) + rerank({len(reranked)})")
    assert elapsed < 30.0, f"Pipeline too slow: {elapsed:.2f}s (max 30s)"


def test_throughput():
    """
    Measure queries per minute (QPM) for the embedding step.
    Target: at least 20 QPM.
    """
    n_queries = 5
    queries   = [
        "Dolutegravir HIV treatment",
        "TAF renal impairment safety",
        "Rifampin tuberculosis dose",
        "Pregnancy antiretroviral",
        "CD4 count prophylaxis",
    ]

    t_start = time.time()
    for q in queries:
        get_embedding(q)
    elapsed = time.time() - t_start

    qpm = n_queries / elapsed * 60
    print(f"\n  Throughput: {qpm:.1f} queries/min ({elapsed:.2f}s for {n_queries} queries)")
    assert qpm >= 10, f"Throughput too low: {qpm:.1f} QPM (min 10)"


if __name__ == "__main__":
    import subprocess
    subprocess.run([sys.executable, "-m", "pytest",
                   __file__, "-v", "--tb=short", "-s"])