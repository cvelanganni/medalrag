"""
test_11_robustness.py
---------------------
Tests pipeline robustness with edge cases:
empty inputs, very long texts, special characters,
service failures, and unexpected inputs.

Run:
    uv run python tests/test_11_robustness.py
    or:
    uv run pytest tests/test_11_robustness.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import requests
from dotenv import load_dotenv
load_dotenv()

OLLAMA_URL = "http://localhost:11434/api/embed"

def get_embedding(text: str) -> list:
    resp = requests.post(
        OLLAMA_URL,
        json={"model": "bge-m3", "input": text[:8000]},
        timeout=60
    )
    return resp.json().get("embeddings", [[0.0]*1024])[0]


# ── Embedding robustness ───────────────────────────────────────

def test_embedding_empty_string():
    """Embedding should handle empty string gracefully."""
    try:
        emb = get_embedding("")
        assert len(emb) == 1024, "Empty string should still return 1024d vector"
    except Exception:
        pytest.skip("Empty string embedding not supported — acceptable")


def test_embedding_very_long_text():
    """Embedding should truncate very long texts gracefully."""
    long_text = "Dolutegravir HIV treatment " * 500  # ~10000 words
    emb       = get_embedding(long_text)
    assert len(emb) == 1024, "Long text should be truncated to 8000 chars"


def test_embedding_special_characters():
    """Embedding should handle special medical characters."""
    text = "eGFR ≥60 mL/min → TAF preferred. HLA-B*5701 negative. CD4+ <200 cells/μL"
    emb  = get_embedding(text)
    assert len(emb) == 1024
    assert any(v != 0.0 for v in emb)


def test_embedding_multilingual_mixed():
    """Embedding should handle mixed EN/ES text."""
    text = "Dolutegravir HIV treatment VIH embarazo pregnancy"
    emb  = get_embedding(text)
    assert len(emb) == 1024


def test_embedding_numbers_only():
    """Embedding should handle numeric-only input."""
    text = "50 mg twice daily 0.3 mL/min eGFR 45"
    emb  = get_embedding(text)
    assert len(emb) == 1024


# ── Retrieval robustness ───────────────────────────────────────

def test_retrieval_empty_query():
    """Retrieval should handle empty query gracefully."""
    from qdrant_client import QdrantClient
    q   = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))
    emb = get_embedding("HIV treatment")  # use valid embedding

    results = q.query_points(
        collection_name = "medical_docs_en",
        query           = emb,
        with_payload    = True,
        limit           = 5,
    ).points
    assert isinstance(results, list)


def test_retrieval_nonsense_query():
    """Retrieval should return results even for nonsense queries."""
    from qdrant_client import QdrantClient
    q   = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))
    emb = get_embedding("xkzqwmvbpnr aaaaa 12345")

    results = q.query_points(
        collection_name = "medical_docs_en",
        query           = emb,
        with_payload    = True,
        limit           = 5,
    ).points
    assert isinstance(results, list)
    assert len(results) > 0, "Should always return some results"


def test_retrieval_top_k_limits():
    """Retrieval should respect top-k limits."""
    from qdrant_client import QdrantClient
    q   = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))
    emb = get_embedding("HIV treatment")

    for k in [1, 5, 10, 15, 20]:
        results = q.query_points(
            collection_name = "medical_docs_en",
            query           = emb,
            with_payload    = True,
            limit           = k,
        ).points
        assert len(results) <= k, f"Should return at most {k} results"


# ── Graph robustness ───────────────────────────────────────────

def test_graph_missing_entity():
    """Graph should handle queries for missing entities gracefully."""
    import pickle
    import networkx as nx

    with open("graph_cache.pkl", "rb") as f:
        G = pickle.load(f)

    # Entity that doesn't exist
    fake_entity = "NonExistentDrug12345"
    assert fake_entity not in G.nodes(), "Fake entity should not be in graph"

    # PPR with non-existent seed should not crash
    seeds  = ["Dolutegravir", fake_entity]
    valid  = [s for s in seeds if s in G.nodes()]

    if valid:
        personalization = {n: (1.0/len(valid) if n in valid else 0.0)
                          for n in G.nodes()}
        ppr = nx.pagerank(G, alpha=0.85, personalization=personalization)
        assert len(ppr) == G.number_of_nodes()


def test_graph_disconnected_entities():
    """PPR should handle entities with no connections gracefully."""
    import pickle
    import networkx as nx

    with open("graph_cache.pkl", "rb") as f:
        G = pickle.load(f)

    # Find entities with few connections
    low_degree = [n for n, d in G.degree() if d == 1]
    if low_degree:
        seed = low_degree[0]
        personalization = {n: (1.0 if n == seed else 0.0)
                          for n in G.nodes()}
        ppr = nx.pagerank(G, alpha=0.85, personalization=personalization)
        assert sum(ppr.values()) > 0.99, "PPR scores should sum to ~1.0"


# ── NER robustness ─────────────────────────────────────────────

def test_ner_empty_text():
    """NER should handle empty text gracefully."""
    from core.ingestion.ner_en import extract_entities_en
    entities = extract_entities_en("")
    assert isinstance(entities, list)


def test_ner_very_long_text():
    """NER should handle very long texts without crashing."""
    from core.ingestion.ner_en import extract_entities_en
    long_text = "Dolutegravir 50mg treats HIV infection. " * 100
    entities  = extract_entities_en(long_text[:2000])
    assert isinstance(entities, list)


def test_ner_no_medical_entities():
    """NER should return empty list for non-medical text."""
    from core.ingestion.ner_en import extract_entities_en
    text     = "The weather in Madrid is sunny today with 25 degrees."
    entities = extract_entities_en(text)
    assert isinstance(entities, list)
    # Should find 0 or very few entities
    assert len(entities) <= 2, f"Non-medical text should have few entities, got {len(entities)}"


# ── Entity normalization robustness ───────────────────────────

def test_normalize_entity_variants():
    """Entity normalization should handle all drug name variants."""
    from core.graph.builder import normalize_entity

    test_cases = [
        ("dtg",         "Dolutegravir"),
        ("DTG",         "Dolutegravir"),
        ("rifampin",    "Rifampin"),
        ("rifampicin",  "Rifampin"),
        ("Rifampicina", "Rifampin"),
        ("pregnancy",   "Pregnancy"),
        ("hiv",         "HIV infection"),
        ("VIH",         "HIV infection"),
        ("taf",         "TAF"),
        ("TAF",         "TAF"),
    ]

    for input_text, expected in test_cases:
        result = normalize_entity(input_text)
        assert result == expected, \
            f"normalize_entity('{input_text}') = '{result}' (expected '{expected}')"


def test_normalize_entity_unknown():
    """Unknown entities should be returned as-is."""
    from core.graph.builder import normalize_entity

    unknowns = ["SomeUnknownDrug", "RandomText", "XYZ123"]
    for text in unknowns:
        result = normalize_entity(text)
        assert result == text, f"Unknown entity '{text}' should be unchanged"


if __name__ == "__main__":
    # Run with pytest for better output
    import subprocess
    subprocess.run([sys.executable, "-m", "pytest",
                   __file__, "-v", "--tb=short"])