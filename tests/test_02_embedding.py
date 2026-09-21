"""
test_02_embedding.py
--------------------
Verifies that the BGE-M3 embedding model is working correctly
via Ollama, and that embeddings are consistent and meaningful.

Tests:
    1. BGE-M3 returns a valid 1024-dimensional vector
    2. Embeddings are not all zeros
    3. Similar texts have higher cosine similarity than unrelated ones
    4. EN and ES medical terms embed in a similar space (multilingual)

Run:
    uv run python tests/test_02_embedding.py
"""

import sys
import math
import requests

GREEN = "\033[92m"
RED   = "\033[91m"
RESET = "\033[0m"

def ok(msg):   print(f"  {GREEN}✓{RESET} {msg}")
def fail(msg): print(f"  {RED}✗{RESET} {msg}")


OLLAMA_URL = "http://localhost:11434/api/embed"
MODEL      = "bge-m3"
DIM        = 1024


# ── Helpers ────────────────────────────────────────────────────

def get_embedding(text: str) -> list:
    resp = requests.post(
        OLLAMA_URL,
        json={"model": MODEL, "input": text[:8000]},
        timeout=60
    )
    return resp.json().get("embeddings", [[]])[0]


def cosine_similarity(v1: list, v2: list) -> float:
    dot     = sum(a * b for a, b in zip(v1, v2))
    norm_v1 = math.sqrt(sum(a * a for a in v1))
    norm_v2 = math.sqrt(sum(b * b for b in v2))
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    return dot / (norm_v1 * norm_v2)


# ── Tests ──────────────────────────────────────────────────────

def test_embedding_dimension() -> bool:
    """Test that BGE-M3 returns a 1024-dimensional vector."""
    print("\n[1] Embedding dimension")
    try:
        emb = get_embedding("Dolutegravir HIV treatment")
        if len(emb) == DIM:
            ok(f"Embedding dimension correct: {DIM}d")
            return True
        fail(f"Wrong dimension: {len(emb)} (expected {DIM})")
        return False
    except Exception as e:
        fail(f"Embedding failed: {e}")
        return False


def test_embedding_not_zero() -> bool:
    """Test that embeddings are not all zeros."""
    print("\n[2] Embedding non-zero values")
    try:
        emb = get_embedding("Dolutegravir HIV treatment")
        if all(v == 0.0 for v in emb):
            fail("Embedding is all zeros — model not loaded correctly")
            return False
        nonzero = sum(1 for v in emb if v != 0.0)
        ok(f"Embedding has {nonzero}/{DIM} non-zero values")
        return True
    except Exception as e:
        fail(f"Test failed: {e}")
        return False


def test_semantic_similarity() -> bool:
    """
    Test that semantically similar texts have higher cosine similarity
    than unrelated texts.
    """
    print("\n[3] Semantic similarity")
    try:
        # Similar texts (both about HIV treatment)
        emb_a = get_embedding(
            "Dolutegravir 50mg twice daily when co-administered with Rifampin"
        )
        emb_b = get_embedding(
            "DTG dose adjustment required with Rifampicin for tuberculosis"
        )
        # Unrelated text
        emb_c = get_embedding(
            "The weather in Madrid is sunny today with 25 degrees"
        )

        sim_similar   = cosine_similarity(emb_a, emb_b)
        sim_unrelated = cosine_similarity(emb_a, emb_c)

        ok(f"Similar texts cosine similarity    : {sim_similar:.3f}")
        ok(f"Unrelated texts cosine similarity  : {sim_unrelated:.3f}")

        if sim_similar > sim_unrelated:
            ok("Similar texts are closer in embedding space ✓")
            return True
        fail(f"Similar texts should be closer ({sim_similar:.3f}) than unrelated ({sim_unrelated:.3f})")
        return False
    except Exception as e:
        fail(f"Test failed: {e}")
        return False


def test_multilingual_embedding() -> bool:
    """
    Test that BGE-M3 maps EN and ES medical terms to similar vectors.
    This validates the multilingual capability for EN+ES HIV guidelines.
    """
    print("\n[4] Multilingual embedding (EN/ES)")
    try:
        pairs = [
            ("Dolutegravir HIV treatment pregnancy", "Dolutegravir tratamiento VIH embarazo"),
            ("Tenofovir renal impairment safety",   "Tenofovir seguridad renal"),
            ("antiretroviral therapy tuberculosis",  "tratamiento antirretroviral tuberculosis"),
        ]

        all_pass = True
        for text_en, text_es in pairs:
            emb_en = get_embedding(text_en)
            emb_es = get_embedding(text_es)
            sim    = cosine_similarity(emb_en, emb_es)

            if sim > 0.75:
                ok(f"EN/ES similarity {sim:.3f} > 0.75  '{text_en[:40]}'")
            else:
                fail(f"EN/ES similarity too low {sim:.3f} < 0.75  '{text_en[:40]}'")
                all_pass = False

        return all_pass
    except Exception as e:
        fail(f"Test failed: {e}")
        return False


def test_batch_consistency() -> bool:
    """Test that the same text always produces the same embedding."""
    print("\n[5] Embedding consistency")
    try:
        text = "HIV treatment with Dolutegravir and TAF/FTC"
        emb1 = get_embedding(text)
        emb2 = get_embedding(text)
        sim  = cosine_similarity(emb1, emb2)

        if sim > 0.999:
            ok(f"Same text produces consistent embeddings (sim={sim:.4f})")
            return True
        fail(f"Inconsistent embeddings (sim={sim:.4f}) — model might be non-deterministic")
        return False
    except Exception as e:
        fail(f"Test failed: {e}")
        return False


# ── Main ───────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  MedalRAG — Embedding (BGE-M3) Tests")
    print("=" * 55)

    tests = [
        test_embedding_dimension,
        test_embedding_not_zero,
        test_semantic_similarity,
        test_multilingual_embedding,
        test_batch_consistency,
    ]

    results  = [t() for t in tests]
    n_pass   = sum(results)
    n_total  = len(results)

    print("\n" + "=" * 55)
    print(f"  Results: {n_pass}/{n_total} tests passed")

    if all(results):
        ok("BGE-M3 embedding is working correctly")
        sys.exit(0)
    else:
        fail(f"{n_total - n_pass} test(s) failed")
        sys.exit(1)


if __name__ == "__main__":
    main()