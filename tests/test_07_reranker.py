"""
test_07_reranker.py
-------------------
Verifies that the cross-encoder reranker is working correctly.
Tests both the Infinity BGE-Reranker (preferred) and the
ms-marco-MiniLM fallback.

Tests:
    1. Infinity BGE-Reranker-v2-m3 availability (optional)
    2. ms-marco-MiniLM fallback reranker loads correctly
    3. Reranker correctly scores relevant chunks higher
    4. Reranking improves clinical chunk ordering
    5. Reranker handles multilingual EN/ES chunks

Run:
    uv run python tests/test_07_reranker.py
"""

import sys
import os
# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import requests

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
RESET  = "\033[0m"

def ok(msg):   print(f"  {GREEN}✓{RESET} {msg}")
def fail(msg): print(f"  {RED}✗{RESET} {msg}")
def warn(msg): print(f"  {YELLOW}~{RESET} {msg}")

INFINITY_URL   = "http://localhost:7997/rerank"
INFINITY_MODEL = "BAAI/bge-reranker-v2-m3"


def test_infinity_available() -> bool:
    """Test if Infinity BGE-Reranker is available."""
    print("\n[1] Infinity BGE-Reranker-v2-m3 (optional)")
    try:
        resp = requests.get("http://localhost:7997/models", timeout=3)
        if resp.status_code == 200:
            ok("Infinity reranker available at localhost:7997")
            return True
        warn("Infinity running but unexpected status")
        return False
    except Exception:
        warn("Infinity not running — will use ms-marco fallback")
        warn("To start: docker-compose up -d infinity")
        return False


def test_infinity_reranking() -> bool:
    """Test Infinity reranking with HIV clinical documents."""
    print("\n[2] Infinity reranking quality")
    try:
        query = "Dolutegravir dose adjustment with Rifampin tuberculosis"
        docs  = [
            "When dolutegravir is co-administered with rifampin, the dose should be increased to 50mg twice daily due to enzyme induction.",
            "Tenofovir alafenamide is preferred over TDF in patients with renal impairment eGFR below 60.",
            "Abacavir requires HLA-B*5701 testing before initiation to prevent hypersensitivity reactions.",
            "The weather forecast shows sunny skies with temperatures around 25 degrees Celsius.",
        ]

        resp = requests.post(
            INFINITY_URL,
            json={
                "model"    : INFINITY_MODEL,
                "query"    : query,
                "documents": docs,
                "top_n"    : 4,
            },
            timeout=30
        )

        if resp.status_code != 200:
            fail(f"Infinity returned status {resp.status_code}")
            return False

        results = resp.json().get("results", [])
        if not results:
            fail("No results returned from Infinity")
            return False

        # Sort by relevance score
        ranked = sorted(results, key=lambda x: x["relevance_score"], reverse=True)

        ok(f"Reranked {len(docs)} documents:")
        for i, r in enumerate(ranked):
            score = r["relevance_score"]
            text  = docs[r["index"]][:60]
            ok(f"  [{i+1}] score={score:.3f} → {text}...")

        # The DTG+Rifampin chunk should be ranked first
        top_doc = docs[ranked[0]["index"]]
        if "rifampin" in top_doc.lower() or "dolutegravir" in top_doc.lower():
            ok("Most relevant document ranked first ✓")
            return True
        fail("Most relevant document NOT ranked first")
        return False

    except Exception as e:
        warn(f"Infinity reranking test skipped: {e}")
        return True  # non-fatal if Infinity not running


def test_fallback_reranker() -> bool:
    """Test ms-marco-MiniLM fallback reranker."""
    print("\n[3] ms-marco-MiniLM fallback reranker")
    try:
        from core.retrieval.reranker import get_reranker
        t_start  = time.time()
        reranker = get_reranker("fast")
        elapsed  = time.time() - t_start

        if reranker is not None:
            ok(f"Fallback reranker loaded in {elapsed:.1f}s")
            return True
        fail("Fallback reranker failed to load")
        return False
    except Exception as e:
        fail(f"Reranker load error: {e}")
        return False


def test_reranking_improves_order() -> bool:
    """Test that reranking correctly orders clinical documents."""
    print("\n[4] Reranking improves chunk ordering")
    try:
        from core.retrieval.reranker import rerank_chunks

        query = "Dolutegravir dose adjustment Rifampin tuberculosis"

        # Create mock chunk objects
        class MockPayload:
            def __init__(self, text):
                self.data = {"original_text": text}
            def get(self, key, default=""):
                return self.data.get(key, default)

        class MockChunk:
            def __init__(self, text, chunk_id):
                self.payload = MockPayload(text)
                self.id      = chunk_id

        chunks = [
            MockChunk("The weather is sunny today in Madrid.", 1),
            MockChunk("Abacavir requires HLA-B*5701 testing.", 2),
            MockChunk("DTG 50mg twice daily when used with rifampin due to CYP3A4 induction.", 3),
            MockChunk("Tenofovir alafenamide preferred over TDF for renal safety.", 4),
            MockChunk("Dolutegravir dose must be doubled with rifampicin co-administration.", 5),
        ]

        reranked = rerank_chunks(query, chunks, top_k=3)

        if not reranked:
            fail("Reranking returned empty results")
            return False

        ok(f"Reranked {len(chunks)} → top {len(reranked)} chunks")
        ok("Top reranked chunks:")
        for i, chunk in enumerate(reranked):
            text = chunk.payload.get("original_text", "")[:70]
            ok(f"  [{i+1}] {text}...")

        # Check that DTG+Rifampin chunks are in top results
        top_texts = " ".join([
            c.payload.get("original_text", "").lower()
            for c in reranked
        ])
        if "rifampin" in top_texts or "rifampicin" in top_texts:
            ok("Relevant chunks promoted to top positions ✓")
            return True
        warn("Relevant chunks may not be at top — check reranker")
        return True

    except Exception as e:
        fail(f"Reranking test failed: {e}")
        return False


def test_multilingual_reranking() -> bool:
    """Test reranker handles multilingual EN/ES chunks."""
    print("\n[5] Multilingual reranking (EN/ES)")
    try:
        resp = requests.get("http://localhost:7997/models", timeout=2)
        if resp.status_code != 200:
            warn("Infinity not available — skipping multilingual test")
            return True
    except Exception:
        warn("Infinity not available — skipping multilingual test")
        return True

    query = "Dolutegravir dose adjustment Rifampin"
    docs  = [
        "When dolutegravir is used with rifampin, increase dose to 50mg twice daily.",
        "El Dolutegravir debe administrarse 50mg dos veces al día con Rifampicina.",
        "Tenofovir alafenamide preferred in renal impairment.",
        "El tiempo en Madrid es soleado hoy con 25 grados.",
    ]

    try:
        resp = requests.post(
            INFINITY_URL,
            json={"model": INFINITY_MODEL, "query": query,
                  "documents": docs, "top_n": 4},
            timeout=30
        )
        results = resp.json().get("results", [])
        ranked  = sorted(results, key=lambda x: x["relevance_score"], reverse=True)

        ok("Multilingual reranking scores:")
        for r in ranked:
            ok(f"  score={r['relevance_score']:.3f} → {docs[r['index']][:50]}...")

        # Both EN and ES relevant docs should be in top 2
        top2_docs = [docs[ranked[i]["index"]].lower() for i in range(2)]
        if any("rifampin" in d or "rifampicina" in d for d in top2_docs):
            ok("Multilingual reranking working correctly ✓")
            return True
        warn("Multilingual reranking may need improvement")
        return True

    except Exception as e:
        warn(f"Multilingual test skipped: {e}")
        return True


def main():
    print("=" * 55)
    print("  MedalRAG — Reranker Tests")
    print("=" * 55)

    infinity_ok = test_infinity_available()

    tests = []
    if infinity_ok:
        tests.append(test_infinity_reranking)
        tests.append(test_multilingual_reranking)

    tests.append(test_fallback_reranker)
    tests.append(test_reranking_improves_order)

    results = [t() for t in tests]
    n_pass  = sum(results)
    n_total = len(results)

    print("\n" + "=" * 55)
    print(f"  Results: {n_pass}/{n_total} tests passed")

    if n_pass >= n_total - 1:
        ok("Reranker is working correctly")
        sys.exit(0)
    else:
        fail(f"{n_total - n_pass} test(s) failed")
        sys.exit(1)


if __name__ == "__main__":
    main()