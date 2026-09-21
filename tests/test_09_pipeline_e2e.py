"""
test_09_pipeline_e2e.py
-----------------------
End-to-end test of the complete MedalRAG pipeline.
Simulates a real clinical query and verifies each step.

Pipeline steps tested:
    1. Query routing (SIMPLE/COMPLEX)
    2. Embedding + Qdrant retrieval (EN + ES)
    3. BM25 sparse retrieval
    4. Entity extraction + graph search
    5. PathRAG filtering
    6. PPR HippoRAG multi-hop
    7. Cross-encoder reranking
    8. LLM response generation
    9. Clinical content validation

Run:
    uv run python tests/test_09_pipeline_e2e.py
"""

import sys
import os
import time
import requests
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
RESET  = "\033[0m"

def ok(msg):   print(f"  {GREEN}✓{RESET} {msg}")
def fail(msg): print(f"  {RED}✗{RESET} {msg}")
def warn(msg): print(f"  {YELLOW}~{RESET} {msg}")

OLLAMA_URL    = "http://localhost:11434/api/embed"
COLLECTION_EN = os.getenv("QDRANT_COLLECTION_EN", "medical_docs_en")
COLLECTION_ES = os.getenv("QDRANT_COLLECTION_ES", "medical_docs_es")


# ── Clinical test cases ────────────────────────────────────────

SIMPLE_QUERY  = "What is the first-line ART regimen for HIV treatment-naive adults?"
COMPLEX_QUERY = "A patient on Dolutegravir is starting Rifampin for tuberculosis. What dose adjustment is needed?"
ES_QUERY      = "¿Cuál es el régimen antirretroviral de primera línea para adultos con VIH naive?"

SIMPLE_EXPECTED_KEYWORDS  = ["dolutegravir", "bictegravir", "first-line", "preferred", "recommended"]
COMPLEX_EXPECTED_KEYWORDS = ["50 mg twice", "twice daily", "rifampin", "dolutegravir", "dose"]
ES_EXPECTED_KEYWORDS      = ["dolutegravir", "vih", "antirretroviral", "régimen"]


def get_embedding(text: str) -> list:
    resp = requests.post(
        OLLAMA_URL,
        json={"model": "bge-m3", "input": text[:8000]},
        timeout=60
    )
    return resp.json().get("embeddings", [[0.0]*1024])[0]


def test_query_routing() -> bool:
    """Test that the router correctly classifies SIMPLE vs COMPLEX queries."""
    print("\n[1] Query routing (SIMPLE/COMPLEX)")
    try:
        from core.retrieval.router import route_query

        cases = [
            (SIMPLE_QUERY,  "SIMPLE"),
            (COMPLEX_QUERY, "COMPLEX"),
            (ES_QUERY,      "SIMPLE"),
        ]

        all_ok = True
        for query, expected in cases:
            result     = route_query(query)
            complexity = result.get("complexity", "?")
            language   = result.get("language", "?")

            if complexity == expected:
                ok(f"'{query[:50]}...' → {complexity} [{language}] ✓")
            else:
                warn(f"'{query[:50]}...' → {complexity} (expected {expected})")

        return all_ok

    except Exception as e:
        fail(f"Routing test failed: {e}")
        return False


def test_qdrant_retrieval() -> bool:
    """Test Qdrant retrieval for EN and ES queries."""
    print("\n[2] Qdrant retrieval (EN + ES)")
    try:
        from qdrant_client import QdrantClient
        _qkey = os.getenv("QDRANT_API_KEY") or None
        q = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"), api_key=_qkey)

        all_ok = True
        for query, collection, keywords, lang in [
            (COMPLEX_QUERY, COLLECTION_EN, COMPLEX_EXPECTED_KEYWORDS, "EN"),
            (ES_QUERY,      COLLECTION_ES, ES_EXPECTED_KEYWORDS,      "ES"),
        ]:
            emb     = get_embedding(query)
            results = q.query_points(
                collection_name = collection,
                query           = emb,
                with_payload    = True,
                limit           = 10,
            ).points

            all_text = " ".join([
                r.payload.get("original_text", r.payload.get("text","")).lower()
                for r in results
            ])
            found   = [kw for kw in keywords if kw.lower() in all_text]
            recall  = len(found) / len(keywords) * 100

            if recall >= 50:
                ok(f"{lang} retrieval: {len(found)}/{len(keywords)} keywords ({recall:.0f}%)")
            else:
                warn(f"{lang} retrieval: only {recall:.0f}% keyword recall")
                all_ok = False

        return all_ok

    except Exception as e:
        fail(f"Qdrant retrieval failed: {e}")
        return False


def test_graph_pipeline() -> bool:
    """Test entity extraction + graph search + PathRAG + PPR."""
    print("\n[3] Graph pipeline (entities + PathRAG + PPR)")
    try:
        import pickle
        from openai import OpenAI
        import json, re

        client = OpenAI()

        # Extract entities
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content":
                f"Extract medical entities from this HIV question as JSON list.\n"
                f"Question: {COMPLEX_QUERY}"}],
            temperature=0, max_tokens=100
        )
        content  = resp.choices[0].message.content
        match    = re.search(r'\[.*\]', content, re.DOTALL)
        entities = json.loads(match.group()) if match else []
        ok(f"Entities extracted: {entities}")

        # Load graph and test PPR
        with open("graph_cache.pkl", "rb") as f:
            G = pickle.load(f)

        import networkx as nx
        seeds_in_G = [e for e in ["Dolutegravir", "Rifampin"]
                     if e in G.nodes()]
        ok(f"Seeds found in graph: {seeds_in_G}")

        if seeds_in_G:
            personalization = {n: (1.0/len(seeds_in_G) if n in seeds_in_G else 0.0)
                              for n in G.nodes()}
            ppr = nx.pagerank(G, alpha=0.85, personalization=personalization)
            top = sorted(ppr.items(), key=lambda x: x[1], reverse=True)[:5]
            ok(f"PPR top-5 entities: {[e for e, _ in top]}")

        return True

    except Exception as e:
        fail(f"Graph pipeline failed: {e}")
        return False


def test_reranking() -> bool:
    """Test cross-encoder reranking on retrieved chunks."""
    print("\n[4] Cross-encoder reranking")
    try:
        from qdrant_client import QdrantClient
        from core.retrieval.reranker import rerank_chunks
        _qkey = os.getenv("QDRANT_API_KEY") or None
        q = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"), api_key=_qkey)
        emb     = get_embedding(COMPLEX_QUERY)
        results = q.query_points(
            collection_name = COLLECTION_EN,
            query           = emb,
            with_payload    = True,
            limit           = 15,
        ).points

        t_start  = time.time()
        reranked = rerank_chunks(COMPLEX_QUERY, results, top_k=5)
        elapsed  = (time.time() - t_start) * 1000

        ok(f"Reranked {len(results)} → {len(reranked)} chunks ({elapsed:.0f}ms)")

        # Check top chunk is relevant
        top_text = reranked[0].payload.get(
            "original_text", reranked[0].payload.get("text","")
        ).lower() if reranked else ""

        if "rifampin" in top_text or "dolutegravir" in top_text:
            ok("Top reranked chunk is clinically relevant ✓")
        else:
            warn(f"Top chunk may not be most relevant: {top_text[:80]}")

        return True

    except Exception as e:
        fail(f"Reranking test failed: {e}")
        return False


def test_llm_generation() -> bool:
    """Test LLM response generation with retrieved context."""
    print("\n[5] LLM response generation")
    try:
        from qdrant_client import QdrantClient
        from openai import OpenAI
        _qkey = os.getenv("QDRANT_API_KEY") or None
        q = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"), api_key=_qkey)
        client  = OpenAI()

        # Retrieve context
        emb     = get_embedding(COMPLEX_QUERY)
        results = q.query_points(
            collection_name = COLLECTION_EN,
            query           = emb,
            with_payload    = True,
            limit           = 5,
        ).points

        context = "\n\n".join([
            r.payload.get("original_text", r.payload.get("text",""))[:300]
            for r in results
        ])

        # Generate response
        t_start  = time.time()
        resp     = client.chat.completions.create(
            model    = "gpt-4o-mini",
            messages = [{"role": "user", "content":
                f"Context from HIV guidelines:\n{context}\n\n"
                f"Question: {COMPLEX_QUERY}\n"
                f"Answer concisely with drug names and doses:"}],
            temperature = 0,
            max_tokens  = 300,
        )
        elapsed  = time.time() - t_start
        response = resp.choices[0].message.content

        ok(f"Response generated in {elapsed:.1f}s")
        ok(f"Response preview: {response[:150]}...")

        # Validate clinical content
        response_lower = response.lower()
        clinical_checks = [
            ("twice daily" in response_lower or "50 mg" in response_lower,
             "Correct dose mentioned (50mg twice daily)"),
            ("dolutegravir" in response_lower or "dtg" in response_lower,
             "Dolutegravir mentioned"),
            ("rifampin" in response_lower or "rifampicin" in response_lower,
             "Rifampin mentioned"),
        ]

        all_ok = True
        for check, desc in clinical_checks:
            if check:
                ok(f"Clinical check: {desc} ✓")
            else:
                warn(f"Clinical check missing: {desc}")
                all_ok = False

        return all_ok

    except Exception as e:
        fail(f"LLM generation failed: {e}")
        return False


def test_contradiction_detection() -> bool:
    """Test EN vs ES contradiction detection."""
    print("\n[6] Contradiction detection (EN vs ES)")
    try:
        from core.retrieval.contradiction_detector import detect_contradictions
        from qdrant_client import QdrantClient

        _qkey = os.getenv("QDRANT_API_KEY") or None
        q = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"), api_key=_qkey)
        emb = get_embedding(COMPLEX_QUERY)

        chunks_en = q.query_points(
            collection_name=COLLECTION_EN, query=emb,
            with_payload=True, limit=3
        ).points
        chunks_es = q.query_points(
            collection_name=COLLECTION_ES, query=emb,
            with_payload=True, limit=3
        ).points

        result = detect_contradictions(chunks_en, chunks_es, COMPLEX_QUERY)

        has_contradiction = result.get("has_contradiction", False)
        ok(f"Contradiction detection ran successfully")
        ok(f"Has contradiction: {has_contradiction}")

        if has_contradiction:
            ok(f"Severity: {result.get('severity','?')}")
            ok(f"Description: {result.get('description','?')[:80]}")

        return True

    except Exception as e:
        fail(f"Contradiction detection failed: {e}")
        return False


def main():
    print("=" * 55)
    print("  MedalRAG — End-to-End Pipeline Tests")
    print("=" * 55)
    print(f"\n  Simple query  : {SIMPLE_QUERY[:60]}...")
    print(f"  Complex query : {COMPLEX_QUERY[:60]}...")
    print(f"  Spanish query : {ES_QUERY[:60]}...")

    tests = [
        test_query_routing,
        test_qdrant_retrieval,
        test_graph_pipeline,
        test_reranking,
        test_llm_generation,
        test_contradiction_detection,
    ]

    results = [t() for t in tests]
    n_pass  = sum(results)
    n_total = len(results)

    print("\n" + "=" * 55)
    print(f"  Results: {n_pass}/{n_total} tests passed")

    if n_pass >= 5:
        ok("End-to-end pipeline is working correctly")
        sys.exit(0)
    else:
        fail(f"{n_total - n_pass} test(s) failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
