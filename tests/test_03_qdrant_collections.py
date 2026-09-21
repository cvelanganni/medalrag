"""
test_03_qdrant_collections.py
------------------------------
Verifies that Qdrant collections are properly indexed
with the correct vector dimensions and a sufficient number
of chunks for both EN and ES guidelines.

Tests:
    1. Collections exist (medical_docs_en, medical_docs_es)
    2. Vector dimensions are correct (1024d for BGE-M3)
    3. Minimum chunk counts are met
    4. Section type distribution is reasonable
    5. Payloads contain required fields

Run:
    uv run python tests/test_03_qdrant_collections.py
"""

import sys
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
RESET  = "\033[0m"

def ok(msg):   print(f"  {GREEN}✓{RESET} {msg}")
def fail(msg): print(f"  {RED}✗{RESET} {msg}")
def warn(msg): print(f"  {YELLOW}~{RESET} {msg}")

QDRANT_URL      = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_EN   = os.getenv("QDRANT_COLLECTION_EN", "medical_docs_en")
COLLECTION_ES   = os.getenv("QDRANT_COLLECTION_ES", "medical_docs_es")
EXPECTED_DIM    = 1024
MIN_CHUNKS_EN   = 5000
MIN_CHUNKS_ES   = 500
REQUIRED_FIELDS = ["text", "filename", "page", "lang"]


def test_collections_exist(q: QdrantClient) -> bool:
    """Test that both EN and ES collections exist."""
    print("\n[1] Collections existence")
    all_ok = True
    for col in [COLLECTION_EN, COLLECTION_ES]:
        if q.collection_exists(col):
            ok(f"Collection exists: {col}")
        else:
            fail(f"Collection MISSING: {col} — run: uv run build_pipeline.py")
            all_ok = False
    return all_ok


def test_vector_dimensions(q: QdrantClient) -> bool:
    """Test that vector dimensions match BGE-M3 (1024d)."""
    print("\n[2] Vector dimensions")
    all_ok = True
    for col in [COLLECTION_EN, COLLECTION_ES]:
        try:
            info = q.get_collection(col)
            dim  = info.config.params.vectors.size
            if dim == EXPECTED_DIM:
                ok(f"{col}: {dim}d vectors ✓")
            else:
                fail(f"{col}: wrong dimension {dim}d (expected {EXPECTED_DIM}d)")
                all_ok = False
        except Exception as e:
            fail(f"{col}: could not get info — {e}")
            all_ok = False
    return all_ok


def test_chunk_counts(q: QdrantClient) -> bool:
    """Test that collections have enough indexed chunks."""
    print("\n[3] Chunk counts")
    all_ok = True

    count_en = q.count(COLLECTION_EN).count
    count_es = q.count(COLLECTION_ES).count

    if count_en >= MIN_CHUNKS_EN:
        ok(f"EN chunks: {count_en} (min: {MIN_CHUNKS_EN})")
    else:
        fail(f"EN chunks too low: {count_en} (min: {MIN_CHUNKS_EN})")
        all_ok = False

    if count_es >= MIN_CHUNKS_ES:
        ok(f"ES chunks: {count_es} (min: {MIN_CHUNKS_ES})")
    else:
        fail(f"ES chunks too low: {count_es} (min: {MIN_CHUNKS_ES})")
        all_ok = False

    return all_ok


def test_section_type_distribution(q: QdrantClient) -> bool:
    """Test that section_type distribution is reasonable."""
    print("\n[4] Section type distribution")
    all_ok = True

    for col in [COLLECTION_EN, COLLECTION_ES]:
        points, _ = q.scroll(
            collection_name = col,
            limit           = 500,
            with_payload    = True,
            with_vectors    = False,
        )

        types = {}
        for p in points:
            stype = p.payload.get("section_type", "unknown")
            types[stype] = types.get(stype, 0) + 1

        total    = sum(types.values())
        clinical = types.get("clinical", 0)
        pct      = clinical / total * 100 if total > 0 else 0

        ok(f"{col} section types: {dict(sorted(types.items()))}")

        if pct >= 20:
            ok(f"{col} clinical chunks: {clinical}/{total} ({pct:.1f}%) ✓")
        else:
            warn(f"{col} low clinical chunks: {pct:.1f}% (expected ≥20%)")

    return all_ok


def test_payload_fields(q: QdrantClient) -> bool:
    """Test that payloads contain required fields."""
    print("\n[5] Payload fields")
    all_ok = True

    for col in [COLLECTION_EN, COLLECTION_ES]:
        points, _ = q.scroll(
            collection_name = col,
            limit           = 10,
            with_payload    = True,
            with_vectors    = False,
        )

        if not points:
            fail(f"{col}: no points found")
            all_ok = False
            continue

        sample  = points[0].payload
        missing = [f for f in REQUIRED_FIELDS if f not in sample]

        if not missing:
            ok(f"{col}: all required fields present {REQUIRED_FIELDS}")
        else:
            fail(f"{col}: missing fields {missing}")
            all_ok = False

    return all_ok


def main():
    print("=" * 55)
    print("  MedalRAG — Qdrant Collections Tests")
    print("=" * 55)

    qdrant_key = os.getenv("QDRANT_API_KEY") or None
    try:
        q = QdrantClient(url=QDRANT_URL, api_key=qdrant_key)
    except Exception as e:
        fail(f"Cannot connect to Qdrant at {QDRANT_URL}: {e}")
        sys.exit(1)

    tests = [
        lambda: test_collections_exist(q),
        lambda: test_vector_dimensions(q),
        lambda: test_chunk_counts(q),
        lambda: test_section_type_distribution(q),
        lambda: test_payload_fields(q),
    ]

    results = [t() for t in tests]
    n_pass  = sum(results)
    n_total = len(results)

    print("\n" + "=" * 55)
    print(f"  Results: {n_pass}/{n_total} tests passed")

    if all(results):
        ok("Qdrant collections are properly indexed")
        sys.exit(0)
    else:
        fail(f"{n_total - n_pass} test(s) failed")
        sys.exit(1)


if __name__ == "__main__":
    main()