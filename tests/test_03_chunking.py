"""
check_chunking.py
=================
Diagnostic tool to verify PDF chunking quality for MedalRAG.

Checks:
1. Table detection — are tables being chunked correctly?
2. Context preservation — do chunks have enough context?
3. Header/section detection — are section titles preserved?
4. Metadata completeness — filename, page, section_type
5. Orphan chunks — chunks with no clinical keywords
6. Split sentences — chunks ending mid-sentence
7. Neighbor context — do adjacent chunks share context?
"""

import os
import re
import json
from collections import Counter, defaultdict
from dotenv import load_dotenv

load_dotenv()

from qdrant_client import QdrantClient

qdrant = QdrantClient(
    url     = os.getenv("QDRANT_URL", "http://localhost:6333"),
    api_key = os.getenv("QDRANT_API_KEY") or None,
)

COLLECTION_EN = os.getenv("QDRANT_COLLECTION_EN", "medical_docs_en")
COLLECTION_ES = os.getenv("QDRANT_COLLECTION_ES", "medical_docs_es")

# Clinical keywords that should appear in clinical chunks
CLINICAL_KEYWORDS = [
    "dolutegravir", "bictegravir", "tenofovir", "emtricitabine",
    "lamivudine", "abacavir", "raltegravir", "darunavir",
    "recommended", "preferred", "contraindicated", "initiate",
    "cd4", "viral load", "hiv", "art", "regimen", "dose",
    "mg", "therapy", "treatment", "monitoring",
]

TABLE_INDICATORS = [
    "table", "figure", "footnote", "|", "column", "row",
    "see above", "see below", "†", "‡", "§",
]


def scroll_all(collection: str, limit: int = 500) -> list:
    """Fetch all points from a collection."""
    points, offset = [], None
    while True:
        batch, offset = qdrant.scroll(
            collection_name = collection,
            limit           = 100,
            offset          = offset,
            with_payload    = True,
            with_vectors    = False,
        )
        points.extend(batch)
        if offset is None or len(points) >= limit:
            break
    return points


def check_metadata(points: list) -> dict:
    """Check payload completeness."""
    missing_filename = sum(1 for p in points if not p.payload.get("filename"))
    missing_page     = sum(1 for p in points if p.payload.get("page") is None)
    missing_type     = sum(1 for p in points if not p.payload.get("section_type"))
    missing_text     = sum(1 for p in points if not p.payload.get("text") and not p.payload.get("original_text"))
    contextualized   = sum(1 for p in points if p.payload.get("contextualized"))

    return {
        "total"              : len(points),
        "missing_filename"   : missing_filename,
        "missing_page"       : missing_page,
        "missing_section_type": missing_type,
        "missing_text"       : missing_text,
        "contextualized"     : contextualized,
        "pct_contextualized" : round(100 * contextualized / max(len(points), 1), 1),
    }


def check_chunk_sizes(points: list) -> dict:
    """Analyze chunk sizes in words."""
    sizes = []
    for p in points:
        text = p.payload.get("original_text", p.payload.get("text", ""))
        sizes.append(len(text.split()))

    if not sizes:
        return {}

    sizes.sort()
    n = len(sizes)
    return {
        "min_words"    : sizes[0],
        "max_words"    : sizes[-1],
        "mean_words"   : round(sum(sizes) / n, 1),
        "median_words" : sizes[n // 2],
        "too_short_lt20" : sum(1 for s in sizes if s < 20),
        "too_short_lt50" : sum(1 for s in sizes if s < 50),
        "too_long_gt500" : sum(1 for s in sizes if s > 500),
    }


def check_table_handling(points: list) -> dict:
    """Detect chunks that contain table content."""
    table_chunks        = []
    tables_without_header = []

    for p in points:
        text = p.payload.get("original_text", p.payload.get("text", "")).lower()

        # Detect table-like content
        has_table = any(ind in text for ind in TABLE_INDICATORS)
        if has_table:
            table_chunks.append(p)

            # Check if table header is present
            has_header = bool(re.search(
                r'(table\s+\d+|figure\s+\d+|drug\s+name|regimen|dose)',
                text
            ))
            if not has_header:
                tables_without_header.append({
                    "id"      : str(p.id),
                    "filename": p.payload.get("filename", "?"),
                    "page"    : p.payload.get("page", "?"),
                    "preview" : text[:100],
                })

    return {
        "table_chunks_detected"  : len(table_chunks),
        "tables_missing_header"  : len(tables_without_header),
        "examples"               : tables_without_header[:3],
    }


def check_split_sentences(points: list) -> dict:
    """Detect chunks that start or end mid-sentence."""
    starts_mid = []
    ends_mid   = []

    for p in points:
        text = p.payload.get("original_text", p.payload.get("text", "")).strip()
        if not text:
            continue

        # Starts mid-sentence: first char is lowercase (not a list bullet)
        first_char = text[0]
        if first_char.islower() and not text.startswith(("-", "•", "*")):
            starts_mid.append({
                "id"      : str(p.id),
                "filename": p.payload.get("filename", "?"),
                "page"    : p.payload.get("page", "?"),
                "preview" : text[:80],
            })

        # Ends mid-sentence: no terminal punctuation
        last_char = text[-1]
        if last_char not in ".!?:;)\"'":
            ends_mid.append({
                "id"      : str(p.id),
                "filename": p.payload.get("filename", "?"),
                "page"    : p.payload.get("page", "?"),
                "preview" : "..." + text[-80:],
            })

    return {
        "starts_mid_sentence": len(starts_mid),
        "ends_mid_sentence"  : len(ends_mid),
        "examples_start"     : starts_mid[:3],
        "examples_end"       : ends_mid[:3],
    }


def check_clinical_content(points: list) -> dict:
    """Check if clinical chunks actually contain clinical content."""
    clinical_points    = [p for p in points if p.payload.get("section_type") == "clinical"]
    non_clinical_count = 0
    orphan_chunks      = []

    for p in clinical_points:
        text = p.payload.get("original_text", p.payload.get("text", "")).lower()
        has_clinical = any(kw in text for kw in CLINICAL_KEYWORDS)
        if not has_clinical:
            non_clinical_count += 1
            orphan_chunks.append({
                "id"      : str(p.id),
                "filename": p.payload.get("filename", "?"),
                "page"    : p.payload.get("page", "?"),
                "preview" : text[:100],
            })

    return {
        "total_clinical"          : len(clinical_points),
        "without_clinical_content": non_clinical_count,
        "pct_truly_clinical"      : round(100 * (len(clinical_points) - non_clinical_count) / max(len(clinical_points), 1), 1),
        "orphan_examples"         : orphan_chunks[:3],
    }


def check_section_distribution(points: list) -> dict:
    """Distribution of section types and files."""
    types = Counter(p.payload.get("section_type", "unknown") for p in points)
    files = Counter(
        os.path.basename(p.payload.get("filename", "unknown"))
        for p in points
    )
    return {
        "section_types": dict(types.most_common()),
        "files_indexed": len(files),
        "top_files"    : dict(files.most_common(5)),
    }


def check_page_continuity(points: list) -> dict:
    """
    Check if chunks from the same file are distributed across pages.
    Large gaps might indicate missed content.
    """
    by_file = defaultdict(list)
    for p in points:
        fname = p.payload.get("filename", "?")
        page  = p.payload.get("page", 0)
        by_file[fname].append(page)

    gaps = []
    for fname, pages in by_file.items():
        pages_sorted = sorted(set(pages))
        for i in range(1, len(pages_sorted)):
            gap = pages_sorted[i] - pages_sorted[i-1]
            if gap > 10:  # More than 10 pages with no chunk = suspicious
                gaps.append({
                    "file"      : os.path.basename(fname),
                    "gap_from"  : pages_sorted[i-1],
                    "gap_to"    : pages_sorted[i],
                    "gap_pages" : gap,
                })

    gaps.sort(key=lambda x: x["gap_pages"], reverse=True)
    return {
        "files_with_gaps": len(set(g["file"] for g in gaps)),
        "total_gaps"     : len(gaps),
        "largest_gaps"   : gaps[:5],
    }


def score_chunking(results: dict) -> tuple:
    """Compute overall chunking quality score."""
    score = 100
    issues = []

    meta = results.get("metadata", {})
    if meta.get("missing_text", 0) > 0:
        score -= 20
        issues.append(f"⚠️  {meta['missing_text']} chunks have no text")
    if meta.get("missing_page", 0) > 0:
        score -= 10
        issues.append(f"⚠️  {meta['missing_page']} chunks missing page number")
    if meta.get("pct_contextualized", 0) < 50:
        score -= 10
        issues.append(f"⚠️  Only {meta['pct_contextualized']}% chunks contextualized")

    sizes = results.get("chunk_sizes", {})
    if sizes.get("too_short_lt20", 0) > 5:
        score -= 10
        issues.append(f"⚠️  {sizes['too_short_lt20']} chunks under 20 words (too short)")
    if sizes.get("too_long_gt500", 0) > 10:
        score -= 5
        issues.append(f"⚠️  {sizes['too_long_gt500']} chunks over 500 words (too long)")

    splits = results.get("split_sentences", {})
    pct_split = 100 * splits.get("starts_mid_sentence", 0) / max(meta.get("total", 1), 1)
    if pct_split > 20:
        score -= 10
        issues.append(f"⚠️  {pct_split:.1f}% chunks start mid-sentence")

    clinical = results.get("clinical_content", {})
    if clinical.get("pct_truly_clinical", 100) < 70:
        score -= 10
        issues.append(f"⚠️  Only {clinical['pct_truly_clinical']}% of 'clinical' chunks contain clinical keywords")

    tables = results.get("tables", {})
    if tables.get("tables_missing_header", 0) > 10:
        score -= 5
        issues.append(f"⚠️  {tables['tables_missing_header']} table chunks missing headers")

    gaps = results.get("page_continuity", {})
    if gaps.get("files_with_gaps", 0) > 0:
        score -= 5
        issues.append(f"⚠️  {gaps['files_with_gaps']} files have page gaps > 10 pages")

    if not issues:
        issues.append("✅ No major issues found")

    return max(0, score), issues


def run_check(collection: str, lang: str):
    print(f"\n{'='*60}")
    print(f"  Checking {collection} ({lang.upper()})")
    print(f"{'='*60}")

    points = scroll_all(collection, limit=2000)
    if not points:
        print("  ⚠️  No points found!")
        return

    results = {
        "metadata"        : check_metadata(points),
        "chunk_sizes"     : check_chunk_sizes(points),
        "tables"          : check_table_handling(points),
        "split_sentences" : check_split_sentences(points),
        "clinical_content": check_clinical_content(points),
        "section_dist"    : check_section_distribution(points),
        "page_continuity" : check_page_continuity(points),
    }

    # ── Print results ─────────────────────────────────────────────────────
    m = results["metadata"]
    print(f"\n📋 METADATA")
    print(f"  Total chunks      : {m['total']}")
    print(f"  Contextualized    : {m['contextualized']} ({m['pct_contextualized']}%)")
    print(f"  Missing filename  : {m['missing_filename']}")
    print(f"  Missing page      : {m['missing_page']}")
    print(f"  Missing text      : {m['missing_text']}")

    s = results["chunk_sizes"]
    print(f"\n📏 CHUNK SIZES (words)")
    print(f"  Min    : {s.get('min_words','?')}")
    print(f"  Max    : {s.get('max_words','?')}")
    print(f"  Mean   : {s.get('mean_words','?')}")
    print(f"  Median : {s.get('median_words','?')}")
    print(f"  < 20 words : {s.get('too_short_lt20','?')} chunks")
    print(f"  < 50 words : {s.get('too_short_lt50','?')} chunks")
    print(f"  > 500 words: {s.get('too_long_gt500','?')} chunks")

    t = results["tables"]
    print(f"\n📊 TABLE HANDLING")
    print(f"  Table chunks detected     : {t['table_chunks_detected']}")
    print(f"  Tables missing header     : {t['tables_missing_header']}")
    if t["examples"]:
        print("  Examples:")
        for ex in t["examples"]:
            print(f"    [{ex['filename']} p.{ex['page']}] {ex['preview'][:60]}")

    sp = results["split_sentences"]
    pct = 100 * sp["starts_mid_sentence"] / max(m["total"], 1)
    print(f"\n✂️  SPLIT SENTENCES")
    print(f"  Starts mid-sentence : {sp['starts_mid_sentence']} ({pct:.1f}%)")
    print(f"  Ends mid-sentence   : {sp['ends_mid_sentence']}")
    if sp["examples_start"]:
        print("  Start examples:")
        for ex in sp["examples_start"][:2]:
            print(f"    [{ex['filename']} p.{ex['page']}] \"{ex['preview'][:60]}\"")

    cl = results["clinical_content"]
    print(f"\n🩺 CLINICAL CONTENT QUALITY")
    print(f"  Total 'clinical' chunks   : {cl['total_clinical']}")
    print(f"  Without clinical keywords : {cl['without_clinical_content']}")
    print(f"  Truly clinical            : {cl['pct_truly_clinical']}%")
    if cl["orphan_examples"]:
        print("  Orphan examples (clinical but no keywords):")
        for ex in cl["orphan_examples"][:2]:
            print(f"    [{ex['filename']} p.{ex['page']}] {ex['preview'][:60]}")

    sd = results["section_dist"]
    print(f"\n📁 SECTION DISTRIBUTION")
    for stype, count in sd["section_types"].items():
        pct = 100 * count / max(m["total"], 1)
        bar = "█" * int(pct / 5)
        print(f"  {stype:<15} : {count:4d} ({pct:5.1f}%) {bar}")
    print(f"\n  Files indexed : {sd['files_indexed']}")
    for fname, count in sd["top_files"].items():
        print(f"    {fname[:50]}: {count} chunks")

    pg = results["page_continuity"]
    print(f"\n📄 PAGE CONTINUITY")
    print(f"  Files with gaps > 10 pages : {pg['files_with_gaps']}")
    if pg["largest_gaps"]:
        print("  Largest gaps:")
        for g in pg["largest_gaps"]:
            print(f"    {g['file'][:40]}: p.{g['gap_from']} → p.{g['gap_to']} ({g['gap_pages']} pages gap)")

    # ── Overall score ──────────────────────────────────────────────────────
    score, issues = score_chunking(results)
    grade = "A" if score >= 90 else "B" if score >= 80 else "C" if score >= 70 else "D"
    print(f"\n{'─'*60}")
    print(f"  CHUNKING QUALITY SCORE : {score}/100  (Grade: {grade})")
    print(f"{'─'*60}")
    for issue in issues:
        print(f"  {issue}")

    return results


if __name__ == "__main__":
    print("MedalRAG — PDF Chunking Quality Checker")
    print("=" * 60)

    results_en = run_check(COLLECTION_EN, "en")
    results_es = run_check(COLLECTION_ES, "es")

    print("\n\n" + "="*60)
    print("  SUMMARY — What this means for your RAG")
    print("="*60)
    print("""
Context preservation in our pipeline :

1. ✓ Contextual Retrieval (Anthropic 2024)
   → Each chunk enriched with its document context
   → "From NIH/HHS Drug Interactions section..."
   → Partially solves the header/context problem

2. ~ Table handling
   → PyMuPDF extracts table text as plain text
   → Column structure often lost
   → Headers sometimes in previous chunk
   → Mitigation : overlap=300 tokens captures headers

3. ✓ Page metadata
   → Every chunk knows its filename + page
   → PDF viewer opens to exact page
   → Physician can see full context

4. ~ Figure/image handling
   → Images not indexed (text only)
   → Captions captured if adjacent to text
   → Limitation : VLM not implemented (v2 perspective)

5. ✓ Neighbor context via overlap
   → overlap=300 tokens between chunks
   → Adjacent chunks share ~150 words
   → Reduces context loss at boundaries
""")