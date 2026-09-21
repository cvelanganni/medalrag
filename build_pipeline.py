"""
build_pipeline.py — Full ingestion pipeline for MedalRAG.

1. Load and chunk EN + ES guidelines (PyMuPDF + LangChain)
2. Classify each chunk (heuristic + LLM contextualization)
3. Index into Qdrant with enriched metadata
4. Extract triplets and build the NetworkX knowledge graph
5. Sync to Neo4j for interactive visualization

Usage:
    uv run build_pipeline.py
    uv run build_pipeline.py --skip-qdrant   (graph only)
    uv run build_pipeline.py --skip-graph    (Qdrant only)
    uv run build_pipeline.py --no-contextual (faster, cheaper)
"""

import os
import re
import json
import hashlib
import requests
import pickle
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from core.ingestion.loader import load_and_chunk
from core.graph.builder import (
    extract_triplets_from_chunk,
    build_networkx_graph,
    merge_graphs,
    save_graph,
    load_graph,
)
from core.graph.neo4j_sync import sync_graph_to_neo4j, get_neo4j_stats

load_dotenv()

openai_client = OpenAI()

qdrant = QdrantClient(
    url     = os.getenv("QDRANT_URL", "http://localhost:6333"),
    api_key = os.getenv("QDRANT_API_KEY") or None
)

COLLECTION_EN = os.getenv("QDRANT_COLLECTION_EN", "medical_docs_en")
COLLECTION_ES = os.getenv("QDRANT_COLLECTION_ES", "medical_docs_es")
EMBEDDING_DIM = 1024
OLLAMA_URL    = "http://localhost:11434/api/embed"
BATCH_SIZE    = 5


# ── Embedding ──────────────────────────────────────────────────────────────

def get_embedding(text: str) -> list:
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": "bge-m3", "input": text[:8000]},
            timeout=60
        )
        return resp.json().get("embeddings", [[0.0] * EMBEDDING_DIM])[0]
    except Exception:
        return [0.0] * EMBEDDING_DIM


# ── Heuristic Classification ───────────────────────────────────────────────

# EN + ES recommendation patterns
CLINICAL_PATTERNS = [
    r"\brecommend\w*\b", r"\bshould\b", r"\bpreferred\b",
    r"\bmust\b", r"\brequired\b", r"\bindicated\b",
    r"\brecomend\w*\b", r"\bse recomienda\b", r"\bpreferid\w*\b",
    r"\bdebe\b", r"\bdeberá\b", r"\bindicad\w*\b",
    r"\d+\s*mg\b", r"\bonce daily\b", r"\btwice daily\b",
    r"\bonce a day\b", r"\buna vez al día\b", r"\bdos veces\b",
    r"\bq\d+h\b", r"\bBID\b", r"\bQD\b", r"\bTID\b",
    r"\bgrade [a-c]\b", r"\blevel [i-iii]\b",
    r"\bstrong recommendation\b", r"\bevidence\b",
    r"\bclass [i-iii]\b", r"\bgrade of recommendation\b",
    r"\bcontraindicated\b", r"\bcontraindica\w*\b",
    r"\bmonitor\w*\b", r"\badjust\w*\b",
    r"\binteraction\b", r"\binteracción\b",
    r"\bside effect\b", r"\badverse\b",
    r"\bdose adjustment\b", r"\bajuste de dosis\b",
    r"\bdolutegravir\b", r"\bbictegravir\b",
    r"\btenofovir\b", r"\bemtricitabine\b",
    r"\britonavir\b", r"\bdarunavir\b",
    r"\befavirenz\b", r"\braltegravir\b",
    r"\bDTG\b", r"\bTAF\b", r"\bTDF\b",
    r"\bFTC\b", r"\b3TC\b", r"\bBIC\b",
]

NOISE_PATTERNS = [
    r"\bet al\.\b", r"\bdoi:", r"\bavailable at\b",
    r"ncbi\.nlm\.nih\.gov", r"pubmed\.ncbi",
    r"\bcopyright\b", r"\ball rights reserved\b",
    r"\btable of contents\b", r"\bíndice\b",
    r"\backnowledg\w*\b", r"\bagradecim\w*\b",
    r"\babbreviation\b", r"\babreviatur\w*\b",
    r"https?://[^\s]+",
    r"\[\d+\]",
    r"^\s*\d+\.\s+[A-Z][a-z]+\s+[A-Z]",
]

INTRO_PATTERNS = [
    r"\bintroduction\b", r"\bintroducción\b",
    r"\bbackground\b", r"\bantecedentes\b",
    r"\bsummary\b", r"\bresumen\b",
    r"\bobjective\b", r"\bobjetivo\b",
    r"\bmethods?\b", r"\bmétodos?\b",
    r"\bpurpose of this\b", r"\bthis document\b",
    r"\bthis guideline\b", r"\bthis recommendation\b",
]


def classify_chunk_heuristic(text: str) -> str:
    """
    Fast heuristic classification of a chunk.
    Returns: 'clinical' | 'uncertain' | 'noise' | 'intro'
    """
    text_lower = text.lower()
    word_count = len(text.split())

    if word_count < 25:
        return "noise"

    noise_score = sum(1 for p in NOISE_PATTERNS if re.search(p, text_lower))
    if noise_score >= 3:
        return "noise"

    intro_score    = sum(1 for p in INTRO_PATTERNS if re.search(p, text_lower))
    clinical_score = sum(1 for p in CLINICAL_PATTERNS if re.search(p, text_lower, re.IGNORECASE))

    if clinical_score >= 3:
        return "clinical"
    if intro_score >= 2 and clinical_score < 2:
        return "intro"
    if clinical_score >= 1:
        return "uncertain"
    return "noise"


# ── Contextual Retrieval ───────────────────────────────────────────────────

def contextualize_chunk(
    chunk_text : str,
    filename   : str,
    page       : int,
    lang       : str,
    hint       : str = "uncertain"
) -> tuple:
    """
    Enriches a chunk with LLM-generated context (GPT-4o-mini).
    Returns (enriched_text, confirmed_section_type).
    Cost: ~$0.0001 per chunk.
    """
    lang_hint = "Spanish GESIDA guidelines" if lang == "es" else "English NIH/HHS guidelines"

    try:
        resp = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": f"""You are analyzing a chunk from HIV clinical guidelines ({lang_hint}).

Chunk from: {filename}, page {page}
Pre-classification: {hint.upper()}

Chunk content:
{chunk_text[:1200]}

Tasks:
1. Confirm or correct the section type:
   - CLINICAL: contains specific drug recommendations,
     doses, contraindications, monitoring parameters,
     evidence levels, or treatment protocols
   - INTRO: introduction, objectives, methodology,
     summary, background information
   - REFERENCE: bibliography, citations, references
   - ADMINISTRATIVE: authors, copyright, index, glossary

2. Write exactly 1-2 sentences summarizing what
   CLINICAL information this chunk contains.
   If no clinical content, write "No specific clinical
   recommendation in this passage."

Return in this exact format (no other text):
SECTION_TYPE: [CLINICAL|INTRO|REFERENCE|ADMINISTRATIVE]
CONTEXT: [your 1-2 sentences]"""}],
            temperature=0,
            max_tokens=150
        )

        content      = resp.choices[0].message.content.strip()
        section_type = hint
        context_line = ""

        for line in content.split("\n"):
            line = line.strip()
            if line.startswith("SECTION_TYPE:"):
                raw = line.split(":", 1)[1].strip().upper()
                if raw in ("CLINICAL", "INTRO", "REFERENCE", "ADMINISTRATIVE"):
                    section_type = raw.lower()
            elif line.startswith("CONTEXT:"):
                context_line = line.split(":", 1)[1].strip()

        if context_line and "No specific clinical" not in context_line:
            enriched = f"[{section_type.upper()}] {context_line}\n\n{chunk_text}"
        else:
            enriched = chunk_text

        return enriched, section_type

    except Exception as e:
        print(f"    Contextualization error: {e}")
        return chunk_text, hint


# ── Qdrant ─────────────────────────────────────────────────────────────────

def ensure_collection(name: str):
    if not qdrant.collection_exists(name):
        qdrant.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)
        )
        print(f"  Created collection: {name}")
    else:
        print(f"  Collection exists:  {name}")


def fill_qdrant(
    chunks             : list,
    collection_name    : str,
    use_contextual     : bool = True,
    contextualize_noise: bool = False,
):
    """
    Embeds and inserts chunks into Qdrant.

    Per-chunk pipeline:
      1. Heuristic classification (fast, free)
      2. LLM contextualization for clinical/uncertain chunks
      3. Embed enriched text
      4. Insert with section_type in payload
    """
    print(f"\n  Inserting {len(chunks)} chunks into {collection_name}...")

    existing_ids = set()
    try:
        offset = None
        while True:
            result, offset = qdrant.scroll(
                collection_name=collection_name,
                limit=1000, offset=offset,
                with_payload=False, with_vectors=False
            )
            for p in result:
                existing_ids.add(p.id)
            if offset is None:
                break
    except Exception:
        pass

    batch, inserted, skipped, errors = [], 0, 0, 0
    type_counts = {
        "clinical": 0, "uncertain": 0, "intro": 0,
        "noise": 0, "reference": 0, "administrative": 0,
    }

    for i, chunk in enumerate(chunks):
        chunk_hash = hashlib.md5(
            (chunk["text"] + chunk["filename"] + str(chunk["page"])).encode()
        ).hexdigest()
        chunk_id = int(chunk_hash[:16], 16)

        if chunk_id in existing_ids:
            skipped += 1
            continue

        hint = classify_chunk_heuristic(chunk["text"])

        if use_contextual and hint in ("clinical", "uncertain"):
            enriched_text, section_type = contextualize_chunk(
                chunk_text=chunk["text"], filename=chunk["filename"],
                page=chunk["page"], lang=chunk.get("lang", "en"), hint=hint
            )
        elif use_contextual and contextualize_noise and hint == "intro":
            enriched_text, section_type = contextualize_chunk(
                chunk_text=chunk["text"], filename=chunk["filename"],
                page=chunk["page"], lang=chunk.get("lang", "en"), hint=hint
            )
        else:
            enriched_text = chunk["text"]
            section_type  = hint

        type_counts[section_type] = type_counts.get(section_type, 0) + 1

        embedding = get_embedding(enriched_text)
        if all(v == 0.0 for v in embedding):
            errors += 1
            continue

        batch.append(PointStruct(
            id      = chunk_id,
            vector  = embedding,
            payload = {
                **chunk,
                "text"          : enriched_text,
                "original_text" : chunk["text"],
                "section_type"  : section_type,
                "contextualized": use_contextual and hint in ("clinical", "uncertain"),
            }
        ))
        inserted += 1

        if len(batch) >= 100:
            qdrant.upsert(collection_name=collection_name, wait=True, points=batch)
            print(f"    {i+1}/{len(chunks)} | inserted: {inserted} | "
                  f"skipped: {skipped} | errors: {errors}")
            batch = []

    if batch:
        qdrant.upsert(collection_name=collection_name, wait=True, points=batch)

    total = qdrant.count(collection_name=collection_name).count
    print(f"  Done — {inserted} inserted, {skipped} skipped, {errors} errors")
    print(f"  Total in {collection_name}: {total}")
    print(f"\n  Section type breakdown:")
    for stype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        if count > 0:
            print(f"    {stype:<15}: {count:5d} ({count/max(inserted,1)*100:.1f}%)")


# ── Knowledge Graph ────────────────────────────────────────────────────────

def extract_and_build_graph(
    chunks     : list,
    lang       : str,
    existing_G = None,
    checkpoint : str = "graph_checkpoint.txt"
):
    """
    Extracts triplets from clinical chunks and builds the NetworkX graph.

    NER backends:
      EN: PubMedBERT (d4data/biomedical-ner-all)
      ES: BSC-TeMU RoBERTa (PlanTL-GOB-ES/bsc-bio-ehr-es-pharmaconer)
          → fallback to PubMedBERT if not installed
    """
    if lang == "en":
        from core.ingestion.ner_en import extract_entities_en
        ner_fn = lambda text: [
            e["text"] for e in extract_entities_en(text, backend="pubmedbert")
            if e.get("type") in ("drug", "condition", "lab_value", "organism")
            and e.get("score", 0) >= 0.75
        ]
        print("  NER backend: PubMedBERT (EN)")
    else:
        from core.ingestion.ner_es import extract_entities_es
        ner_fn = lambda text: [
            e["text"] for e in extract_entities_es(text, backend="bsc_temu")
            if e.get("score", 0) >= 0.75
        ]
        print("  NER backend: BSC-TeMU RoBERTa (ES) — fallback to PubMedBERT")

    clinical = [
        c for c in chunks
        if c.get("is_clinical", False)
        or c.get("section_type") in ("clinical", "uncertain")
    ]
    print(f"\n  Extracting triplets from {len(clinical)} clinical chunks [{lang.upper()}]...")

    start_idx = 0
    ckpt_key  = f"{lang}_idx"
    if os.path.exists(checkpoint):
        with open(checkpoint, "r") as f:
            data      = json.load(f)
            start_idx = data.get(ckpt_key, 0)
        print(f"  Resuming from chunk {start_idx}")

    all_triplets, total_extracted, ner_validated = [], 0, 0

    for i in range(start_idx, len(clinical), BATCH_SIZE):
        batch = clinical[i:i + BATCH_SIZE]

        for chunk in batch:
            text     = chunk.get("original_text", chunk["text"])
            triplets = extract_triplets_from_chunk(
                text=text, source=chunk["filename"], lang=lang
            )

            try:
                ner_entities = set(e.lower() for e in ner_fn(text))
            except Exception:
                ner_entities = set()

            validated = []
            for t in triplets:
                subj = t.get("subject", "").strip()
                obj  = t.get("object",  "").strip()
                subj_in_ner = any(subj.lower() in e or e in subj.lower() for e in ner_entities)
                obj_in_ner  = any(obj.lower()  in e or e in obj.lower()  for e in ner_entities)

                if subj_in_ner or obj_in_ner:
                    t["ner_validated"] = True
                    t["confidence"]    = min(0.95, t.get("confidence", 0.85) + 0.05)
                    ner_validated += 1
                else:
                    t["ner_validated"] = False

                t["source"] = chunk["filename"]
                t["lang"]   = lang
                validated.append(t)

            all_triplets.extend(validated)
            total_extracted += len(validated)

        if (i // BATCH_SIZE + 1) % 10 == 0:
            ckpt_data = {}
            if os.path.exists(checkpoint):
                with open(checkpoint, "r") as f:
                    ckpt_data = json.load(f)
            ckpt_data[ckpt_key] = i
            with open(checkpoint, "w") as f:
                json.dump(ckpt_data, f)
            print(f"    {i+1}/{len(clinical)} chunks | "
                  f"{total_extracted} triplets | "
                  f"{ner_validated} NER-validated")

    print(f"  {total_extracted} triplets [{lang.upper()}]")
    print(f"  {ner_validated} NER-validated "
          f"({ner_validated/max(total_extracted,1)*100:.1f}%)")

    G_new = build_networkx_graph(all_triplets)
    if existing_G:
        return merge_graphs(existing_G, G_new)
    return G_new


def reset_checkpoint():
    if os.path.exists("graph_checkpoint.txt"):
        os.remove("graph_checkpoint.txt")


# ── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MedalRAG Build Pipeline")
    parser.add_argument("--skip-qdrant",   action="store_true",
                        help="Skip Qdrant indexing (graph only)")
    parser.add_argument("--skip-graph",    action="store_true",
                        help="Skip graph building (Qdrant only)")
    parser.add_argument("--no-contextual", action="store_true",
                        help="Disable LLM contextualization (faster, cheaper)")
    args = parser.parse_args()

    use_contextual = not args.no_contextual

    print("=" * 60)
    print("  MedalRAG — Build Pipeline")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Contextual Retrieval: {'ON' if use_contextual else 'OFF'}")
    print("=" * 60)

    # Step 1 — Load and chunk PDFs
    print("\n[1/4] Loading and chunking PDFs...")
    chunks_en = load_and_chunk("data/en", lang="en")
    chunks_es = load_and_chunk("data/es", lang="es")

    # Step 2 — Qdrant indexing
    if not args.skip_qdrant:
        print("\n[2/4] Filling Qdrant collections...")
        ensure_collection(COLLECTION_EN)
        ensure_collection(COLLECTION_ES)

        if use_contextual:
            print("\n  Contextual Retrieval active:")
            print("     → heuristic classification (fast, free)")
            print("     → LLM contextualization for clinical/uncertain chunks")
            clinical_en   = sum(1 for c in chunks_en if classify_chunk_heuristic(c["text"]) in ("clinical", "uncertain"))
            clinical_es   = sum(1 for c in chunks_es if classify_chunk_heuristic(c["text"]) in ("clinical", "uncertain"))
            total_calls   = clinical_en + clinical_es
            print(f"\n  Estimated LLM calls: {total_calls}")
            print(f"  Estimated cost     : ~${total_calls * 0.0001:.2f}")

        fill_qdrant(chunks_en, COLLECTION_EN, use_contextual=use_contextual)
        fill_qdrant(chunks_es, COLLECTION_ES, use_contextual=use_contextual)
    else:
        print("\n[2/4] Skipping Qdrant (--skip-qdrant)")

    # Step 3 — Build NetworkX graph
    if not args.skip_graph:
        print("\n[3/4] Building NetworkX knowledge graph...")
        G = load_graph()
        G = extract_and_build_graph(chunks_en, lang="en", existing_G=G)
        G = extract_and_build_graph(chunks_es, lang="es", existing_G=G)
        save_graph(G)
        reset_checkpoint()
        print(f"\n  Final graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    else:
        print("\n[3/4] Skipping graph (--skip-graph)")
        G = load_graph()

    # Step 4 — Sync to Neo4j
    if not args.skip_graph:
        print("\n[4/4] Syncing to Neo4j for visualization...")
        sync_graph_to_neo4j(G)
        stats = get_neo4j_stats()
        print(f"  Neo4j: {stats['nodes']} nodes, {stats['relations']} relations")
    else:
        print("\n[4/4] Skipping Neo4j sync")

    print("\n✓ Build complete!")
    print(f"  EN chunks  : {qdrant.count(COLLECTION_EN).count}")
    print(f"  ES chunks  : {qdrant.count(COLLECTION_ES).count}")
    if not args.skip_graph:
        print(f"  Graph nodes: {G.number_of_nodes()}")
        print(f"  Graph edges: {G.number_of_edges()}")
    if use_contextual:
        print("\n  Contextual Retrieval applied:")
        print("    → clinical/uncertain chunks enriched with LLM context")
        print("    → section_type stored in Qdrant payload")