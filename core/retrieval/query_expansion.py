"""
query_expansion.py — Query Expansion with RRF fusion.

Strategy:
  1. Generate N reformulations of the original question via GPT-4o-mini
  2. Search Qdrant for each reformulation
  3. Fuse results with weighted Reciprocal Rank Fusion (RRF)

The original query gets a 1.5x weight boost over reformulations.
"""

import os
import re
import json
import requests
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from dotenv import load_dotenv

load_dotenv()

openai_client = OpenAI()

qdrant = QdrantClient(
    url    = os.getenv("QDRANT_URL", "http://localhost:6333"),
    api_key = os.getenv("QDRANT_API_KEY") or None
)

OLLAMA_URL    = "http://localhost:11434/api/embed"
EMBEDDING_DIM = 1024


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


# ── Query Expansion ────────────────────────────────────────────────────────

def expand_query(query: str, n: int = 3, lang: str = "auto") -> list:
    """
    Generates N reformulations of the original clinical query using GPT-4o-mini.
    Each reformulation targets a different retrieval angle:
      1. Formal medical terminology (drug abbreviations, class names)
      2. Pharmacological mechanism (enzyme induction, CYP interactions)
      3. Clinical guideline phrasing (recommended regimen, dose adjustment)
    """
    if lang == "es":
        lang_instruction = "Generate all reformulations in SPANISH."
    elif lang == "en":
        lang_instruction = "Generate all reformulations in ENGLISH."
    else:
        lang_instruction = "Generate reformulations in the SAME language as the input question."

    try:
        resp = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": f"""You are an expert in HIV/AIDS clinical guidelines.

Generate exactly {n} different reformulations of this clinical question
to improve retrieval from HIV medical guidelines (NIH/HHS and GESIDA).

Each reformulation should use a different angle:
  1. Formal medical terminology and drug abbreviations (DTG, TAF, FTC, INSTI, NNRTI, PI)
  2. Mechanism or pharmacology (CYP3A4, enzyme induction, drug interaction)
  3. Clinical guideline-style phrasing (recommended regimen, preferred treatment, dose adjustment)

{lang_instruction}
Do NOT repeat the original question. Do NOT number the reformulations.

Return ONLY a valid JSON list of {n} strings:
["reformulation 1", "reformulation 2", "reformulation 3"]

Original question: {query}"""}],
            temperature=0.4,
            max_tokens=300,
        )
        content = resp.choices[0].message.content.strip()
        match   = re.search(r'\[.*\]', content, re.DOTALL)
        if not match:
            return []
        variants = json.loads(match.group())
        seen, unique = {query.lower().strip()}, []
        for v in variants:
            v_clean = v.strip()
            if v_clean.lower() not in seen and len(v_clean) > 10:
                seen.add(v_clean.lower())
                unique.append(v_clean)
        return unique[:n]
    except Exception as e:
        print(f"  Query expansion error: {e}")
        return []


# ── RRF Fusion ─────────────────────────────────────────────────────────────

def reciprocal_rank_fusion(
    all_results  : list,
    k            : int  = 60,
    query_weights: list = None,
) -> list:
    """
    Fuses multiple Qdrant result lists using weighted RRF.
    RRF score = Σ weight_i / (k + rank_i)
    Default weights: original query = 1.5x, variants = 1.0x.
    """
    if not all_results:
        return []

    if query_weights is None:
        n = len(all_results)
        weights = [1.5] + [1.0] * (n - 1)
        total   = sum(weights)
        query_weights = [w / total for w in weights]

    scores, points = {}, {}
    for result_list, weight in zip(all_results, query_weights):
        for rank, point in enumerate(result_list, start=1):
            pid         = str(point.id)
            scores[pid] = scores.get(pid, 0) + weight / (k + rank)
            if pid not in points:
                points[pid] = point

    return [points[pid] for pid in sorted(scores, key=lambda p: scores[p], reverse=True)]


# ── Single Query Search ────────────────────────────────────────────────────

def _search_single(
    query      : str,
    collection : str,
    limit      : int,
    use_hyde   : bool = False,
    hyde_doc   : str  = None,
) -> list:
    """
    Searches Qdrant for a single query with section_type filtering.
    Prioritizes 'clinical' chunks, falls back to 'uncertain' then unfiltered.
    """
    prefix = "Represent this sentence for searching relevant passages: "
    if use_hyde and hyde_doc:
        q_emb = get_embedding(prefix + query)
        h_emb = get_embedding(prefix + hyde_doc)
        emb   = [(q + h) / 2 for q, h in zip(q_emb, h_emb)]
    else:
        emb = get_embedding(prefix + query)

    try:
        if not qdrant.collection_exists(collection):
            return []

        def _search_filtered(section: str, n: int) -> list:
            return qdrant.query_points(
                collection_name=collection, query=emb,
                with_payload=True, limit=n,
                query_filter=Filter(must=[
                    FieldCondition(key="section_type", match=MatchValue(value=section))
                ])
            ).points

        def _search_all(n: int) -> list:
            return qdrant.query_points(
                collection_name=collection, query=emb,
                with_payload=True, limit=n,
            ).points

        results = list(_search_filtered("clinical", limit))

        # Fill up with 'uncertain' chunks if needed
        if len(results) < limit // 2:
            existing  = {p.id for p in results}
            uncertain = [p for p in _search_filtered("uncertain", (limit - len(results)) * 2)
                         if p.id not in existing]
            results  += uncertain[:limit - len(results)]

        # Final fallback: unfiltered search
        if len(results) < limit // 2:
            existing = {p.id for p in results}
            fallback = [p for p in _search_all(limit * 2) if p.id not in existing]
            results += fallback[:limit - len(results)]

        return results[:limit]

    except Exception:
        return []


# ── Main Entry Points ──────────────────────────────────────────────────────

def expand_and_search(
    query      : str,
    collection : str,
    limit      : int  = 10,
    n_variants : int  = 3,
    use_hyde   : bool = False,
    hyde_doc   : str  = None,
    lang       : str  = "auto",
    verbose    : bool = False,
) -> dict:
    """
    Full query expansion + RRF fusion pipeline.
    Returns {"results": list, "variants": list, "n_queries": int}.
    """
    variants = expand_query(query, n=n_variants, lang=lang)

    if verbose:
        print(f"  Query expansion ({len(variants)} variants):")
        for i, v in enumerate(variants, 1):
            print(f"    {i}. {v[:70]}")

    # Search original query (higher limit since it has 1.5x weight)
    all_results = [_search_single(query, collection, limit + 5, use_hyde, hyde_doc)]

    # Search each variant (no HyDE — only applied to original)
    for variant in variants:
        all_results.append(_search_single(variant, collection, limit))

    fused = reciprocal_rank_fusion(all_results)

    # Deduplicate on text content
    seen, unique = set(), []
    for p in fused:
        key = p.payload.get("original_text", p.payload.get("text", ""))[:100]
        if key not in seen:
            seen.add(key)
            unique.append(p)

    return {
        "results"  : unique[:limit],
        "variants" : variants,
        "n_queries": 1 + len(variants),
    }


def expand_and_search_simple(
    query      : str,
    collection : str,
    limit      : int  = 10,
    use_hyde   : bool = False,
    hyde_doc   : str  = None,
    lang       : str  = "auto",
) -> list:
    """Simplified wrapper — returns results list directly."""
    return expand_and_search(
        query=query, collection=collection,
        limit=limit, use_hyde=use_hyde,
        hyde_doc=hyde_doc, lang=lang,
    )["results"]