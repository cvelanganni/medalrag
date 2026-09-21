"""
evaluate.py — Ablation Study & RAGAs Benchmark for MedalRAG.

Evaluates each pipeline component incrementally:
  v1  Semantic only (baseline)
  v2  +HyDE
  v3  +GraphRAG
  v4  +PathRAG
  v5  +PPR HippoRAG
  v6  +Cross-encoder Re-ranker
  v7  +GESIDA ES guidelines
  v8  Full pipeline (no hybrid)
  v9  +Hybrid BM25+Dense

Usage:
    uv run evaluate.py
    uv run evaluate.py --quick
    uv run evaluate.py --versions v1_semantic_baseline v9_hybrid
    uv run evaluate.py --evaluator gpt4o-mini --dataset golden_dataset_v2.json
"""

# ── Patches — before any imports ───────────────────────────────────────────
import os
import sys
from unittest.mock import MagicMock
from qdrant_client.models import Filter, FieldCondition, MatchValue

# Disable LangSmith to avoid 403 errors
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_PROJECT"]     = ""
os.environ.pop("LANGCHAIN_API_KEY", None)
os.environ.pop("LANGSMITH_API_KEY", None)

# Compatibility patches for ragas/langchain-community
for _mod in [
    "langchain_community.chat_models.vertexai",
    "langchain_community.chat_models.google_palm",
    "langchain_community.llms.vertexai",
    "langchain_community.llms.google_palm",
]:
    sys.modules.setdefault(_mod, MagicMock())

import re
import json
import math
import time
import pickle
import requests
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from neo4j import GraphDatabase
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from ragas import evaluate, EvaluationDataset
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness
from ragas.run_config import RunConfig

from core.graph.builder import normalize_entity as get_canonical_name
from core.graph.pathrag import pathrag_filter
from core.graph.hipporag_ppr import get_ppr_triplets
from core.retrieval.reranker import rerank_chunks
from core.retrieval.hybrid_search import hybrid_search_simple
from core.retrieval.query_expansion import expand_and_search_simple

load_dotenv()


# ── Clients ────────────────────────────────────────────────────────────────

openai_client = OpenAI()

qdrant = QdrantClient(
    url    = os.getenv("QDRANT_URL", "http://localhost:6333"),
    api_key = os.getenv("QDRANT_API_KEY") or None
)

neo4j_driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI", "bolt://localhost:7687"),
    auth=(os.getenv("NEO4J_USERNAME", "neo4j"), os.getenv("NEO4J_PASSWORD", ""))
)

COLLECTION_EN     = os.getenv("QDRANT_COLLECTION_EN", "medical_docs_en")
COLLECTION_ES     = os.getenv("QDRANT_COLLECTION_ES", "medical_docs_es")
NEO4J_DATABASE    = os.getenv("NEO4J_DATABASE", "neo4j")
ENTITY_CACHE_PATH = "entity_embeddings_cache.pkl"
GRAPH_CACHE_PATH  = "graph_cache.pkl"
MATCH_THRESHOLD   = 0.82

OUTPUT_DIR = Path("evaluations")
OUTPUT_DIR.mkdir(exist_ok=True)


# ── RAGAs Evaluator ────────────────────────────────────────────────────────

gpt_evaluator = ChatOpenAI(
    model="gpt-4o-mini", temperature=0,
    api_key=os.getenv("OPENAI_API_KEY"),
)
evaluator_llm = LangchainLLMWrapper(gpt_evaluator)

RAGAS_CONFIG = RunConfig(max_workers=2, timeout=180, max_retries=10, max_wait=60)


# ── Golden Dataset ─────────────────────────────────────────────────────────

GOLDEN_DATASET = []
_ds_path = Path("golden_dataset_v2.json")
if _ds_path.exists():
    with open(_ds_path, "r", encoding="utf-8") as f:
        GOLDEN_DATASET = json.load(f)
    print(f"  Golden dataset: {len(GOLDEN_DATASET)} pairs")


# ── Pipeline Configurations ────────────────────────────────────────────────

PIPELINE_CONFIGS = [
    {
        "version": "v1_semantic_baseline", "label": "v1\nSemantic\nbaseline",
        "short_label": "v1", "description": "Standard semantic search only",
        "use_hyde": False, "use_graph": False, "use_pathrag": False,
        "use_ppr": False, "use_reranker": False, "use_es": False,
        "use_hybrid": False, "use_expansion": False, "color": "#95a5a6",
    },
    {
        "version": "v2_hyde", "label": "v2\n+HyDE", "short_label": "v2",
        "description": "Semantic + HyDE",
        "use_hyde": True, "use_graph": False, "use_pathrag": False,
        "use_ppr": False, "use_reranker": False, "use_es": False,
        "use_hybrid": False, "use_expansion": False, "color": "#3498db",
    },
    {
        "version": "v3_graph", "label": "v3\n+GraphRAG", "short_label": "v3",
        "description": "HyDE + GraphRAG",
        "use_hyde": True, "use_graph": True, "use_pathrag": False,
        "use_ppr": False, "use_reranker": False, "use_es": False,
        "use_hybrid": False, "use_expansion": False, "color": "#2ecc71",
    },
    {
        "version": "v4_pathrag", "label": "v4\n+PathRAG", "short_label": "v4",
        "description": "HyDE + GraphRAG + PathRAG",
        "use_hyde": True, "use_graph": True, "use_pathrag": True,
        "use_ppr": False, "use_reranker": False, "use_es": False,
        "use_hybrid": False, "use_expansion": False, "color": "#9b59b6",
    },
    {
        "version": "v5_ppr", "label": "v5\n+PPR\nHippoRAG", "short_label": "v5",
        "description": "HyDE + GraphRAG + PathRAG + PPR",
        "use_hyde": True, "use_graph": True, "use_pathrag": True,
        "use_ppr": True, "use_reranker": False, "use_es": False,
        "use_hybrid": False, "use_expansion": False, "color": "#e67e22",
    },
    {
        "version": "v6_reranker", "label": "v6\n+Cross-\nEncoder", "short_label": "v6",
        "description": "HyDE + GraphRAG + PathRAG + PPR + Re-ranker",
        "use_hyde": True, "use_graph": True, "use_pathrag": True,
        "use_ppr": True, "use_reranker": True, "use_es": False,
        "use_hybrid": False, "use_expansion": False, "color": "#e74c3c",
    },
    {
        "version": "v7_es", "label": "v7\n+GESIDA\nES", "short_label": "v7",
        "description": "Full pipeline + GESIDA ES",
        "use_hyde": True, "use_graph": True, "use_pathrag": True,
        "use_ppr": True, "use_reranker": True, "use_es": True,
        "use_hybrid": False, "use_expansion": False, "color": "#f39c12",
    },
    {
        "version": "v8_full", "label": "v8\nFull\npipeline", "short_label": "v8",
        "description": "Complete MedalRAG pipeline (no hybrid)",
        "use_hyde": True, "use_graph": True, "use_pathrag": True,
        "use_ppr": True, "use_reranker": True, "use_es": True,
        "use_hybrid": False, "use_expansion": True, "color": "#1abc9c",
    },
    {
        "version": "v9_hybrid", "label": "v9\n+Hybrid\nBM25", "short_label": "v9",
        "description": "Full pipeline + Hybrid BM25+Dense",
        "use_hyde": True, "use_graph": True, "use_pathrag": True,
        "use_ppr": True, "use_reranker": True, "use_es": True,
        "use_hybrid": True, "use_expansion": True, "color": "#2980b9",
    },
]

METRIC_LABELS = {
    "context_recall"     : "Context Recall",
    "faithfulness"       : "Faithfulness",
    "factual_correctness": "Factual Correctness",
    "average"            : "Average",
}


# ── Graph Cache ────────────────────────────────────────────────────────────

def init_graph_cache():
    """Loads the NetworkX graph and entity embedding cache into memory."""
    ppr_graph = None
    if os.path.exists(GRAPH_CACHE_PATH):
        with open(GRAPH_CACHE_PATH, "rb") as f:
            ppr_graph = pickle.load(f)

    try:
        with neo4j_driver.session(database=NEO4J_DATABASE) as session:
            node_names = [r["name"] for r in session.run(
                "MATCH (n:MedicalEntity) RETURN n.name AS name"
            )]
    except Exception:
        node_names = []

    if not node_names and ppr_graph:
        node_names = list(ppr_graph.nodes())

    node_names_lower = {n.lower(): n for n in node_names}

    entity_names, entity_matrix = [], None
    if os.path.exists(ENTITY_CACHE_PATH):
        with open(ENTITY_CACHE_PATH, "rb") as f:
            cache = pickle.load(f)
        entity_names  = list(cache.keys())
        matrix        = np.array([cache[n] for n in entity_names])
        norms         = np.linalg.norm(matrix, axis=1, keepdims=True)
        entity_matrix = matrix / (norms + 1e-8)
        print(f"  Entity cache: {len(entity_names)} entities")
    else:
        print("  No entity cache found — graph features disabled")

    return entity_names, entity_matrix, node_names_lower, ppr_graph


# ── Core Retrieval ─────────────────────────────────────────────────────────

def get_embedding(text: str) -> list:
    try:
        resp = requests.post(
            "http://localhost:11434/api/embed",
            json={"model": "bge-m3", "input": text[:8000]},
            timeout=60,
        )
        return resp.json().get("embeddings", [[0.0] * 1024])[0]
    except Exception:
        return [0.0] * 1024


def translate_to_english(query: str) -> str:
    try:
        return openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content":
                f"Translate to English for HIV guidelines search. Return ONLY translation: {query}"}],
            temperature=0, max_tokens=100,
        ).choices[0].message.content.strip()
    except Exception:
        return query


def generate_hyde_doc(query: str) -> str:
    try:
        return openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content":
                f"""Write 3-5 sentences as if from NIH/HHS HIV clinical guidelines
answering this question. Use precise drug names and doses.
Question: {query}
Guideline excerpt:"""}],
            temperature=0, max_tokens=200,
        ).choices[0].message.content.strip()
    except Exception:
        return query


def search_qdrant(query: str, collection: str,
                  limit: int = 10, use_hyde: bool = False) -> list:
    """Dense search with section_type priority: clinical > uncertain > fallback."""
    query_en = translate_to_english(query)
    prefix   = "Represent this sentence for searching relevant passages: "

    if use_hyde:
        hyde  = generate_hyde_doc(query_en)
        q_emb = get_embedding(prefix + query_en)
        h_emb = get_embedding(prefix + hyde)
        emb   = [(q + h) / 2 for q, h in zip(q_emb, h_emb)]
    else:
        emb = get_embedding(prefix + query_en)

    try:
        if not qdrant.collection_exists(collection):
            return []

        def _search_filtered(section: str, n: int) -> list:
            try:
                return qdrant.query_points(
                    collection_name=collection, query=emb,
                    with_payload=True, limit=n,
                    query_filter=Filter(must=[
                        FieldCondition(key="section_type", match=MatchValue(value=section))
                    ])
                ).points
            except Exception:
                return []

        def _search_all(n: int) -> list:
            try:
                return qdrant.query_points(
                    collection_name=collection, query=emb,
                    with_payload=True, limit=n,
                ).points
            except Exception:
                return []

        results = list(_search_filtered("clinical", limit))
        if len(results) < limit // 2:
            existing  = {p.id for p in results}
            uncertain = [p for p in _search_filtered("uncertain", (limit - len(results)) * 2)
                         if p.id not in existing]
            results  += uncertain[:limit - len(results)]
        if len(results) < limit // 2:
            existing = {p.id for p in results}
            fallback = [p for p in _search_all(limit * 2) if p.id not in existing]
            results += fallback[:limit - len(results)]

        seen, unique = set(), []
        for p in results:
            key = p.payload.get("original_text", p.payload.get("text", ""))[:100]
            if key not in seen:
                seen.add(key)
                unique.append(p)
        return unique[:limit]

    except Exception as e:
        print(f"  Qdrant error ({collection}): {e}")
        return []


def resolve_entity(entity, names, matrix, node_names_lower):
    if not names or matrix is None:
        return None
    canonical = get_canonical_name(entity.strip())
    cl        = canonical.lower()
    if cl in node_names_lower:
        return node_names_lower[cl]
    for lower, original in node_names_lower.items():
        if cl in lower or lower in cl:
            return original
    try:
        emb  = np.array(get_embedding(canonical))
        norm = np.linalg.norm(emb)
        if norm > 0:
            scores   = matrix @ (emb / norm)
            best_idx = int(np.argmax(scores))
            if float(scores[best_idx]) >= MATCH_THRESHOLD:
                return names[best_idx]
    except Exception:
        pass
    return None


def extract_entities(query: str) -> list:
    """Extracts medical entities from a query using keyword matching + GPT-4o-mini fallback."""
    from core.graph.builder import ENTITY_CANONICAL_MAP
    query_lower = query.lower()
    direct_hits = list(set(
        canonical for variant, canonical in ENTITY_CANONICAL_MAP.items()
        if len(variant) >= 3 and variant.lower() in query_lower
    ))
    if len(direct_hits) >= 2:
        return direct_hits

    try:
        resp = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content":
                f"""Extract ALL medical entities from this HIV clinical question.
Include: drug names, conditions, lab values, procedures.
Normalize: DTG→Dolutegravir, TAF→TAF, TDF→TDF, 3TC→Lamivudine.
Return ONLY a JSON array of strings.
Question: {query}
JSON:"""}],
            temperature=0, max_tokens=200,
        )
        content = resp.choices[0].message.content.strip()
        match   = re.search(r'\[.*?\]', content, re.DOTALL)
        if match:
            entities = json.loads(match.group())
            if isinstance(entities, list):
                clean = list(set(e.strip() for e in entities if isinstance(e, str) and e.strip()))
                return clean + [h for h in direct_hits if h not in clean]
    except Exception as e:
        print(f"    Entity extraction error: {e}")

    return list(set(direct_hits)) if direct_hits else []


def get_graph_triplets(entities, names, matrix, node_names_lower, limit_per=5):
    if not entities or not names:
        return []
    resolved = list(set(filter(None, [
        resolve_entity(e, names, matrix, node_names_lower) for e in entities
    ])))
    if not resolved:
        return []

    all_triplets, seen = [], set()

    # Direct connections between seed entities
    if len(resolved) >= 2:
        try:
            with neo4j_driver.session(database=NEO4J_DATABASE) as session:
                for rec in session.run("""
                    MATCH (n:MedicalEntity)-[r]-(m:MedicalEntity)
                    WHERE n.name IN $names AND m.name IN $names AND n.name <> m.name
                    RETURN n.name AS subject, type(r) AS relation,
                           m.name AS object, r.confidence AS confidence
                """, names=resolved):
                    t   = dict(rec)
                    key = (t["subject"], t["relation"], t["object"])
                    if key not in seen:
                        seen.add(key)
                        all_triplets.append(t)
        except Exception:
            pass

    # Outgoing edges per entity
    try:
        with neo4j_driver.session(database=NEO4J_DATABASE) as session:
            for name in resolved:
                for rec in session.run("""
                    MATCH (n:MedicalEntity {name: $name})-[r]->(m:MedicalEntity)
                    RETURN n.name AS subject, type(r) AS relation,
                           m.name AS object, r.confidence AS confidence
                    LIMIT $limit
                """, name=name, limit=limit_per):
                    t   = dict(rec)
                    key = (t["subject"], t["relation"], t["object"])
                    if key not in seen:
                        seen.add(key)
                        all_triplets.append(t)
    except Exception:
        pass

    return all_triplets


def format_triplets(triplets: list) -> str:
    return "\n".join(
        f"- {t['subject']} --[{t['relation']}]--> {t['object']}"
        for t in triplets
    ) if triplets else ""


def generate_response(question: str, context_en: str,
                      context_es: str, graph_context: str) -> str:
    system = """You are an expert HIV/AIDS clinical decision support assistant.
Answer based ONLY on the provided context. Be precise and cite sources.
If US and Spanish guidelines differ, present both.
Add a brief medical disclaimer at the end."""

    prompt = (
        f"US NIH/HHS Guidelines:\n{context_en or 'No EN sources.'}\n\n"
        f"Spanish GESIDA Guidelines:\n{context_es or 'No ES sources.'}\n\n"
        + (f"Knowledge Graph:\n{graph_context}\n\n" if graph_context else "")
        + f"Question: {question}\nAnswer:"
    )

    resp = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt},
        ],
        temperature=0,
    )
    return resp.choices[0].message.content


# ── ML Metrics ─────────────────────────────────────────────────────────────

def safe_float(value) -> float:
    try:
        arr = np.asarray(value, dtype=float).flatten()
        arr = arr[~np.isnan(arr)]
        return float(arr.mean()) if len(arr) > 0 else 0.0
    except Exception:
        try:
            return float(value)
        except Exception:
            return 0.0


def get_fc_score(ragas_result) -> float:
    for key in ["factual_correctness", "factual_correctness(mode=f1)",
                "answer_correctness", "answer_similarity"]:
        try:
            return safe_float(ragas_result[key])
        except (KeyError, Exception):
            continue
    return 0.0


def get_fc_column(df_columns) -> str | None:
    for col in ["factual_correctness", "factual_correctness(mode=f1)",
                "answer_correctness", "answer_similarity"]:
        if col in df_columns:
            return col
    return None


def confidence_interval_95(scores: list) -> dict:
    """Computes 95% CI: mean ± 1.96 * std / sqrt(n)."""
    arr = [s for s in scores if s is not None and not math.isnan(float(s))]
    n   = len(arr)
    if n < 2:
        return {"mean": arr[0] if arr else 0.0,
                "ci_lower": 0.0, "ci_upper": 1.0, "std": 0.0, "n": n}
    mean = sum(arr) / n
    std  = math.sqrt(sum((x - mean) ** 2 for x in arr) / (n - 1))
    ci   = 1.96 * std / math.sqrt(n)
    return {
        "mean"    : round(mean, 4),
        "ci_lower": round(max(0.0, mean - ci), 4),
        "ci_upper": round(min(1.0, mean + ci), 4),
        "std"     : round(std, 4),
        "n"       : n,
    }


def compute_ml_metrics(per_q_log: list) -> dict:
    """Computes CI95% breakdowns by category and language."""
    ml = {}
    for metric in ["context_recall", "faithfulness", "factual_correctness"]:
        scores = [q.get(metric, 0.0) for q in per_q_log if metric in q]
        ml[metric] = {
            "ci_95"       : confidence_interval_95(scores),
            "per_category": {},
            "per_lang"    : {},
        }
        for cat in ["simple", "complex"]:
            cat_scores = [q.get(metric, 0.0) for q in per_q_log
                          if q.get("category") == cat and metric in q]
            if cat_scores:
                ml[metric]["per_category"][cat] = confidence_interval_95(cat_scores)
        for lang in ["en", "es"]:
            lang_scores = [q.get(metric, 0.0) for q in per_q_log
                           if q.get("lang") == lang and metric in q]
            if lang_scores:
                ml[metric]["per_lang"][lang] = confidence_interval_95(lang_scores)
    return ml


def print_ml_metrics(ml: dict, version: str):
    print(f"\n  ML Metrics (CI 95%) — {version}")
    print(f"  {'Metric':<25} {'Mean':>6} {'CI95-':>7} {'CI95+':>7} {'Std':>6} {'N':>4}")
    print(f"  {'─'*25} {'─'*6} {'─'*7} {'─'*7} {'─'*6} {'─'*4}")
    for metric, data in ml.items():
        ci = data["ci_95"]
        print(f"  {metric:<25} {ci['mean']:>6.3f} {ci['ci_lower']:>7.3f} "
              f"{ci['ci_upper']:>7.3f} {ci['std']:>6.3f} {ci['n']:>4}")
    for cat in ["simple", "complex"]:
        ci = ml["context_recall"]["per_category"].get(cat, {})
        if ci:
            print(f"  [{cat:8}] CR={ci['mean']:.3f} [{ci['ci_lower']:.3f},{ci['ci_upper']:.3f}] n={ci['n']}")
    for lang in ["en", "es"]:
        ci = ml["context_recall"]["per_lang"].get(lang, {})
        if ci:
            print(f"  [{lang:8}] CR={ci['mean']:.3f} [{ci['ci_lower']:.3f},{ci['ci_upper']:.3f}] n={ci['n']}")


# ── Statistical Tests ──────────────────────────────────────────────────────

def cohens_d(group1: list, group2: list) -> float:
    if not group1 or not group2:
        return 0.0
    pooled = np.sqrt((np.std(group1, ddof=1)**2 + np.std(group2, ddof=1)**2) / 2)
    return float((np.mean(group1) - np.mean(group2)) / (pooled + 1e-8))


def compute_statistical_tests(all_results: list) -> dict:
    """Paired t-test, Wilcoxon, Cohen's d and bootstrap CI between v1 and v9."""
    from scipy import stats as scipy_stats

    v1 = next((r for r in all_results if r["version"] == "v1_semantic_baseline"), None)
    v9 = next((r for r in all_results if r["version"] == "v9_hybrid"), None)
    if not v1 or not v9:
        if len(all_results) >= 2:
            v1, v9 = all_results[0], all_results[-1]
        else:
            return {}

    results = {}
    for metric in ["context_recall", "faithfulness", "factual_correctness"]:
        s1 = [q.get(metric, 0.0) for q in v1["per_q_log"] if metric in q]
        s9 = [q.get(metric, 0.0) for q in v9["per_q_log"] if metric in q]
        n  = min(len(s1), len(s9))
        if n < 3:
            continue
        s1, s9 = s1[:n], s9[:n]
        delta  = np.mean(s9) - np.mean(s1)
        d      = cohens_d(s9, s1)

        try:
            t_stat, p_ttest = scipy_stats.ttest_rel(s9, s1)
        except Exception:
            t_stat, p_ttest = 0.0, 1.0

        try:
            w_stat, p_wilcoxon = scipy_stats.wilcoxon(s9, s1)
        except Exception:
            w_stat, p_wilcoxon = 0.0, 1.0

        rng    = np.random.default_rng(42)
        deltas = [np.mean([s9[i] for i in rng.integers(0, n, size=n)]) -
                  np.mean([s1[i] for i in rng.integers(0, n, size=n)])
                  for _ in range(1000)]

        abs_d  = abs(d)
        effect = ("negligible" if abs_d < 0.2 else "small" if abs_d < 0.5
                  else "medium" if abs_d < 0.8 else "large" if abs_d < 1.2
                  else "very large")

        results[metric] = {
            "v1_mean": round(float(np.mean(s1)), 4),
            "v9_mean": round(float(np.mean(s9)), 4),
            "delta"  : round(float(delta), 4),
            "delta_ci_lower": round(float(np.percentile(deltas, 2.5)), 4),
            "delta_ci_upper": round(float(np.percentile(deltas, 97.5)), 4),
            "cohens_d"      : round(d, 4),
            "effect_size"   : effect,
            "ttest"     : {"t_statistic": round(float(t_stat), 4),
                           "p_value": round(float(p_ttest), 4),
                           "significant": p_ttest < 0.05},
            "wilcoxon"  : {"w_statistic": round(float(w_stat), 4),
                           "p_value": round(float(p_wilcoxon), 4),
                           "significant": p_wilcoxon < 0.05},
            "n": n,
        }
    return results


def print_statistical_tests(tests: dict, v1_name: str, v9_name: str):
    if not tests:
        return
    print(f"\n  Statistical Tests — {v1_name} vs {v9_name}")
    print(f"  {'─'*65}")
    print(f"  {'Metric':<25} {'Delta':>7} {'p(t)':>7} {'p(W)':>7} {'d':>6} {'Effect':>12} {'Sig':>5}")
    print(f"  {'─'*65}")
    for metric, r in tests.items():
        sig = "✓" if r["ttest"]["significant"] else "✗"
        print(f"  {metric:<25} {r['delta']:>+7.3f} {r['ttest']['p_value']:>7.4f} "
              f"{r['wilcoxon']['p_value']:>7.4f} {r['cohens_d']:>6.3f} "
              f"{r['effect_size']:>12} {sig:>5}")
    print(f"\n  Bootstrap 95% CI on delta:")
    for metric, r in tests.items():
        print(f"    {metric:<25} Δ = {r['delta']:+.3f} "
              f"[{r['delta_ci_lower']:+.3f}, {r['delta_ci_upper']:+.3f}]")
    print("  p(t) = paired t-test | p(W) = Wilcoxon | d = Cohen's d")


# ── Single Pipeline Run ────────────────────────────────────────────────────

def run_pipeline(question, config, entity_names, entity_matrix,
                 node_names_lower, ppr_graph) -> dict:
    use_hyde      = config["use_hyde"]
    use_graph     = config["use_graph"]
    use_pathrag   = config["use_pathrag"]
    use_ppr       = config["use_ppr"]
    use_reranker  = config["use_reranker"]
    use_es        = config["use_es"]
    use_hybrid    = config.get("use_hybrid",    False)
    use_expansion = config.get("use_expansion", False)

    hyde_doc = generate_hyde_doc(translate_to_english(question)) if use_hyde else None

    # Retrieve EN chunks
    if use_expansion:
        chunks_en = expand_and_search_simple(
            query=question, collection=COLLECTION_EN,
            limit=15, use_hyde=use_hyde, hyde_doc=hyde_doc, lang="en",
        )
    elif use_hybrid:
        chunks_en = hybrid_search_simple(
            query=question, collection=COLLECTION_EN,
            limit=15, use_hyde=use_hyde, hyde_doc=hyde_doc,
        )
    else:
        chunks_en = search_qdrant(question, COLLECTION_EN, limit=15, use_hyde=use_hyde)

    if use_reranker and chunks_en:
        chunks_en = rerank_chunks(query=question, chunks=chunks_en, top_k=8)

    # Retrieve ES chunks
    chunks_es = []
    if use_es:
        if use_expansion:
            chunks_es = expand_and_search_simple(
                query=question, collection=COLLECTION_ES,
                limit=8, use_hyde=use_hyde, hyde_doc=hyde_doc, lang="es",
            )
        elif use_hybrid:
            chunks_es = hybrid_search_simple(
                query=question, collection=COLLECTION_ES,
                limit=8, use_hyde=use_hyde, hyde_doc=hyde_doc,
            )
        else:
            chunks_es = search_qdrant(question, COLLECTION_ES, limit=8, use_hyde=use_hyde)
        if use_reranker and chunks_es:
            chunks_es = rerank_chunks(query=question, chunks=chunks_es, top_k=5)

    # GraphRAG
    entities, triplets, ppr_count = [], [], 0
    if use_graph and entity_names:
        entities     = extract_entities(question)
        raw_triplets = get_graph_triplets(entities, entity_names, entity_matrix, node_names_lower)

        if use_pathrag and raw_triplets:
            resolved = list(set(filter(None, [
                resolve_entity(e, entity_names, entity_matrix, node_names_lower)
                for e in entities
            ])))
            triplets = pathrag_filter(raw_triplets, resolved)
        else:
            triplets = raw_triplets

        if use_ppr and ppr_graph and entities:
            resolved_ppr = list(set(filter(None, [
                resolve_entity(e, entity_names, entity_matrix, node_names_lower)
                for e in entities
            ])))
            ppr_trips = get_ppr_triplets(ppr_graph, resolved_ppr, top_k=10)
            existing  = {(t["subject"], t["relation"], t["object"]) for t in triplets}
            new_ppr   = [t for t in ppr_trips
                         if (t["subject"], t["relation"], t["object"]) not in existing]
            triplets  += new_ppr
            ppr_count  = len(new_ppr)

    graph_text = format_triplets(triplets)

    retrieved_contexts = (
        [p.payload.get("original_text", p.payload.get("text", "")) for p in chunks_en] +
        [p.payload.get("original_text", p.payload.get("text", "")) for p in chunks_es] +
        ([f"Knowledge Graph relations:\n{graph_text}"] if graph_text else [])
    )

    return {
        "retrieved_contexts": retrieved_contexts,
        "context_en"        : "\n\n".join(
            p.payload.get("original_text", p.payload.get("text", "")) for p in chunks_en),
        "context_es"        : "\n\n".join(
            p.payload.get("original_text", p.payload.get("text", "")) for p in chunks_es),
        "graph_context"     : graph_text,
        "entities"          : entities,
        "triplets_count"    : len(triplets),
        "ppr_count"         : ppr_count,
        "chunks_en"         : len(chunks_en),
        "chunks_es"         : len(chunks_es),
    }


# ── Evaluation Loop ────────────────────────────────────────────────────────

def run_evaluation(config, entity_names, entity_matrix,
                   node_names_lower, ppr_graph) -> dict:
    print(f"\n{'='*60}")
    print(f"  {config['version']} — {config['description']}")
    print(f"  HyDE={config['use_hyde']} | Graph={config['use_graph']} | "
          f"PathRAG={config['use_pathrag']} | PPR={config['use_ppr']} | "
          f"Reranker={config['use_reranker']} | ES={config['use_es']} | "
          f"Hybrid={config.get('use_hybrid', False)}")
    print(f"{'='*60}")

    dataset, categories, langs, per_q_log = [], [], [], []

    for i, item in enumerate(GOLDEN_DATASET):
        print(f"  [{i+1}/{len(GOLDEN_DATASET)}] ({item['category']}/{item['lang']}) "
              f"{item['question'][:55]}...")
        try:
            result = run_pipeline(
                question=item["question"], config=config,
                entity_names=entity_names, entity_matrix=entity_matrix,
                node_names_lower=node_names_lower, ppr_graph=ppr_graph,
            )
            answer = generate_response(
                question=item["question"],
                context_en=result["context_en"],
                context_es=result["context_es"],
                graph_context=result["graph_context"],
            )
            dataset.append({
                "user_input"        : item["question"],
                "retrieved_contexts": result["retrieved_contexts"],
                "response"          : answer,
                "reference"         : item["reference"],
            })
            per_q_log.append({
                "category": item["category"], "lang": item["lang"],
                "question": item["question"], "entities": result["entities"],
                "triplets_count": result["triplets_count"],
                "ppr_count": result["ppr_count"],
                "chunks_en": result["chunks_en"], "chunks_es": result["chunks_es"],
            })
            print(f"    EN:{result['chunks_en']} ES:{result['chunks_es']} "
                  f"triplets:{result['triplets_count']} entities:{result['entities']}")
        except Exception as e:
            print(f"    Error: {e}")
            dataset.append({
                "user_input": item["question"],
                "retrieved_contexts": ["Error retrieving context"],
                "response": "Error generating response",
                "reference": item["reference"],
            })
            per_q_log.append({"category": item["category"], "lang": item["lang"],
                               "question": item["question"], "error": str(e)})
        categories.append(item["category"])
        langs.append(item["lang"])

    # RAGAs evaluation
    print("\n  Running RAGAs evaluation...")
    eval_dataset = EvaluationDataset.from_list(dataset)
    ragas_result = evaluate(
        dataset=eval_dataset,
        metrics=[LLMContextRecall(), Faithfulness(), FactualCorrectness()],
        llm=evaluator_llm, run_config=RAGAS_CONFIG,
    )

    overall = {
        "context_recall"     : safe_float(ragas_result["context_recall"]),
        "faithfulness"       : safe_float(ragas_result["faithfulness"]),
        "factual_correctness": get_fc_score(ragas_result),
    }
    overall["average"] = sum(overall.values()) / 3

    # Breakdown by category and language
    breakdown = {}
    try:
        df     = ragas_result.to_pandas()
        df["category"] = categories
        df["lang"]     = langs
        fc_col = get_fc_column(df.columns)
        for cat in ["simple", "complex"]:
            sub = df[df["category"] == cat]
            if len(sub) > 0:
                scores = {
                    "context_recall"     : safe_float(sub["context_recall"].values) if "context_recall" in sub.columns else 0.0,
                    "faithfulness"       : safe_float(sub["faithfulness"].values)    if "faithfulness"   in sub.columns else 0.0,
                    "factual_correctness": safe_float(sub[fc_col].values)            if fc_col           else 0.0,
                }
                scores["average"]  = sum(scores.values()) / 3
                breakdown[cat]     = scores
        for lang in ["en", "es"]:
            sub = df[df["lang"] == lang]
            if len(sub) > 0:
                scores = {
                    "context_recall"     : safe_float(sub["context_recall"].values) if "context_recall" in sub.columns else 0.0,
                    "faithfulness"       : safe_float(sub["faithfulness"].values)    if "faithfulness"   in sub.columns else 0.0,
                    "factual_correctness": safe_float(sub[fc_col].values)            if fc_col           else 0.0,
                }
                scores["average"]        = sum(scores.values()) / 3
                breakdown[f"lang_{lang}"] = scores
    except Exception as e:
        print(f"  Breakdown error: {e}")

    # Per-question scores for ML metrics
    try:
        df_result = ragas_result.to_pandas()
        fc_col    = get_fc_column(df_result.columns)
        for i, q_log in enumerate(per_q_log):
            if i < len(df_result):
                row = df_result.iloc[i]
                q_log["context_recall"]      = safe_float(row.get("context_recall", 0.0))
                q_log["faithfulness"]         = safe_float(row.get("faithfulness",   0.0))
                q_log["factual_correctness"]  = safe_float(row.get(fc_col, 0.0)) if fc_col else 0.0
    except Exception as e:
        print(f"  Per-question enrichment error: {e}")

    ml_metrics = {}
    try:
        ml_metrics = compute_ml_metrics(per_q_log)
        print_ml_metrics(ml_metrics, config["version"])
    except Exception as e:
        print(f"  ML metrics error: {e}")

    print(f"\n  Results:")
    print(f"    Context Recall     : {overall['context_recall']:.3f}")
    print(f"    Faithfulness       : {overall['faithfulness']:.3f}")
    print(f"    Factual Correctness: {overall['factual_correctness']:.3f}")
    print(f"    Average            : {overall['average']:.3f}")

    return {
        "version"   : config["version"],
        "config"    : config,
        "overall"   : overall,
        "breakdown" : breakdown,
        "ml_metrics": ml_metrics,
        "per_q_log" : per_q_log,
    }


# ── Visualization ──────────────────────────────────────────────────────────

def plot_ablation_bars(results, timestamp):
    """Generates the ablation bar chart (4 metrics × N versions)."""
    metrics = ["context_recall", "faithfulness", "factual_correctness", "average"]
    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    fig.suptitle("MedalRAG — Ablation Study | RAGAs Metrics",
                 fontsize=14, fontweight="bold")

    for ax, metric in zip(axes, metrics):
        versions = [r["config"]["short_label"] for r in results]
        values   = [r["overall"].get(metric, 0) for r in results]
        colors   = [r["config"]["color"] for r in results]
        bars     = ax.bar(range(len(results)), values, color=colors,
                          alpha=0.85, edgecolor="white", linewidth=1.5)

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.008, f"{val:.3f}",
                    ha="center", va="bottom", fontsize=8, fontweight="bold")

        best_idx = int(np.argmax(values))
        bars[best_idx].set_edgecolor("black")
        bars[best_idx].set_linewidth(2.5)

        ax.set_title(METRIC_LABELS[metric], fontsize=11, fontweight="bold")
        ax.set_xticks(range(len(results)))
        ax.set_xticklabels(versions, fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score", fontsize=9)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path = OUTPUT_DIR / f"ablation_bars_{timestamp}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    return path


# ── Summary & Serialization ────────────────────────────────────────────────

def print_summary_table(results):
    print("\n" + "=" * 70)
    print("  ABLATION STUDY — RESULTS SUMMARY")
    print("=" * 70)
    print(f"  {'Version':<35} {'CR':>7} {'Faith':>7} {'FC':>7} {'Avg':>7}")
    print("-" * 70)
    for r in results:
        o = r["overall"]
        h = "H" if r["config"].get("use_hybrid") else " "
        print(f"  {h} {r['version']:<33} "
              f"{o.get('context_recall',0):>7.3f} "
              f"{o.get('faithfulness',0):>7.3f} "
              f"{o.get('factual_correctness',0):>7.3f} "
              f"{o.get('average',0):>7.3f}")
    if len(results) >= 2:
        print("-" * 70)
        v1 = results[0]["overall"]
        vn = results[-1]["overall"]
        print(f"  {'Delta v1 → ' + results[-1]['version']:<35} "
              f"{vn.get('context_recall',0)-v1.get('context_recall',0):>+7.3f} "
              f"{vn.get('faithfulness',0)-v1.get('faithfulness',0):>+7.3f} "
              f"{vn.get('factual_correctness',0)-v1.get('factual_correctness',0):>+7.3f} "
              f"{vn.get('average',0)-v1.get('average',0):>+7.3f}")
    print("=" * 70)


def make_serializable(obj):
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_serializable(i) for i in obj]
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (np.floating, np.ndarray)):
        return obj.tolist() if isinstance(obj, np.ndarray) else float(obj)
    if isinstance(obj, (bool, int, float, str)) or obj is None:
        return obj
    try:
        return float(obj)
    except Exception:
        return str(obj)


# ── Main Benchmark ─────────────────────────────────────────────────────────

def run_benchmark(versions_to_run=None, sleep_between=20,
                  dataset_path=None, evaluator="deepseek"):
    global GOLDEN_DATASET, evaluator_llm

    if dataset_path:
        ds = Path(dataset_path)
        if ds.exists():
            with open(ds, "r", encoding="utf-8") as f:
                GOLDEN_DATASET = json.load(f)
            print(f"  Dataset: {dataset_path} ({len(GOLDEN_DATASET)} pairs)")

    _EV = {
        "deepseek"     : ("deepseek-chat",            os.getenv("DEEPSEEK_API_KEY"),  "https://api.deepseek.com/v1", "DeepSeek",      "openai"),
        "gpt4o-mini"   : ("gpt-4o-mini",              os.getenv("OPENAI_API_KEY"),    None,                          "GPT-4o-mini",   "openai"),
        "gpt4o"        : ("gpt-4o",                   os.getenv("OPENAI_API_KEY"),    None,                          "GPT-4o",        "openai"),
        "claude-haiku" : ("claude-haiku-4-5-20251001", os.getenv("ANTHROPIC_API_KEY"), None,                         "Claude Haiku",  "anthropic"),
        "claude-sonnet": ("claude-sonnet-4-5",         os.getenv("ANTHROPIC_API_KEY"), None,                         "Claude Sonnet", "anthropic"),
        "gemma3"       : ("gemma3:4b",                 None, "http://localhost:11434/v1", "Gemma3 local",    "openai"),
        "gemma3-1b"    : ("gemma3:1b",                 None, "http://localhost:11434/v1", "Gemma3-1b local", "openai"),
    }
    _model, _key, _url, _label, _provider = _EV.get(evaluator, _EV["deepseek"])

    if _provider == "anthropic":
        evaluator_llm = LangchainLLMWrapper(
            ChatAnthropic(model=_model, temperature=0, api_key=_key)
        )
    else:
        _kw = {"base_url": _url} if _url else {}
        evaluator_llm = LangchainLLMWrapper(
            ChatOpenAI(model=_model, temperature=0, api_key=_key, **_kw)
        )
    print(f"  Evaluator: {_label}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\n{'#'*60}")
    print("  MedalRAG — Ablation Study & RAGAs Benchmark")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*60}")
    print(f"  Questions : {len(GOLDEN_DATASET)}")
    print(f"  Evaluator : {_label} | Generator: GPT-4o")

    entity_names, entity_matrix, node_names_lower, ppr_graph = init_graph_cache()
    print(f"  Entities  : {len(entity_names)}")
    print(f"  PPR graph : {ppr_graph.number_of_nodes() if ppr_graph else 'N/A'} nodes")

    configs = PIPELINE_CONFIGS
    if versions_to_run:
        configs = [c for c in PIPELINE_CONFIGS if c["version"] in versions_to_run]
    print(f"  Versions  : {len(configs)}")

    all_results = []
    for i, config in enumerate(configs):
        if i > 0:
            print(f"\nWaiting {sleep_between}s...")
            time.sleep(sleep_between)
        result = run_evaluation(
            config=config, entity_names=entity_names,
            entity_matrix=entity_matrix, node_names_lower=node_names_lower,
            ppr_graph=ppr_graph,
        )
        all_results.append(result)

    print_summary_table(all_results)

    stat_tests = {}
    if len(all_results) >= 2:
        stat_tests = compute_statistical_tests(all_results)
        print_statistical_tests(stat_tests, all_results[0]["version"], all_results[-1]["version"])

    print("\nGenerating visualizations...")
    bar_path = plot_ablation_bars(all_results, timestamp)

    json_path = OUTPUT_DIR / f"benchmark_{timestamp}.json"
    payload   = {
        "date"             : datetime.now().isoformat(),
        "n_questions"      : len(GOLDEN_DATASET),
        "n_versions"       : len(all_results),
        "dataset"          : str(dataset_path) if dataset_path else "golden_dataset_v2.json",
        "evaluator"        : _label,
        "statistical_tests": stat_tests,
        "results"          : [
            {
                "version"    : r["version"],
                "description": r["config"]["description"],
                "config"     : {k: v for k, v in r["config"].items()
                                if k not in ("color", "label", "short_label")},
                "overall"    : r["overall"],
                "breakdown"  : r["breakdown"],
                "ml_metrics" : r.get("ml_metrics", {}),
                "per_q_log"  : r["per_q_log"],
            }
            for r in all_results
        ],
        "visualizations": {"bars": str(bar_path)},
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(make_serializable(payload), f, indent=2, ensure_ascii=False)

    print(f"\n  Benchmark complete!")
    print(f"  JSON : {json_path}")
    print(f"  Plot : {bar_path}")


# ── CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MedalRAG Ablation Study & RAGAs Benchmark")
    parser.add_argument("--versions",  nargs="+", default=None,
                        help="Specific versions to run")
    parser.add_argument("--sleep",     type=int,  default=20,
                        help="Seconds between runs (default: 20)")
    parser.add_argument("--dataset",   type=str,  default="golden_dataset_v2.json",
                        help="Path to golden dataset JSON")
    parser.add_argument("--evaluator", type=str,  default="deepseek",
                        choices=["deepseek","gpt4o-mini","gpt4o",
                                 "claude-haiku","claude-sonnet","gemma3","gemma3-1b"],
                        help="LLM evaluator for RAGAs")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: v1, v3, v8, v9")
    args = parser.parse_args()

    run_benchmark(
        versions_to_run = ["v1_semantic_baseline","v3_graph","v8_full","v9_hybrid"]
                         if args.quick else args.versions,
        sleep_between   = args.sleep,
        dataset_path    = args.dataset,
        evaluator       = args.evaluator,
    )