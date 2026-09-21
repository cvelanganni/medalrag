"""
Graph Search — recherche dans le graphe NetworkX via PPR + PathRAG.
Utilise le graphe en mémoire pour minimiser la latence.
"""

import os
import re
import json
import pickle
import networkx as nx
import numpy as np
import requests
from openai import OpenAI
from entity_normalization import get_canonical_name
from core.graph.pathrag import pathrag_filter, pathrag_stats
from core.graph.hipporag_ppr import personalized_pagerank, get_ppr_triplets
from dotenv import load_dotenv

load_dotenv = __import__('dotenv').load_dotenv
load_dotenv()

openai_client     = OpenAI()
GRAPH_CACHE_PATH  = "graph_cache.pkl"
ENTITY_CACHE_PATH = "entity_embeddings_cache.pkl"
OLLAMA_EMBED_URL  = "http://localhost:11434/api/embed"
MATCH_THRESHOLD   = 0.82

_graph: nx.DiGraph              = None
_entity_names: list             = []
_entity_matrix: np.ndarray|None = None
_node_names_lower: dict         = {}


def load_graph_cache():
    """Charge le graphe NetworkX et le cache d'embeddings en mémoire."""
    global _graph, _entity_names, _entity_matrix, _node_names_lower

    # Graphe NetworkX
    if os.path.exists(GRAPH_CACHE_PATH):
        with open(GRAPH_CACHE_PATH, "rb") as f:
            _graph = pickle.load(f)
        print(f"  Graph loaded : {_graph.number_of_nodes()} nodes")
    else:
        print("  ⚠ graph_cache.pkl not found — run build_pipeline.py first")
        return

    # Cache embeddings entités
    node_names = list(_graph.nodes())
    _node_names_lower = {n.lower(): n for n in node_names}

    if os.path.exists(ENTITY_CACHE_PATH):
        with open(ENTITY_CACHE_PATH, "rb") as f:
            cache = pickle.load(f)
        if set(cache.keys()) == set(node_names):
            _entity_names  = list(cache.keys())
            matrix         = np.array([cache[n] for n in _entity_names])
            norms          = np.linalg.norm(matrix, axis=1, keepdims=True)
            _entity_matrix = matrix / (norms + 1e-8)
            print(f"  Entity embeddings loaded : {len(_entity_names)}")
            return

    # Reconstruit le cache embeddings si nécessaire
    print(f"  Building entity embeddings cache ({len(node_names)} nodes)...")
    cache = {}
    for name in node_names:
        try:
            resp = requests.post(
                OLLAMA_EMBED_URL,
                json={"model": "bge-m3", "input": name[:200]},
                timeout=30
            )
            cache[name] = resp.json().get("embeddings", [[0.0]*1024])[0]
        except Exception:
            cache[name] = [0.0] * 1024

    with open(ENTITY_CACHE_PATH, "wb") as f:
        pickle.dump(cache, f)

    _entity_names  = list(cache.keys())
    matrix         = np.array([cache[n] for n in _entity_names])
    norms          = np.linalg.norm(matrix, axis=1, keepdims=True)
    _entity_matrix = matrix / (norms + 1e-8)
    print(f"  Entity embeddings built and cached")


def resolve_entity(entity: str) -> str | None:
    """Résout une entité extraite vers un noeud du graphe."""
    canonical = get_canonical_name(entity.strip())
    cl        = canonical.lower()

    if cl in _node_names_lower:
        return _node_names_lower[cl]

    for lower, original in _node_names_lower.items():
        if cl in lower or lower in cl:
            return original

    if _entity_matrix is not None:
        try:
            resp = requests.post(
                OLLAMA_EMBED_URL,
                json={"model": "bge-m3", "input": canonical[:200]},
                timeout=30
            )
            emb  = np.array(resp.json().get("embeddings", [[0.0]*1024])[0])
            norm = np.linalg.norm(emb)
            if norm > 0:
                scores   = _entity_matrix @ (emb / norm)
                best_idx = int(np.argmax(scores))
                if float(scores[best_idx]) >= MATCH_THRESHOLD:
                    return _entity_names[best_idx]
        except Exception:
            pass

    return None


def extract_entities(query: str) -> list:
    """Extrait les entités médicales de la query."""
    try:
        resp = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": f"""Extract medical entities
(drug names, diseases, conditions, lab values) EXPLICITLY mentioned.
Normalize to US NIH/HHS English generic names.
Return ONLY valid JSON list. If none: []
Question: {query}"""}],
            temperature=0,
            max_tokens=150
        )
        content = resp.choices[0].message.content.strip()
        match   = re.search(r'\[.*\]', content, re.DOTALL)
        if match:
            return list(set(
                e.strip() for e in json.loads(match.group()) if e.strip()
            ))
        return []
    except Exception:
        return []


def get_direct_triplets(entities: list, limit_per: int = 5) -> list:
    """Récupère les triplets directs depuis le graphe NetworkX."""
    if not _graph or not entities:
        return []

    resolved = list(set(filter(None, [resolve_entity(e) for e in entities])))
    if not resolved:
        return []

    all_triplets = []
    seen = set()

    # Relations entre entités seeds (priorité 1)
    if len(resolved) >= 2:
        for u in resolved:
            for v in resolved:
                if u != v and _graph.has_edge(u, v):
                    data = _graph.get_edge_data(u, v, {})
                    key  = (u, data.get("relation", ""), v)
                    if key not in seen:
                        seen.add(key)
                        all_triplets.append({
                            "subject"   : u,
                            "relation"  : data.get("relation", "RELATED_TO"),
                            "object"    : v,
                            "confidence": data.get("confidence", 0.8),
                            "source"    : data.get("source", "graph"),
                        })

    # Relations de chaque entité (priorité 2)
    for name in resolved:
        count = 0
        for neighbor in _graph.successors(name):
            if count >= limit_per:
                break
            data = _graph.get_edge_data(name, neighbor, {})
            key  = (name, data.get("relation", ""), neighbor)
            if key not in seen:
                seen.add(key)
                all_triplets.append({
                    "subject"   : name,
                    "relation"  : data.get("relation", "RELATED_TO"),
                    "object"    : neighbor,
                    "confidence": data.get("confidence", 0.8),
                    "source"    : data.get("source", "graph"),
                })
                count += 1

    return all_triplets


def graph_search_pipeline(
    query    : str,
    use_ppr  : bool = True,
    use_pathrag: bool = True,
    ppr_top_k: int = 10
) -> dict:
    """
    Pipeline complet de recherche graphe :
    1. Extraction d'entités
    2. Triplets directs Neo4j
    3. PathRAG filtering
    4. PPR multi-hop discovery
    5. Fusion

    Returns:
        {
          "entities"     : list,
          "triplets"     : list,
          "pathrag_stats": dict,
          "ppr_count"    : int
        }
    """
    entities = extract_entities(query)
    resolved = list(set(filter(None, [resolve_entity(e) for e in entities])))

    raw_triplets = get_direct_triplets(entities)

    # PathRAG
    if use_pathrag and raw_triplets:
        filtered_triplets = pathrag_filter(raw_triplets, resolved)
        pr_stats          = pathrag_stats(raw_triplets, filtered_triplets)
    else:
        filtered_triplets = raw_triplets
        pr_stats          = None

    # PPR
    ppr_triplets = []
    if use_ppr and _graph and resolved:
        ppr_triplets = get_ppr_triplets(
            _graph, resolved, top_k=ppr_top_k
        )
        existing = {
            (t["subject"], t["relation"], t["object"])
            for t in filtered_triplets
        }
        ppr_triplets = [
            t for t in ppr_triplets
            if (t["subject"], t["relation"], t["object"]) not in existing
        ]

    all_triplets = filtered_triplets + ppr_triplets

    return {
        "entities"     : entities,
        "resolved"     : resolved,
        "triplets"     : all_triplets,
        "pathrag_stats": pr_stats,
        "ppr_count"    : len(ppr_triplets)
    }


def format_triplets(triplets: list) -> str:
    if not triplets:
        return ""
    return "\n".join(
        f"- {t['subject']} --[{t['relation']}]--> {t['object']}"
        for t in triplets
    )
