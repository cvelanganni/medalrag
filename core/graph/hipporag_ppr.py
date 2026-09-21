"""
hipporag_ppr.py — Personalized PageRank on the NetworkX knowledge graph.

Propagates probability mass from seed entities to discover
distant multi-hop relations not reachable by vector search alone.

Inspired by: HippoRAG 2 - From RAG to Memory (2025)
"""

import networkx as nx


def personalized_pagerank(
    G            : nx.DiGraph,
    seed_entities: list,
    alpha        : float = 0.85,
    top_k        : int   = 15,
) -> list:
    """
    Runs PPR from seed entities and returns top-k discovered nodes.
    Seeds are normalized so each receives equal initial probability mass.
    Seed entities are excluded from the results (only discovered nodes returned).

    Returns:
        list of (entity_name, ppr_score) sorted by descending score
    """
    if not G or not seed_entities:
        return []

    valid_seeds = [e for e in seed_entities if e in G]
    if not valid_seeds:
        return []

    personalization = {e: 1.0 / len(valid_seeds) for e in valid_seeds}

    try:
        ppr_scores = nx.pagerank(
            G,
            alpha           = alpha,
            personalization = personalization,
            weight          = "weight",
            max_iter        = 200,
            tol             = 1e-6,
        )
        results = sorted(
            [(e, s) for e, s in ppr_scores.items() if e not in valid_seeds],
            key=lambda x: x[1],
            reverse=True,
        )
        return results[:top_k]

    except Exception as e:
        print(f"  PPR error: {e}")
        return []


def get_ppr_triplets(
    G            : nx.DiGraph,
    seed_entities: list,
    top_k        : int   = 15,
    min_score    : float = 0.001,
) -> list:
    """
    Returns graph triplets for entities discovered by PPR.
    Includes both outgoing and incoming edges to capture inverse relations.
    Results are sorted by confidence score.

    Returns:
        list of {"subject", "relation", "object", "confidence", "source"}
    """
    top_entities = [
        e for e, score in personalized_pagerank(G, seed_entities, top_k=top_k)
        if score >= min_score
    ]
    if not top_entities:
        return []

    triplets, seen = [], set()

    for entity in top_entities:
        if entity not in G:
            continue
        for neighbor in G.successors(entity):
            data = G.get_edge_data(entity, neighbor, {})
            key  = (entity, data.get("relation", ""), neighbor)
            if key not in seen:
                seen.add(key)
                triplets.append({
                    "subject"   : entity,
                    "relation"  : data.get("relation", "RELATED_TO"),
                    "object"    : neighbor,
                    "confidence": data.get("confidence", 0.8),
                    "source"    : data.get("source", "ppr_discovery"),
                })
        for predecessor in G.predecessors(entity):
            data = G.get_edge_data(predecessor, entity, {})
            key  = (predecessor, data.get("relation", ""), entity)
            if key not in seen:
                seen.add(key)
                triplets.append({
                    "subject"   : predecessor,
                    "relation"  : data.get("relation", "RELATED_TO"),
                    "object"    : entity,
                    "confidence": data.get("confidence", 0.8),
                    "source"    : data.get("source", "ppr_discovery"),
                })

    triplets.sort(key=lambda x: x["confidence"], reverse=True)
    return triplets