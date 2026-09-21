"""
pathrag.py — Relational path pruning for GraphRAG context.

Filters raw Neo4j triplets to keep only high-confidence paths
between seed entities before sending context to the LLM.

Inspired by: PathRAG - Pruning Graph-based RAG with Relational Paths (2025)
"""

import networkx as nx

# Safety-critical relations — always preserved regardless of confidence
SAFETY_RELATIONS = {
    "INTERACTS_WITH",
    "CONTRAINDICATED_IN",
    "REQUIRES_ADJUSTMENT",
    "AVOIDED_IN",
}


def pathrag_filter(
    triplets             : list,
    entities             : list,
    G                    : nx.DiGraph = None,
    confidence_threshold : float = 0.85,
    max_path_length      : int   = 3,
) -> list:
    """
    Filters triplets by keeping only those on short paths between seed entities.

    Strategy:
      1. Build a local subgraph from the retrieved triplets
      2. Find shortest paths between all pairs of seed entities
      3. Keep triplets on those paths if confidence >= threshold
      4. Always preserve safety-critical relations (INTERACTS_WITH, etc.)
      5. Always preserve very high confidence triplets (>= 0.92)

    Args:
        triplets             : raw Neo4j triplets
        entities             : seed entities extracted from the query
        G                    : full NetworkX graph (optional, unused here)
        confidence_threshold : minimum confidence to keep a triplet
        max_path_length      : maximum path length between seed entities

    Returns:
        deduplicated list of filtered triplets
    """
    if not triplets:
        return []

    # Build local subgraph from retrieved triplets
    local_G = nx.DiGraph()
    for t in triplets:
        subj = t["subject"]
        obj  = t["object"]
        conf = t.get("confidence", 0.8)
        local_G.add_node(subj)
        local_G.add_node(obj)
        local_G.add_edge(subj, obj, relation=t["relation"], confidence=conf, weight=conf)

    relevant_edges = set()

    # Find paths between all pairs of seed entities
    if len(entities) >= 2:
        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                src, dst = entities[i], entities[j]
                if src not in local_G or dst not in local_G:
                    continue
                try:
                    for path in nx.all_simple_paths(local_G, src, dst, cutoff=max_path_length):
                        for k in range(len(path) - 1):
                            edge = (path[k], path[k + 1])
                            if local_G.get_edge_data(*edge, {}).get("confidence", 0.8) >= confidence_threshold:
                                relevant_edges.add(edge)
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue

    # Direct connections between seed entities
    for t in triplets:
        if t["subject"] in entities and t["object"] in entities:
            relevant_edges.add((t["subject"], t["object"]))

    # Very high confidence triplets — always keep
    for t in triplets:
        if t.get("confidence", 0.8) >= 0.92:
            relevant_edges.add((t["subject"], t["object"]))

    # Filter and deduplicate
    seen, unique = set(), []
    for t in triplets:
        edge = (t["subject"], t["object"])
        keep = (
            edge in relevant_edges or
            (t["relation"] in SAFETY_RELATIONS and t.get("confidence", 0.8) >= confidence_threshold)
        )
        if keep:
            key = (t["subject"], t["relation"], t["object"])
            if key not in seen:
                seen.add(key)
                unique.append(t)

    return unique


def pathrag_stats(original: list, filtered: list) -> dict:
    """Returns pruning statistics."""
    return {
        "original_count": len(original),
        "filtered_count": len(filtered),
        "pruned_count"  : len(original) - len(filtered),
        "retention_rate": round(len(filtered) / max(len(original), 1), 2),
    }