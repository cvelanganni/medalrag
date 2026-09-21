"""
enrich_graph.py — Enriches the NetworkX graph with UMLS CUI codes.

Process:
  1. Load graph_cache.pkl
  2. Query the UMLS API for each node → retrieve CUI
  3. Merge nodes sharing the same CUI (duplicates)
  4. Save graph_cache_umls.pkl
  5. Re-sync to Neo4j

Usage: uv run enrich_graph.py
"""

import os
import time
import json
import pickle
import requests
import networkx as nx
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

GRAPH_CACHE_PATH    = "graph_cache.pkl"
GRAPH_ENRICHED_PATH = "graph_cache_umls.pkl"
UMLS_CACHE_PATH     = "umls_cache.json"
UMLS_API_KEY        = os.getenv("UMLS_API_KEY", "")
UMLS_BASE_URL       = "https://uts-ws.nlm.nih.gov/rest"


# ── UMLS Local Cache ───────────────────────────────────────────────────────

def load_umls_cache() -> dict:
    if Path(UMLS_CACHE_PATH).exists():
        with open(UMLS_CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_umls_cache(cache: dict):
    with open(UMLS_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


# ── UMLS API ───────────────────────────────────────────────────────────────

def get_cui_from_umls(term: str, cache: dict) -> dict:
    """
    Queries the UMLS API for a medical term's CUI.
    Uses a local cache to avoid repeated API calls.

    Returns {"cui", "name", "semantic_type", "source"} or {} if not found.
    """
    term_key = term.lower().strip()
    if term_key in cache:
        return cache[term_key]
    if not UMLS_API_KEY or len(term) > 200 or len(term) < 2:
        cache[term_key] = {}
        return {}

    try:
        resp = requests.get(
            f"{UMLS_BASE_URL}/search/current",
            params={
                "string"      : term,
                "apiKey"      : UMLS_API_KEY,
                "sabs"        : "RXNORM,SNOMEDCT_US,NCI,MSH,DRUGBANK",
                "returnIdType": "concept",
                "pageSize"    : 1,
            },
            timeout=10,
        )
        if resp.status_code == 401:
            print("  UMLS API key invalid or expired!")
            return {}
        if resp.status_code != 200:
            cache[term_key] = {}
            return {}

        results = resp.json().get("result", {}).get("results", [])
        if not results or results[0].get("ui") == "NONE":
            cache[term_key] = {}
            return {}

        r      = results[0]
        result = {
            "cui"   : r.get("ui", ""),
            "name"  : r.get("name", term),
            "source": r.get("rootSource", ""),
        }

        # Fetch semantic type
        try:
            concept_resp = requests.get(
                f"{UMLS_BASE_URL}/content/current/CUI/{result['cui']}",
                params={"apiKey": UMLS_API_KEY},
                timeout=10,
            )
            if concept_resp.status_code == 200:
                sem_types = concept_resp.json().get("result", {}).get("semanticTypes", [])
                if sem_types:
                    result["semantic_type"] = sem_types[0].get("name", "")
        except Exception:
            pass

        cache[term_key] = result
        return result

    except requests.exceptions.Timeout:
        print(f"  Timeout for '{term[:30]}'")
        return {}
    except Exception as e:
        print(f"  Error for '{term[:30]}': {e}")
        return {}


# ── Graph Enrichment ───────────────────────────────────────────────────────

def enrich_nodes_with_umls(G: nx.DiGraph, cache: dict) -> tuple:
    """Adds CUI, UMLS name and semantic type to each graph node."""
    nodes     = list(G.nodes())
    cui_map   = {}
    found     = 0
    not_found = 0

    print(f"  Querying UMLS for {len(nodes)} nodes ({len(cache)} already cached)...")

    for i, node in enumerate(nodes):
        umls_data = get_cui_from_umls(node, cache)
        if umls_data.get("cui"):
            G.nodes[node]["cui"]           = umls_data["cui"]
            G.nodes[node]["umls_name"]     = umls_data.get("name", node)
            G.nodes[node]["umls_source"]   = umls_data.get("source", "")
            G.nodes[node]["semantic_type"] = umls_data.get("semantic_type", "")
            cui_map[node] = umls_data["cui"]
            found += 1
        else:
            G.nodes[node]["cui"] = ""
            not_found += 1

        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{len(nodes)} ({(i+1)/len(nodes)*100:.0f}%) | "
                  f"found: {found} | not found: {not_found}")
            save_umls_cache(cache)
            time.sleep(0.2)  # UMLS rate limit: ~20 req/sec

        time.sleep(0.05)

    save_umls_cache(cache)
    print(f"  UMLS enrichment: {found}/{len(nodes)} nodes matched "
          f"({found/len(nodes)*100:.1f}%)")
    return G, cui_map


def merge_duplicate_nodes(G: nx.DiGraph, cui_map: dict) -> nx.DiGraph:
    """
    Merges nodes that share the same UMLS CUI.
    Canonical name = UMLS official name, or highest-degree node as fallback.
    Redirects all incoming/outgoing edges before removing duplicates.
    """
    cui_groups  = {}
    for node, cui in cui_map.items():
        if cui:
            cui_groups.setdefault(cui, []).append(node)

    duplicates = {cui: nodes for cui, nodes in cui_groups.items() if len(nodes) > 1}
    if not duplicates:
        print("  No duplicate nodes found via UMLS CUI.")
        return G

    print(f"  Found {len(duplicates)} CUI groups with duplicates:")
    G_merged, merged_count = G.copy(), 0

    for cui, nodes in duplicates.items():
        umls_name = G.nodes[nodes[0]].get("umls_name", "")
        canonical = umls_name if (umls_name and umls_name in nodes) else \
                    max(nodes, key=lambda n: G.degree(n) if n in G else 0)
        others = [n for n in nodes if n != canonical]

        for other in others:
            if other not in G_merged:
                continue
            for pred in list(G.predecessors(other)):
                if pred != canonical and G_merged.has_node(pred):
                    data = G.get_edge_data(pred, other, {})
                    if not G_merged.has_edge(pred, canonical):
                        G_merged.add_edge(pred, canonical, **data)
            for succ in list(G.successors(other)):
                if succ != canonical and G_merged.has_node(succ):
                    data = G.get_edge_data(other, succ, {})
                    if not G_merged.has_edge(canonical, succ):
                        G_merged.add_edge(canonical, succ, **data)
            G_merged.remove_node(other)
            merged_count += 1

    print(f"  Merged {merged_count} duplicate nodes: "
          f"{G.number_of_nodes()} → {G_merged.number_of_nodes()} nodes, "
          f"{G.number_of_edges()} → {G_merged.number_of_edges()} edges")
    return G_merged


# ── Neo4j Sync ─────────────────────────────────────────────────────────────

def sync_to_neo4j(G: nx.DiGraph):
    """Re-syncs the enriched graph to Neo4j with UMLS metadata."""
    from neo4j import GraphDatabase
    driver   = GraphDatabase.driver(
        os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        auth=(os.getenv("NEO4J_USERNAME", "neo4j"), os.getenv("NEO4J_PASSWORD", ""))
    )
    NEO4J_DB = os.getenv("NEO4J_DATABASE", "neo4j")
    BATCH    = 500

    print("  Clearing Neo4j...")
    with driver.session(database=NEO4J_DB) as session:
        session.run("MATCH (n) DETACH DELETE n")

    nodes = list(G.nodes(data=True))
    edges = list(G.edges(data=True))

    print(f"  Syncing {len(nodes)} nodes...")
    with driver.session(database=NEO4J_DB) as session:
        for i in range(0, len(nodes), BATCH):
            batch = nodes[i:i + BATCH]
            session.run("""
                UNWIND $nodes AS n
                MERGE (node:MedicalEntity {name: n.name})
                SET node.cui           = n.cui,
                    node.umls_name     = n.umls_name,
                    node.semantic_type = n.semantic_type,
                    node.umls_source   = n.umls_source,
                    node.lang          = n.lang
            """, nodes=[{
                "name"         : name,
                "cui"          : data.get("cui", ""),
                "umls_name"    : data.get("umls_name", name),
                "semantic_type": data.get("semantic_type", ""),
                "umls_source"  : data.get("umls_source", ""),
                "lang"         : data.get("lang", "en"),
            } for name, data in batch])
            print(f"    Nodes: {min(i+BATCH, len(nodes))}/{len(nodes)}")

    print(f"  Syncing {len(edges)} edges...")
    ok, fail = 0, 0
    with driver.session(database=NEO4J_DB) as session:
        for u, v, data in edges:
            rel_type = data.get("relation", "RELATED_TO").replace(" ", "_").upper()
            try:
                session.run(f"""
                    MATCH (s:MedicalEntity {{name: $s}})
                    MATCH (o:MedicalEntity {{name: $o}})
                    MERGE (s)-[r:{rel_type}]->(o)
                    SET r.confidence = $conf,
                        r.source     = $src,
                        r.lang       = $lang
                """, s=u, o=v,
                    conf=data.get("confidence", 0.8),
                    src=data.get("source", "unknown"),
                    lang=data.get("lang", "en"),
                )
                ok += 1
            except Exception:
                fail += 1

    print(f"  Neo4j sync complete: {ok} edges OK, {fail} failed")
    driver.close()


# ── UMLS Stats ─────────────────────────────────────────────────────────────

def print_umls_stats(G: nx.DiGraph):
    """Prints UMLS enrichment statistics for the graph."""
    nodes_with    = [(n, d) for n, d in G.nodes(data=True) if d.get("cui")]
    nodes_without = [(n, d) for n, d in G.nodes(data=True) if not d.get("cui")]
    total         = G.number_of_nodes()

    sem_types = {}
    sources   = {}
    for _, d in nodes_with:
        sem_types[d.get("semantic_type", "Unknown")] = \
            sem_types.get(d.get("semantic_type", "Unknown"), 0) + 1
        sources[d.get("umls_source", "Unknown")] = \
            sources.get(d.get("umls_source", "Unknown"), 0) + 1

    print(f"\n  UMLS Stats:")
    print(f"    With CUI    : {len(nodes_with)} ({len(nodes_with)/total*100:.1f}%)")
    print(f"    Without CUI : {len(nodes_without)} ({len(nodes_without)/total*100:.1f}%)")
    print(f"\n  Top Semantic Types:")
    for st, count in sorted(sem_types.items(), key=lambda x: -x[1])[:10]:
        print(f"    {count:4d} × {st}")
    print(f"\n  Top Sources:")
    for src, count in sorted(sources.items(), key=lambda x: -x[1])[:5]:
        print(f"    {count:4d} × {src}")
    print(f"\n  Normalization examples:")
    shown = 0
    for n, d in G.nodes(data=True):
        if d.get("cui") and d.get("umls_name") and d.get("umls_name") != n:
            print(f"    '{n}' → CUI {d['cui']} → '{d['umls_name']}'")
            shown += 1
            if shown >= 10:
                break


# ── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from datetime import datetime
    print("=" * 60)
    print("  MedalRAG — UMLS Graph Enrichment")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    if not UMLS_API_KEY:
        print("\n  UMLS_API_KEY not set in .env")
        print("  Get your free key at: https://uts.nlm.nih.gov")
        exit(1)

    # [1/5] Load graph
    print("\n[1/5] Loading graph...")
    if not Path(GRAPH_CACHE_PATH).exists():
        print(f"  {GRAPH_CACHE_PATH} not found — run build_pipeline.py first.")
        exit(1)
    with open(GRAPH_CACHE_PATH, "rb") as f:
        G = pickle.load(f)
    print(f"  Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # [2/5] Load UMLS cache
    print("\n[2/5] Loading UMLS local cache...")
    cache = load_umls_cache()
    print(f"  {len(cache)} terms already cached")

    # Test API key
    test = get_cui_from_umls("dolutegravir", cache)
    if not test.get("cui"):
        print("  API test failed — check your UMLS_API_KEY")
        exit(1)
    print(f"  API OK — dolutegravir → {test['cui']} ({test.get('name')})")

    # [3/5] Enrich
    print("\n[3/5] Enriching nodes with UMLS CUI...")
    G_enriched, cui_map = enrich_nodes_with_umls(G, cache)

    # [4/5] Merge duplicates
    print("\n[4/5] Merging duplicate nodes (same CUI)...")
    G_merged = merge_duplicate_nodes(G_enriched, cui_map)
    print_umls_stats(G_merged)

    # [5/5] Save and sync
    print(f"\n[5/5] Saving enriched graph to {GRAPH_ENRICHED_PATH}...")
    with open(GRAPH_ENRICHED_PATH, "wb") as f:
        pickle.dump(G_merged, f)
    print(f"  Saved: {G_merged.number_of_nodes()} nodes, {G_merged.number_of_edges()} edges")

    print("\n[5/5b] Syncing to Neo4j...")
    try:
        sync_to_neo4j(G_merged)
    except Exception as e:
        print(f"  Neo4j sync failed: {e} — enriched graph saved locally.")

    print("\n" + "=" * 60)
    print("  UMLS enrichment complete!")
    print("=" * 60)
    print("""
  Next steps:
    1. Replace main graph:
       Copy-Item graph_cache_umls.pkl graph_cache.pkl
    2. Re-run benchmark to measure impact:
       uv run evaluate.py --quick
""")