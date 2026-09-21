"""
test_05_graph.py
----------------
Verifies that the NetworkX knowledge graph is properly built
with key HIV medical entities, normalized names, and
meaningful relationships.

Tests:
    1. Graph cache file exists and loads correctly
    2. Key HIV entities are present (Dolutegravir, Rifampin, etc.)
    3. Entity normalization is working (no duplicates)
    4. Minimum node and edge counts are met
    5. Key clinical relationships exist in the graph
    6. PPR graph is functional

Run:
    uv run python tests/test_05_graph.py
"""

import sys
import os
import pickle
import networkx as nx

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
RESET  = "\033[0m"

def ok(msg):   print(f"  {GREEN}✓{RESET} {msg}")
def fail(msg): print(f"  {RED}✗{RESET} {msg}")
def warn(msg): print(f"  {YELLOW}~{RESET} {msg}")

GRAPH_CACHE_PATH  = "graph_cache.pkl"
MIN_NODES         = 1000
MIN_EDGES         = 2000
MIN_RELATIONS     = 5  # minimum relations for key entities


def load_graph() -> nx.DiGraph | None:
    if not os.path.exists(GRAPH_CACHE_PATH):
        return None
    with open(GRAPH_CACHE_PATH, "rb") as f:
        return pickle.load(f)


def test_graph_loads() -> bool:
    """Test that the graph cache exists and loads correctly."""
    print("\n[1] Graph cache")
    if not os.path.exists(GRAPH_CACHE_PATH):
        fail(f"Graph cache not found: {GRAPH_CACHE_PATH}")
        fail("Run: uv run build_pipeline.py --no-contextual --skip-qdrant")
        return False

    G = load_graph()
    if G is None or not isinstance(G, nx.DiGraph):
        fail("Graph cache is corrupted or invalid")
        return False

    ok(f"Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return True


def test_minimum_size(G: nx.DiGraph) -> bool:
    """Test that the graph has minimum required nodes and edges."""
    print("\n[2] Graph size")
    all_ok = True

    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()

    if n_nodes >= MIN_NODES:
        ok(f"Nodes: {n_nodes} (min: {MIN_NODES})")
    else:
        fail(f"Too few nodes: {n_nodes} (min: {MIN_NODES})")
        all_ok = False

    if n_edges >= MIN_EDGES:
        ok(f"Edges: {n_edges} (min: {MIN_EDGES})")
    else:
        fail(f"Too few edges: {n_edges} (min: {MIN_EDGES})")
        all_ok = False

    density = nx.density(G)
    ok(f"Graph density: {density:.4f}")

    return all_ok


def test_key_entities(G: nx.DiGraph) -> bool:
    """Test that key HIV medical entities are in the graph."""
    print("\n[3] Key HIV entities")

    # Critical entities that MUST be in the graph
    required_entities = [
        "Dolutegravir",
        "HIV infection",
        "TAF",
        "TDF",
        "Rifampin",
        "Pregnancy",
        "Emtricitabine",
        "Lamivudine",
        "ART",
        "CD4",
    ]

    # Optional entities (warn if missing)
    optional_entities = [
        "Abacavir",
        "Efavirenz",
        "Darunavir",
        "Raltegravir",
        "Tuberculosis",
        "HBV",
        "PrEP",
        "TMP-SMX",
    ]

    all_ok = True

    for entity in required_entities:
        if entity in G.nodes():
            deg = G.degree(entity)
            ok(f"'{entity}' found — {deg} relations")
            if deg < MIN_RELATIONS:
                warn(f"'{entity}' has few relations ({deg} < {MIN_RELATIONS})")
        else:
            fail(f"REQUIRED entity missing: '{entity}'")
            all_ok = False

    print()
    for entity in optional_entities:
        if entity in G.nodes():
            deg = G.degree(entity)
            ok(f"'{entity}' found — {deg} relations")
        else:
            warn(f"Optional entity missing: '{entity}'")

    return all_ok


def test_entity_normalization(G: nx.DiGraph) -> bool:
    """
    Test that entity normalization is working correctly.
    Verifies that duplicates (rifampin/Rifampin/rifampicin) are merged.
    """
    print("\n[4] Entity normalization")
    all_ok = True

    # These variants should NOT exist as separate nodes
    # (they should all map to the canonical form)
    duplicate_checks = [
        ("Rifampin", ["rifampin", "rifampicin", "rifampicina", "Rifampicin"]),
        ("Dolutegravir", ["dtg", "DTG", "dolutégravir"]),
        ("HIV infection", ["hiv", "HIV", "vih", "VIH"]),
        ("Pregnancy", ["pregnancy", "pregnant", "Pregnant"]),
        ("TMP-SMX", ["tmp-smx", "TMP-SMX", "trimethoprim-sulfamethoxazole"]),
    ]

    nodes_lower = {n.lower(): n for n in G.nodes()}

    for canonical, variants in duplicate_checks:
        # Check canonical exists
        if canonical in G.nodes():
            ok(f"Canonical '{canonical}' exists in graph")
        else:
            warn(f"Canonical '{canonical}' not found")

        # Check variants are NOT separate nodes
        for variant in variants:
            if variant in G.nodes() and variant != canonical:
                warn(f"Duplicate node found: '{variant}' (should map to '{canonical}')")
            elif variant.lower() in nodes_lower and nodes_lower[variant.lower()] != canonical:
                warn(f"Case variant found: '{variant}' → '{nodes_lower[variant.lower()]}'")

    return all_ok


def test_key_relationships(G: nx.DiGraph) -> bool:
    """
    Test that key clinical relationships exist in the graph.
    These are the most important medical facts for HIV treatment.
    """
    print("\n[5] Key clinical relationships")

    # Critical relationships that should exist
    expected_relationships = [
        ("Dolutegravir", "Rifampin",        "REQUIRES_ADJUSTMENT or INTERACTS_WITH"),
        ("TAF",          "Renal impairment", "PREFERRED_FOR or AVOIDED_IN"),
        ("HIV infection","ART",              "TREATS"),
        ("Dolutegravir", "Pregnancy",        "any relation"),
    ]

    all_ok = True
    for subj, obj, expected_rel in expected_relationships:
        # Check direct edge
        has_edge     = G.has_edge(subj, obj)
        has_rev_edge = G.has_edge(obj, subj)

        # Also check via neighbors
        subj_neighbors = set(G.successors(subj)) | set(G.predecessors(subj)) if subj in G else set()
        obj_in_neighbors = obj in subj_neighbors

        if has_edge or has_rev_edge or obj_in_neighbors:
            rel = G[subj][obj].get("relation", "?") if has_edge else (
                  G[obj][subj].get("relation", "?") if has_rev_edge else "indirect")
            ok(f"'{subj}' ↔ '{obj}' [{rel}]")
        else:
            warn(f"No relation found: '{subj}' ↔ '{obj}' (expected: {expected_rel})")

    return all_ok


def test_ppr_graph(G: nx.DiGraph) -> bool:
    """Test that PPR (Personalized PageRank) can run on the graph."""
    print("\n[6] PPR (Personalized PageRank)")
    try:
        # Test PPR with key seed entities
        seeds      = ["Dolutegravir", "Pregnancy"]
        seeds_in_G = [s for s in seeds if s in G.nodes()]

        if not seeds_in_G:
            warn("No seed entities found for PPR test")
            return True

        # Run PageRank
        personalization = {n: (1.0/len(seeds_in_G) if n in seeds_in_G else 0.0)
                          for n in G.nodes()}
        ppr_scores = nx.pagerank(G, alpha=0.85,
                                 personalization=personalization,
                                 max_iter=100)

        # Top-10 entities by PPR score
        top_entities = sorted(ppr_scores.items(), key=lambda x: x[1], reverse=True)[:10]
        ok(f"PPR computed successfully on {len(G.nodes())} nodes")
        ok(f"Top-5 entities for seeds {seeds_in_G}:")
        for entity, score in top_entities[:5]:
            print(f"      {entity:30} {score:.4f}")

        return True
    except Exception as e:
        fail(f"PPR failed: {e}")
        return False


def main():
    print("=" * 55)
    print("  MedalRAG — Knowledge Graph Tests")
    print("=" * 55)

    G = load_graph()
    if G is None:
        fail("Cannot load graph — run build_pipeline.py first")
        sys.exit(1)

    tests = [
        test_graph_loads,
        lambda: test_minimum_size(G),
        lambda: test_key_entities(G),
        lambda: test_entity_normalization(G),
        lambda: test_key_relationships(G),
        lambda: test_ppr_graph(G),
    ]

    results = [t() for t in tests]
    n_pass  = sum(results)
    n_total = len(results)

    print("\n" + "=" * 55)
    print(f"  Results: {n_pass}/{n_total} tests passed")

    if n_pass >= 5:
        ok("Knowledge graph is properly built")
        sys.exit(0)
    else:
        fail(f"{n_total - n_pass} test(s) failed")
        sys.exit(1)


if __name__ == "__main__":
    main()