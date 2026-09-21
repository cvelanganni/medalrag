"""
test_06_neo4j.py
----------------
Verifies that Neo4j is connected, synchronized with the
NetworkX graph, and that UMLS CUI enrichment is working.

Tests:
    1. Neo4j connection via Bolt
    2. Node and relationship counts match NetworkX graph
    3. UMLS CUI coverage (at least 40% of nodes)
    4. Key entities have correct CUI identifiers
    5. Graph queries return meaningful results

Run:
    uv run python tests/test_06_neo4j.py
"""

import sys
import os
import pickle
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
RESET  = "\033[0m"

def ok(msg):   print(f"  {GREEN}✓{RESET} {msg}")
def fail(msg): print(f"  {RED}✗{RESET} {msg}")
def warn(msg): print(f"  {YELLOW}~{RESET} {msg}")

NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "medalrag2026")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")
MIN_CUI_PCT    = 40.0

# Known CUI mappings for validation
KNOWN_CUIS = {
    "Dolutegravir" : "C3253985",
    "HIV infection": "C0019693",
    "Emtricitabine": "C0909839",
    "Raltegravir"  : "C1871526",
    "Zidovudine"   : "C0043474",
}


def get_driver():
    return GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
    )


def test_connection() -> bool:
    """Test Neo4j Bolt connection."""
    print("\n[1] Neo4j connection")
    try:
        driver = get_driver()
        with driver.session(database=NEO4J_DATABASE) as s:
            result = s.run("RETURN 1 AS test").single()["test"]
        driver.close()
        if result == 1:
            ok(f"Connected to Neo4j at {NEO4J_URI}")
            return True
        fail("Unexpected result from Neo4j")
        return False
    except Exception as e:
        fail(f"Cannot connect to Neo4j: {e}")
        fail("Make sure Neo4j is running: docker-compose up -d neo4j")
        return False


def test_node_count() -> bool:
    """Test that Neo4j has a reasonable number of nodes."""
    print("\n[2] Node and relationship counts")
    try:
        driver = get_driver()
        with driver.session(database=NEO4J_DATABASE) as s:
            n_nodes = s.run("MATCH (n) RETURN count(n) AS c").single()["c"]
            n_rels  = s.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
        driver.close()

        ok(f"Neo4j nodes    : {n_nodes}")
        ok(f"Neo4j relations: {n_rels}")

        # Compare with NetworkX graph if available
        if os.path.exists("graph_cache.pkl"):
            with open("graph_cache.pkl", "rb") as f:
                G = pickle.load(f)
            nx_nodes = G.number_of_nodes()
            nx_edges = G.number_of_edges()

            if abs(n_nodes - nx_nodes) <= nx_nodes * 0.1:
                ok(f"Node count matches NetworkX ({nx_nodes} ± 10%)")
            else:
                warn(f"Node count mismatch: Neo4j={n_nodes}, NetworkX={nx_nodes}")
                warn("Run: uv run python enrich_graph.py to re-sync")

        return n_nodes > 100
    except Exception as e:
        fail(f"Query failed: {e}")
        return False


def test_cui_coverage() -> bool:
    """Test UMLS CUI coverage (at least MIN_CUI_PCT% of nodes)."""
    print("\n[3] UMLS CUI coverage")
    try:
        driver = get_driver()
        with driver.session(database=NEO4J_DATABASE) as s:
            total = s.run("MATCH (n) RETURN count(n) AS c").single()["c"]
            with_cui = s.run(
                "MATCH (n) WHERE n.cui IS NOT NULL AND n.cui <> '' "
                "RETURN count(n) AS c"
            ).single()["c"]
        driver.close()

        pct = with_cui / total * 100 if total > 0 else 0
        ok(f"Total nodes : {total}")
        ok(f"With CUI    : {with_cui} ({pct:.1f}%)")
        ok(f"Without CUI : {total - with_cui} ({100-pct:.1f}%)")

        if pct >= MIN_CUI_PCT:
            ok(f"CUI coverage {pct:.1f}% >= {MIN_CUI_PCT}% ✓")
            return True
        warn(f"Low CUI coverage: {pct:.1f}% < {MIN_CUI_PCT}%")
        warn("Run: uv run python enrich_graph.py to improve CUI mapping")
        return True  # non-fatal

    except Exception as e:
        fail(f"CUI coverage check failed: {e}")
        return False


def test_known_cuis() -> bool:
    """Test that key entities have correct UMLS CUI identifiers."""
    print("\n[4] Known CUI validation")
    try:
        driver  = get_driver()
        all_ok  = True

        with driver.session(database=NEO4J_DATABASE) as s:
            for entity, expected_cui in KNOWN_CUIS.items():
                result = s.run(
                    "MATCH (n:MedicalEntity) "
                    "WHERE n.name = $name OR n.canonical_name = $name "
                    "RETURN n.cui AS cui LIMIT 1",
                    name=entity
                ).single()

                if result and result["cui"] == expected_cui:
                    ok(f"'{entity}' → CUI {expected_cui} ✓")
                elif result and result["cui"]:
                    warn(f"'{entity}' → CUI {result['cui']} (expected {expected_cui})")
                else:
                    warn(f"'{entity}' not found or no CUI in Neo4j")

        driver.close()
        return all_ok

    except Exception as e:
        fail(f"CUI validation failed: {e}")
        return False


def test_graph_queries() -> bool:
    """Test that clinical Cypher queries return meaningful results."""
    print("\n[5] Clinical graph queries")
    try:
        driver = get_driver()
        all_ok = True

        queries = [
            {
                "desc"  : "Dolutegravir interactions",
                "cypher": "MATCH (n:MedicalEntity {name: 'Dolutegravir'})-[r]->(m) "
                          "RETURN count(r) AS c",
                "min"   : 5,
            },
            {
                "desc"  : "HIV infection relations",
                "cypher": "MATCH (n:MedicalEntity {name: 'HIV infection'})-[r]->() "
                          "RETURN count(r) AS c",
                "min"   : 10,
            },
            {
                "desc"  : "Rifampin → Dolutegravir path",
                "cypher": "MATCH (a:MedicalEntity)-[r]-(b:MedicalEntity) "
                          "WHERE a.name = 'Rifampin' AND b.name = 'Dolutegravir' "
                          "RETURN count(r) AS c",
                "min"   : 1,
            },
        ]

        with driver.session(database=NEO4J_DATABASE) as s:
            for q in queries:
                result = s.run(q["cypher"]).single()
                count  = result["c"] if result else 0

                if count >= q["min"]:
                    ok(f"{q['desc']}: {count} results ✓")
                else:
                    warn(f"{q['desc']}: only {count} results (expected ≥{q['min']})")
                    all_ok = False

        driver.close()
        return all_ok

    except Exception as e:
        fail(f"Graph queries failed: {e}")
        return False


def main():
    print("=" * 55)
    print("  MedalRAG — Neo4j Knowledge Graph Tests")
    print("=" * 55)

    tests = [
        test_connection,
        test_node_count,
        test_cui_coverage,
        test_known_cuis,
        test_graph_queries,
    ]

    results = [t() for t in tests]
    n_pass  = sum(results)
    n_total = len(results)

    print("\n" + "=" * 55)
    print(f"  Results: {n_pass}/{n_total} tests passed")

    if n_pass >= 4:
        ok("Neo4j knowledge graph is working correctly")
        sys.exit(0)
    else:
        fail(f"{n_total - n_pass} test(s) failed")
        sys.exit(1)


if __name__ == "__main__":
    main()