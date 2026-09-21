"""
neo4j_sync.py — Syncs the NetworkX graph to Neo4j for visualization.

Note: Neo4j is used for interactive visualization only.
      All computation (PPR, PathRAG) runs on the in-memory NetworkX graph.
"""

import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI", "bolt://localhost:7687"),
    auth=(
        os.getenv("NEO4J_USERNAME", "neo4j"),
        os.getenv("NEO4J_PASSWORD", "password")
    )
)
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")


def clear_graph():
    with driver.session(database=NEO4J_DATABASE) as session:
        session.run("MATCH (n) DETACH DELETE n")
    print("  Neo4j cleared")


def sync_graph_to_neo4j(G, batch_size: int = 500):
    """Syncs a NetworkX graph to Neo4j using batched writes."""
    nodes = list(G.nodes(data=True))
    edges = list(G.edges(data=True))
    print(f"  Syncing {len(nodes)} nodes and {len(edges)} edges to Neo4j...")

    with driver.session(database=NEO4J_DATABASE) as session:
        # Sync nodes
        for i in range(0, len(nodes), batch_size):
            batch = nodes[i:i + batch_size]
            session.run("""
                UNWIND $nodes AS node
                MERGE (n:MedicalEntity {name: node.name})
                SET n.lang = node.lang
            """, nodes=[{"name": n, "lang": d.get("lang", "en")} for n, d in batch])

        # Sync edges (one by one — relation type must be dynamic)
        for u, v, data in edges:
            rel_type = data.get("relation", "RELATED_TO")
            session.run(f"""
                MATCH (s:MedicalEntity {{name: $subject}})
                MATCH (o:MedicalEntity {{name: $object}})
                MERGE (s)-[r:{rel_type}]->(o)
                SET r.confidence = $confidence,
                    r.source     = $source,
                    r.lang       = $lang
            """,
                subject    = u,
                object     = v,
                confidence = data.get("confidence", 0.8),
                source     = data.get("source", "unknown"),
                lang       = data.get("lang", "en"),
            )

    print(f"  Neo4j sync complete: {len(nodes)} nodes, {len(edges)} edges")


def get_neo4j_stats() -> dict:
    """Returns node and relation counts from Neo4j."""
    with driver.session(database=NEO4J_DATABASE) as session:
        nodes     = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
        relations = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
    return {"nodes": nodes, "relations": relations}