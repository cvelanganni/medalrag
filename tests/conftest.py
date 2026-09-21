# tests/conftest.py
import sys
import os
import pickle
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()


@pytest.fixture(scope="session")
def q():
    """Qdrant client fixture — shared across all tests."""
    from qdrant_client import QdrantClient
    return QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))


@pytest.fixture(scope="session")
def G():
    """NetworkX graph fixture — loaded once per session."""
    import networkx as nx
    if not os.path.exists("graph_cache.pkl"):
        pytest.skip("graph_cache.pkl not found — run build_pipeline.py first")
    with open("graph_cache.pkl", "rb") as f:
        return pickle.load(f)


@pytest.fixture(scope="session")
def neo4j_driver():
    """Neo4j driver fixture."""
    from neo4j import GraphDatabase
    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI",      "bolt://localhost:7687"),
        auth=(
            os.getenv("NEO4J_USERNAME", "neo4j"),
            os.getenv("NEO4J_PASSWORD", "medalrag2026")
        )
    )
    yield driver
    driver.close()