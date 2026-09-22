"""
test_01_docker_services.py
--------------------------
Verifies that all required Docker services are running
and accessible on their expected ports.

Services checked:
    - Qdrant       : http://localhost:6333
    - Ollama       : http://localhost:11434
    - Neo4j Bolt   : bolt://localhost:7687
    - Neo4j HTTP   : http://localhost:7474
    - Infinity     : http://localhost:7997 (optional)

Run:
    uv run python tests/test_01_docker_services.py
"""

import sys
import socket
import requests

# ── Color helpers ──────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
RESET  = "\033[0m"

def ok(msg):   print(f"  {GREEN}✓{RESET} {msg}")
def fail(msg): print(f"  {RED}✗{RESET} {msg}")
def warn(msg): print(f"  {YELLOW}~{RESET} {msg}")


# ── Helpers ────────────────────────────────────────────────────

def check_http(name: str, url: str, timeout: int = 3) -> bool:
    """Check if an HTTP endpoint responds with 200."""
    try:
        resp = requests.get(url, timeout=timeout)
        if resp.status_code < 500:
            ok(f"{name} reachable at {url}")
            return True
        fail(f"{name} returned {resp.status_code} at {url}")
        return False
    except Exception as e:
        fail(f"{name} not reachable at {url} — {e}")
        return False


def check_port(name: str, host: str, port: int, timeout: int = 3) -> bool:
    """Check if a TCP port is open."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            ok(f"{name} port {port} open")
            return True
    except Exception:
        fail(f"{name} port {port} closed or unreachable")
        return False


# ── Tests ──────────────────────────────────────────────────────

def test_qdrant() -> bool:
    print("\n[1] Qdrant (vector database)")
    return check_http("Qdrant", "http://localhost:6333/healthz")


def test_ollama() -> bool:
    print("\n[2] Ollama (embedding + LLM)")
    ok_http = check_http("Ollama", "http://localhost:11434/api/tags")

    if ok_http:
        # Verify required models are available
        resp     = requests.get("http://localhost:11434/api/tags", timeout=5)
        models   = [m["name"] for m in resp.json().get("models", [])]
        required = ["bge-m3:latest", "gemma3:1b"]

        for model in required:
            if any(model in m for m in models):
                ok(f"Model available: {model}")
            else:
                fail(f"Model MISSING: {model} — run: docker exec ollama_lmph ollama pull {model}")
                ok_http = False

    return ok_http


def test_neo4j() -> bool:
    print("\n[3] Neo4j (knowledge graph)")
    bolt_ok = check_port("Neo4j Bolt", "localhost", 7687)

    # Neo4j HTTP can be slow to start
    # Verify via a real Bolt connection
    try:
        from neo4j import GraphDatabase
        import os
        from dotenv import load_dotenv
        load_dotenv()

        driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            auth=(
                os.getenv("NEO4J_USERNAME", "neo4j"),
                os.getenv("NEO4J_PASSWORD", "medalrag2026")
            )
        )
        with driver.session() as session:
            result = session.run("RETURN 1 AS test").single()["test"]
        driver.close()

        if result == 1:
            ok("Neo4j Bolt connection successful")
            return True
    except Exception as e:
        fail(f"Neo4j Bolt connection failed — {e}")
        return False

    return bolt_ok


def test_infinity() -> bool:
    print("\n[4] Infinity (BGE-Reranker — optional)")
    try:
        resp = requests.get("http://localhost:7997/models", timeout=3)
        if resp.status_code == 200:
            ok("Infinity reranker reachable at http://localhost:7997")
            return True
        warn("Infinity running but returned unexpected status")
        return False
    except Exception:
        warn("Infinity not running — reranker will fallback to ms-marco")
        return False  # optional, not a fatal error


# ── Main ───────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  MedalRAG — Docker Services Health Check")
    print("=" * 55)

    results = {
        "qdrant"  : test_qdrant(),
        "ollama"  : test_ollama(),
        "neo4j"   : test_neo4j(),
        "infinity": test_infinity(),  # optional
    }

    # Summary
    print("\n" + "=" * 55)
    required_pass = results["qdrant"] and results["ollama"] and results["neo4j"]
    total_pass    = sum(results.values())

    print(f"  Results: {total_pass}/{len(results)} services OK")
    print()

    if required_pass:
        ok("All required services are running — pipeline ready")
        sys.exit(0)
    else:
        fail("One or more REQUIRED services are down")
        print()
        print("  Fix suggestions:")
        if not results["qdrant"]:
            print("    → docker-compose up -d qdrant")
        if not results["ollama"]:
            print("    → docker-compose up -d ollama")
        if not results["neo4j"]:
            print("    → docker-compose up -d neo4j")
        sys.exit(1)


if __name__ == "__main__":
    main()
