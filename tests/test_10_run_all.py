"""
test_10_run_all.py
------------------
Runs all MedalRAG tests in sequence and produces
a final health report of the entire pipeline.

Run:
    uv run python tests/test_10_run_all.py

Exit codes:
    0 = all tests passed
    1 = one or more tests failed
"""

import sys
import os
import subprocess
import time

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

TESTS = [
    ("test_01_docker_services.py",    "Docker Services"),
    ("test_02_embedding.py",          "BGE-M3 Embedding"),
    ("test_03_qdrant_collections.py", "Qdrant Collections"),
    ("test_04_retrieval.py",          "Retrieval Pipeline"),
    ("test_05_graph.py",              "Knowledge Graph"),
    ("test_06_neo4j.py",              "Neo4j Database"),
    ("test_07_reranker.py",           "Cross-Encoder Reranker"),
    ("test_08_ner.py",                "NER Models"),
    ("test_09_pipeline_e2e.py",       "End-to-End Pipeline"),
]


def run_test(filename: str) -> tuple:
    filepath     = os.path.join("tests", filename)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    t_start      = time.time()

    result = subprocess.run(
        [sys.executable, filepath],
        capture_output = True,
        text           = True,
        encoding       = "utf-8",      # ← ajoute ça
        errors         = "replace",    # ← ajoute ça
        cwd            = project_root,
        env            = {
            **os.environ,
            "PYTHONPATH"  : project_root,
            "PYTHONIOENCODING": "utf-8",   # ← ajoute ça
        }
    )

    elapsed = time.time() - t_start
    passed  = result.returncode == 0
    return passed, elapsed, result.stdout, result.stderr


def main():
    print("=" * 60)
    print(f"  {BOLD}MedalRAG — Full Pipeline Health Report{RESET}")
    print("=" * 60)
    print()

    results     = []
    total_start = time.time()

    for filename, label in TESTS:
        print(f"  Running {label}...", end="", flush=True)
        passed, elapsed, stdout, stderr = run_test(filename)

        status = f"{GREEN}PASS{RESET}" if passed else f"{RED}FAIL{RESET}"
        print(f"\r  [{status}] {label:<30} ({elapsed:.1f}s)")

        results.append({
            "label"  : label,
            "file"   : filename,
            "passed" : passed,
            "elapsed": elapsed,
            "stdout" : stdout,
            "stderr" : stderr,
        })

    total_elapsed = time.time() - total_start

    # Summary
    n_pass  = sum(1 for r in results if r["passed"])
    n_total = len(results)
    n_fail  = n_total - n_pass

    print()
    print("=" * 60)
    print(f"  {BOLD}SUMMARY{RESET}")
    print("=" * 60)
    print(f"  Total    : {n_total} test suites")
    print(f"  Passed   : {GREEN}{n_pass}{RESET}")
    print(f"  Failed   : {RED}{n_fail}{RESET}")
    print(f"  Duration : {total_elapsed:.1f}s")
    print()

    # Failed tests details
    failed = [r for r in results if not r["passed"]]
    if failed:
        print(f"  {RED}FAILED TESTS:{RESET}")
        for r in failed:
            print(f"  ✗ {r['label']} ({r['file']})")
            # Show last 5 lines of stdout
            lines = r["stdout"].strip().split("\n")
            for line in lines[-5:]:
                print(f"    {line}")
        print()

    # Final verdict
    print("=" * 60)
    if n_fail == 0:
        print(f"  {GREEN}{BOLD}✓ ALL TESTS PASSED — Pipeline ready{RESET}")
        print(f"  {GREEN}  Run: uv run streamlit run app.py{RESET}")
    elif n_fail <= 2:
        print(f"  {YELLOW}{BOLD}~ MOSTLY PASSING ({n_pass}/{n_total}){RESET}")
        print(f"  {YELLOW}  Check failed tests above{RESET}")
    else:
        print(f"  {RED}{BOLD}✗ MULTIPLE FAILURES — Pipeline needs attention{RESET}")
        print(f"  {RED}  Fix failed tests before running the app{RESET}")
    print("=" * 60)

    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()