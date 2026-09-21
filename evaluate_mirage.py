"""
evaluate_mirage.py — MedalRAG evaluation on the MIRAGE HIV benchmark.

Format  : Multiple choice questions (A/B/C/D)
Metric  : Accuracy (% correct answers)
Usage   : uv run evaluate_mirage.py [--max N] [--sleep 0.5] [--dataset path]
"""

import os
import re
import json
import time
import requests
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient

load_dotenv()

openai_client = OpenAI()
qdrant        = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))

COLLECTION_EN = os.getenv("QDRANT_COLLECTION_EN", "medical_docs_en")
COLLECTION_ES = os.getenv("QDRANT_COLLECTION_ES", "medical_docs_es")
OLLAMA_URL    = "http://localhost:11434/api/embed"
EMBEDDING_DIM = 1024


# ── Helpers ────────────────────────────────────────────────────────────────

def get_embedding(text: str) -> list:
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": "bge-m3", "input": text[:8000]},
            timeout=60,
        )
        return resp.json().get("embeddings", [[0.0] * EMBEDDING_DIM])[0]
    except Exception:
        return [0.0] * EMBEDDING_DIM


def search_qdrant(query: str, collection: str, limit: int = 5) -> list:
    emb = get_embedding(
        f"Represent this sentence for searching relevant passages: {query}"
    )
    try:
        return qdrant.query_points(
            collection_name=collection,
            query=emb, with_payload=True, limit=limit,
        ).points
    except Exception:
        return []


def format_context(chunks_en: list, chunks_es: list) -> str:
    """Formats retrieved chunks into a compact context string."""
    context = ""
    for i, c in enumerate(chunks_en):
        text     = c.payload.get("original_text", c.payload.get("text", ""))
        context += f"\n[EN Source {i+1}] {text[:300]}\n"
    for i, c in enumerate(chunks_es):
        text     = c.payload.get("original_text", c.payload.get("text", ""))
        context += f"\n[ES Source {i+1}] {text[:300]}\n"
    return context


def extract_answer(response: str, options: dict) -> str:
    """Extracts the answer letter (A/B/C/D) from the LLM response."""
    response_upper = response.upper().strip()
    patterns = [
        r'ANSWER[:\s]+([ABCD])',
        r'THE ANSWER IS[:\s]+([ABCD])',
        r'CORRECT ANSWER[:\s]+([ABCD])',
        r'^([ABCD])[:\.\)]\s',
        r'\*\*([ABCD])\*\*',
        r'\(([ABCD])\)',
    ]
    for pattern in patterns:
        match = re.search(pattern, response_upper)
        if match:
            return match.group(1)
    for letter, text in options.items():
        if text.lower()[:30] in response.lower():
            return letter
    for letter in ['A', 'B', 'C', 'D']:
        if response_upper.startswith(letter):
            return letter
    return "?"


def answer_mcq(question: str, options: dict, context: str) -> str:
    """Answers a multiple choice question using RAG context."""
    options_text = "\n".join([f"{letter}. {text}" for letter, text in options.items()])
    prompt = f"""You are an expert HIV/AIDS medical knowledge system.
Answer this multiple choice question using the provided context from HIV guidelines.

Context from HIV Guidelines:
{context or "No specific context retrieved."}

Question: {question}

Options:
{options_text}

Respond with ONLY the letter (A, B, C, or D). No explanation.

Answer:"""
    try:
        resp = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=10,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"    LLM error: {e}")
        return "?"


# ── Evaluation ─────────────────────────────────────────────────────────────

def evaluate_mirage(
    dataset_path : str   = "mirage_hiv_strict.json",
    max_questions: int   = None,
    sleep_between: float = 0.5,
):
    with open(dataset_path, encoding="utf-8") as f:
        questions = json.load(f)
    if max_questions:
        questions = questions[:max_questions]

    print("=" * 60)
    print("  MedalRAG — MIRAGE HIV Evaluation")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Questions: {len(questions)}")
    print("=" * 60)

    results, correct, total, by_dataset = [], 0, 0, {}

    for i, q in enumerate(questions):
        qid      = q["id"]
        dataset  = q["dataset"]
        question = q["question"]
        options  = q["options"]
        answer   = q["answer"]

        print(f"\n[{i+1}/{len(questions)}] ({dataset}) {question[:60]}...")

        chunks_en  = search_qdrant(question, COLLECTION_EN, limit=5)
        chunks_es  = search_qdrant(question, COLLECTION_ES, limit=3)
        context    = format_context(chunks_en, chunks_es)
        raw_answer = answer_mcq(question, options, context)
        predicted  = extract_answer(raw_answer, options)
        is_correct = predicted == answer

        if is_correct:
            correct += 1
        total += 1

        by_dataset.setdefault(dataset, {"correct": 0, "total": 0})
        by_dataset[dataset]["total"]   += 1
        by_dataset[dataset]["correct"] += int(is_correct)

        print(f"  Predicted: {predicted} | Correct: {answer} | {'✓' if is_correct else '✗'}")

        results.append({
            "id"        : qid,
            "dataset"   : dataset,
            "question"  : question[:100],
            "predicted" : predicted,
            "correct"   : answer,
            "is_correct": is_correct,
            "chunks_en" : len(chunks_en),
            "chunks_es" : len(chunks_es),
        })

        time.sleep(sleep_between)

    accuracy = correct / total if total > 0 else 0

    print("\n" + "=" * 60)
    print("  MIRAGE HIV EVALUATION — RESULTS")
    print("=" * 60)
    print(f"  Overall Accuracy: {accuracy:.3f} ({correct}/{total})")
    print("\n  By dataset:")
    for ds, stats in by_dataset.items():
        acc = stats["correct"] / stats["total"]
        print(f"    {ds:12}: {acc:.3f} ({stats['correct']}/{stats['total']})")
    print("\n  MIRAGE Baselines (from paper):")
    print("    No RAG (GPT-4o) : ~0.780")
    print("    BM25 + GPT-4o   : ~0.820")
    print("    MedRAG best     : ~0.860")
    print(f"    MedalRAG (ours) : {accuracy:.3f}")

    output = {
        "timestamp"   : datetime.now().isoformat(),
        "dataset_path": dataset_path,
        "n_questions" : total,
        "accuracy"    : accuracy,
        "correct"     : correct,
        "by_dataset"  : by_dataset,
        "results"     : results,
    }

    out_path = f"evaluations/mirage_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs("evaluations", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved: {out_path}")
    return output


# ── Entry Point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max",     type=int,   default=None,
                        help="Max questions to evaluate (default: all)")
    parser.add_argument("--sleep",   type=float, default=0.5,
                        help="Sleep between questions in seconds (default: 0.5)")
    parser.add_argument("--dataset", type=str,   default="mirage_hiv_strict.json",
                        help="Path to MIRAGE dataset JSON")
    args = parser.parse_args()

    evaluate_mirage(
        dataset_path  = args.dataset,
        max_questions = args.max,
        sleep_between = args.sleep,
    )