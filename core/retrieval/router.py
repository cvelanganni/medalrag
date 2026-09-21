"""
router.py — Query complexity classifier and pipeline router.
Uses gemma3:1b (local, ~20ms) with GPT-4o-mini as fallback.
"""

import requests
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

openai_client = OpenAI()
OLLAMA_URL    = "http://localhost:11434/api/generate"


# ── Complexity Classification ──────────────────────────────────────────────

def classify_complexity_local(question: str) -> str:
    """
    Classifies query complexity using gemma3:1b (local, free, ~20ms).
    Falls back to GPT-4o-mini if Ollama is unavailable.
    """
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model": "gemma3:1b",
                "prompt": f"""Classify this HIV clinical question as SIMPLE or COMPLEX.

COMPLEX if it mentions:
- Specific drug names (Dolutegravir, Rifampin, Tenofovir...)
- Drug-drug interactions or dose adjustments
- Comorbidities (TB, hepatitis B/C, pregnancy, renal failure...)
- Co-infections or opportunistic infections (Toxoplasma, MAC, CMV, PCP...)
- Specific regimen combinations

SIMPLE if it asks:
- General guidelines (when to start ART, monitoring frequency)
- Definitions (what is viral suppression, what is CD4)
- General goals of therapy

Respond with ONLY one word: SIMPLE or COMPLEX

Question: {question}
Answer:""",
                "stream": False,
                "options": {"temperature": 0, "num_predict": 5}
            },
            timeout=10
        )
        result = resp.json().get("response", "SIMPLE").strip().upper()
        return result if result in ("SIMPLE", "COMPLEX") else "SIMPLE"
    except Exception:
        return classify_complexity_openai(question)


def classify_complexity_openai(question: str) -> str:
    """GPT-4o-mini fallback for complexity classification."""
    try:
        resp = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content":
                f"""Classify as SIMPLE or COMPLEX.
COMPLEX: specific drugs, interactions, comorbidities, opportunistic infections.
SIMPLE: general guidelines, definitions, monitoring frequency.
Return ONLY "SIMPLE" or "COMPLEX".
Question: {question}"""}],
            temperature=0,
            max_tokens=5
        )
        result = resp.choices[0].message.content.strip().upper()
        return result if result in ("SIMPLE", "COMPLEX") else "SIMPLE"
    except Exception:
        return "SIMPLE"


# ── Language Detection ─────────────────────────────────────────────────────

def detect_language(text: str) -> str:
    """Detects the query language using gemma3:1b."""
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model": "gemma3:1b",
                "prompt": f"Detect language. Respond ONLY with: English, French, Spanish, or Other.\nText: {text[:200]}\nLanguage:",
                "stream": False,
                "options": {"temperature": 0, "num_predict": 5}
            },
            timeout=10
        )
        return resp.json().get("response", "English").strip()
    except Exception:
        return "English"


# ── Router ─────────────────────────────────────────────────────────────────

def route_query(question: str) -> dict:
    """
    Routes a query to the appropriate pipeline configuration.
    COMPLEX queries activate HyDE, GraphRAG, PathRAG and PPR.

    Returns:
        {
          "complexity"  : "SIMPLE" | "COMPLEX",
          "language"    : "English" | "Spanish" | ...,
          "use_hyde"    : bool,
          "use_graph"   : bool,
          "use_ppr"     : bool,
          "use_pathrag" : bool,
          "search_lang" : "en" | "es" | "both"
        }
    """
    complexity = classify_complexity_local(question)
    language   = detect_language(question)
    is_spanish = "Spanish" in language or "Español" in language

    return {
        "complexity"  : complexity,
        "language"    : language,
        "use_hyde"    : complexity == "COMPLEX",
        "use_graph"   : complexity == "COMPLEX",
        "use_ppr"     : complexity == "COMPLEX",
        "use_pathrag" : complexity == "COMPLEX",
        "search_lang" : "es" if is_spanish else "both",
    }