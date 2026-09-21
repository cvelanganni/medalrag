"""
ner_en.py — Biomedical named entity recognition for English texts.
Primary backend: GPT-4o-mini.
Local fallback: gemma3:1b via Ollama.
"""

import re
import json
import requests
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

openai_client = OpenAI()
OLLAMA_URL    = "http://localhost:11434/api/generate"

MEDICAL_ENTITY_TYPES = [
    "drug", "condition", "lab_value",
    "organism", "procedure", "anatomy"
]


def extract_entities_openai(text: str) -> list:
    """Extracts medical entities using GPT-4o-mini (accurate, multilingual)."""
    try:
        resp = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": f"""Extract all medical entities from this HIV guideline text.

For each entity return:
- text       : exact string as it appears
- type       : one of {MEDICAL_ENTITY_TYPES}
- normalized : canonical NIH/HHS name

Return ONLY a valid JSON list. If none found: []
Example:
[
  {{"text": "DTG", "type": "drug", "normalized": "Dolutegravir"}},
  {{"text": "CD4 count", "type": "lab_value", "normalized": "CD4 count"}}
]

Text: {text[:1500]}"""}],
            temperature=0,
            max_tokens=400,
        )
        content = resp.choices[0].message.content.strip()
        match   = re.search(r'\[.*\]', content, re.DOTALL)
        return json.loads(match.group()) if match else []
    except Exception as e:
        print(f"  NER EN error: {e}")
        return []


def extract_entities_local(text: str) -> list:
    """Extracts medical entities using gemma3:1b via Ollama (free, local, fast)."""
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model": "gemma3:1b",
                "prompt": f"""Extract medical entities from this HIV text.
Return ONLY a JSON list with fields: text, type, normalized.
Types: drug, condition, lab_value, organism, procedure, anatomy.
Text: {text[:1000]}
JSON:""",
                "stream": False,
            },
            timeout=30,
        )
        content = resp.json().get("response", "")
        match   = re.search(r'\[.*\]', content, re.DOTALL)
        return json.loads(match.group()) if match else []
    except Exception as e:
        print(f"  NER EN local error: {e}")
        return []


def extract_entities_en(text: str, use_local: bool = False) -> list:
    """
    Main entry point — selects NER backend based on config.

    Args:
        text      : text to analyze
        use_local : True → gemma3:1b (Ollama), False → GPT-4o-mini
    """
    if use_local:
        return extract_entities_local(text)
    return extract_entities_openai(text)