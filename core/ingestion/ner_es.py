# core/ingestion/ner_es.py
from core.ingestion.ner_en import extract_entities_en

def extract_entities_es(text: str, backend: str = "bsc_temu") -> list:
    """Fallback vers GPT-4o-mini EN pour les textes ES."""
    return extract_entities_en(text)