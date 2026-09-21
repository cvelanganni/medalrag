"""
Glossary Parser — parse les glossaires NIH bilingues EN/ES
pour construire une table de correspondance terminologique.
Utilisé par entity_normalization et contradiction_detector.
"""

import re
import fitz  # pymupdf
from pathlib import Path


def parse_glossary(pdf_path: str, lang: str = "en") -> dict:
    """
    Parse un glossaire NIH HIV et retourne un dict :
    {
      "Antiretroviral Therapy": {
        "synonyms"    : ["ART", "HAART", "cART"],
        "definition"  : "The daily use of...",
        "related"     : ["Drug Class", "HIV Regimen"],
        "lang"        : "en"
      },
      ...
    }
    """

    if lang == "en":
        syn_prefix  = "Synonym(s):"
        rel_prefix  = "Related Term(s):"
        see_prefix  = "See:"
    else:
        syn_prefix  = "Sinónimo(s):"
        rel_prefix  = "Término(s) relacionado(s):"
        see_prefix  = "VEA:"

    doc    = fitz.open(pdf_path)
    terms  = {}
    current_term = None
    current_def  = []

    def save_current():
        if current_term and current_term not in ("A", "B", "C", "D",
            "E", "F", "G", "H", "I", "J", "K", "L", "M", "N",
            "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X",
            "Y", "Z"):
            terms[current_term] = terms.get(current_term, {
                "synonyms"  : [],
                "definition": "",
                "related"   : [],
                "lang"      : lang,
            })
            if current_def:
                terms[current_term]["definition"] = " ".join(current_def).strip()

    for page in doc:
        lines = page.get_text().split("\n")
        i = 0
        while i < len(lines):
            line = lines[i].strip()

            if not line or line.isdigit():
                i += 1
                continue

            # Lettre de section (A, B, C...)
            if re.match(r'^[A-Z]$', line):
                i += 1
                continue

            # Synonymes
            if line.startswith(syn_prefix):
                if current_term:
                    synonyms = line[len(syn_prefix):].strip()
                    for s in synonyms.split(","):
                        s = s.strip()
                        if s:
                            terms.setdefault(current_term, {
                                "synonyms": [], "definition": "",
                                "related": [], "lang": lang
                            })["synonyms"].append(s)
                i += 1
                continue

            # Termes reliés
            if line.startswith(rel_prefix):
                if current_term:
                    related = line[len(rel_prefix):].strip()
                    for r in related.split(","):
                        r = r.strip()
                        if r:
                            terms.setdefault(current_term, {
                                "synonyms": [], "definition": "",
                                "related": [], "lang": lang
                            })["related"].append(r)
                i += 1
                continue

            # VEA / See (renvoi)
            if line.startswith(see_prefix):
                i += 1
                continue

            # Détecte un nouveau terme (ligne avec majuscule initiale,
            # pas trop longue, pas une phrase)
            is_term = (
                len(line) < 80
                and not line.endswith(".")
                and not line.startswith("-")
                and re.match(r'^[A-ZÁÉÍÓÚÑÜ]', line)
                and not any(line.startswith(p) for p in [
                    syn_prefix, rel_prefix, see_prefix,
                    "The ", "A ", "An ", "This ", "These ",
                    "See ", "VEA ",
                ])
            )

            if is_term:
                save_current()
                current_term = line
                current_def  = []
            else:
                if current_term:
                    current_def.append(line)

            i += 1

    save_current()
    doc.close()

    print(f"  Glossary [{lang.upper()}] : {len(terms)} terms parsed")
    return terms


def build_bilingual_mapping(
    glossary_en: dict,
    glossary_es: dict
) -> dict:
    """
    Construit une table de correspondance EN↔ES à partir
    des deux glossaires. Retourne :
    {
      "Antiretroviral Therapy": "Terapia antirretroviral",
      "Viral Load": "Carga viral",
      ...
    }
    """
    mapping = {}

    # Inverse le glossaire ES : terme ES → terme EN
    es_to_en = {}
    for term_es, data_es in glossary_es.items():
        for syn in data_es.get("synonyms", []):
            es_to_en[syn.lower()] = term_es

    # Pour chaque terme EN, cherche son équivalent ES
    for term_en, data_en in glossary_en.items():
        term_en_lower = term_en.lower()

        # Cherche directement
        for term_es in glossary_es.keys():
            if term_en_lower in term_es.lower() or term_es.lower() in term_en_lower:
                mapping[term_en] = term_es
                break

        # Cherche via synonymes EN dans le glossaire ES
        for syn_en in data_en.get("synonyms", []):
            syn_lower = syn_en.lower()
            if syn_lower in es_to_en:
                mapping[term_en] = es_to_en[syn_lower]
                break

    print(f"  Bilingual mapping : {len(mapping)} EN↔ES pairs found")
    return mapping


def load_glossaries(glossary_dir: str) -> tuple:
    """
    Charge les deux glossaires et retourne :
    (glossary_en, glossary_es, bilingual_mapping)
    """
    en_path = Path(glossary_dir) / "Glossary-English_HIVinfo.pdf"
    es_path = Path(glossary_dir) / "Glossary-Spanish_HIVinfo.pdf"

    print("Loading glossaries...")
    glossary_en = parse_glossary(str(en_path), lang="en") if en_path.exists() else {}
    glossary_es = parse_glossary(str(es_path), lang="es") if es_path.exists() else {}
    mapping     = build_bilingual_mapping(glossary_en, glossary_es)

    return glossary_en, glossary_es, mapping