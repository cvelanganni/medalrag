"""
Graph Builder — builds the NetworkX knowledge graph
from triplets extracted from EN + ES HIV guidelines.

NetworkX = in-memory graph used for PPR and PathRAG.
Neo4j    = used for visualization only.
"""

import os
import json
import re
import pickle
import networkx as nx
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
openai_client    = OpenAI()
GRAPH_CACHE_PATH = "graph_cache.pkl"

MEDICAL_RELATIONS = [
    "TREATS", "PREVENTS", "DOSE", "FREQUENCY",
    "CONTRAINDICATED_IN", "INTERACTS_WITH",
    "REQUIRES_ADJUSTMENT", "PREFERRED_FOR",
    "ALTERNATIVE_FOR", "INDICATED_WHEN",
    "DISCONTINUED_WHEN", "MONITORING_REQUIRED",
    "PART_OF", "COMBINED_WITH", "AVOIDED_IN",
    "EQUIVALENT_TO",
]

# Fragments that indicate an entity is actually a sentence fragment, not a concept
BAD_STARTS = [
    "initiating", "based on", "no history", "every ",
    "2 to", "prior to", "after ", "before ", "during ",
    "patients with", "patients on", "patients who",
    "children with", "children aged", "infants ",
    "weighing ", "aged ", "virologically ",
    "should be", "must be", "may be",
]


# ── Entity Canonical Map ───────────────────────────────────────────────────
# Maps all known variants (abbreviations, brand names, Spanish forms)
# to a single canonical entity name used throughout the graph.

ENTITY_CANONICAL_MAP = {
    # Dolutegravir
    "dtg": "Dolutegravir", "dolutégravir": "Dolutegravir",
    "dolutegravir": "Dolutegravir", "tivicay": "Dolutegravir",
    # Bictegravir
    "bic": "Bictegravir", "bictegravir": "Bictegravir",
    # Raltegravir
    "ral": "Raltegravir", "raltegravir": "Raltegravir", "isentress": "Raltegravir",
    # Elvitegravir
    "evg": "Elvitegravir", "elvitegravir": "Elvitegravir",
    # Cabotegravir
    "cab": "Cabotegravir", "cabotegravir": "Cabotegravir",
    # TAF
    "taf": "TAF", "tenofovir alafenamide": "TAF",
    "tenofovir alafenamida": "TAF", "vemlidy": "TAF",
    # TDF
    "tdf": "TDF", "tenofovir disoproxil fumarate": "TDF",
    "tenofovir disoproxil": "TDF", "tenofovir": "TDF", "viread": "TDF",
    # Emtricitabine
    "ftc": "Emtricitabine", "emtricitabine": "Emtricitabine",
    "emtricitabina": "Emtricitabine",
    # Lamivudine
    "3tc": "Lamivudine", "lamivudine": "Lamivudine",
    "lamivudina": "Lamivudine", "epivir": "Lamivudine",
    # Abacavir
    "abc": "Abacavir", "abacavir": "Abacavir", "ziagen": "Abacavir",
    # Zidovudine
    "zdv": "Zidovudine", "azt": "Zidovudine", "zidovudine": "Zidovudine",
    "zidovudina": "Zidovudine", "retrovir": "Zidovudine",
    # Efavirenz
    "efv": "Efavirenz", "efavirenz": "Efavirenz", "efavirencia": "Efavirenz",
    "sustiva": "Efavirenz", "stocrin": "Efavirenz",
    # Nevirapine
    "nvp": "Nevirapine", "nevirapine": "Nevirapine",
    "nevirapina": "Nevirapine", "viramune": "Nevirapine",
    # Rilpivirine
    "rpv": "Rilpivirine", "rilpivirine": "Rilpivirine",
    "rilpivirina": "Rilpivirine", "edurant": "Rilpivirine",
    # Etravirine
    "etr": "Etravirine", "etravirine": "Etravirine", "intelence": "Etravirine",
    # Darunavir
    "drv": "Darunavir", "darunavir": "Darunavir", "prezista": "Darunavir",
    # Lopinavir
    "lpv": "Lopinavir", "lopinavir": "Lopinavir",
    "lopinavir/r": "Lopinavir/ritonavir", "lpv/r": "Lopinavir/ritonavir",
    "kaletra": "Lopinavir/ritonavir",
    # Atazanavir
    "atv": "Atazanavir", "atazanavir": "Atazanavir", "reyataz": "Atazanavir",
    # Ritonavir
    "rtv": "Ritonavir", "ritonavir": "Ritonavir", "norvir": "Ritonavir",
    # Cobicistat
    "cobi": "Cobicistat", "cobicistat": "Cobicistat", "/c": "Cobicistat",
    # Maraviroc
    "mrv": "Maraviroc", "maraviroc": "Maraviroc", "selzentry": "Maraviroc",
    # Enfuvirtide
    "t-20": "Enfuvirtide", "enfuvirtide": "Enfuvirtide", "fuzeon": "Enfuvirtide",
    # Rifampin
    "rifampin": "Rifampin", "rifampicin": "Rifampin", "rifampicina": "Rifampin",
    "rifamycin": "Rifampin", "rifamycins": "Rifampin", "rifadin": "Rifampin",
    "increased with rifampin": "Rifampin",
    # Rifabutin
    "rifabutin": "Rifabutin", "mycobutin": "Rifabutin",
    # TMP-SMX
    "tmp-smx": "TMP-SMX", "trimethoprim-sulfamethoxazole": "TMP-SMX",
    "co-trimoxazole": "TMP-SMX", "cotrimoxazole": "TMP-SMX",
    "bactrim": "TMP-SMX", "septra": "TMP-SMX",
    # HIV
    "hiv": "HIV infection", "hiv infection": "HIV infection",
    "hiv-1": "HIV infection", "hiv-2": "HIV infection",
    "vih": "HIV infection", "infección por vih": "HIV infection",
    "infeccion por vih": "HIV infection",
    # ART
    "art": "ART", "arv": "ART", "antiretroviral therapy": "ART",
    "antiretroviral treatment": "ART", "haart": "ART",
    "tar": "ART", "tratamiento antirretroviral": "ART",
    # Pregnancy
    "pregnancy": "Pregnancy", "pregnant": "Pregnancy",
    "pregnant women": "Pregnancy", "gestation": "Pregnancy",
    "embarazo": "Pregnancy", "gestación": "Pregnancy",
    # CD4
    "cd4": "CD4", "cd4 count": "CD4", "cd4+ count": "CD4",
    "cd4+ t-lymphocyte": "CD4", "cd4 cell count": "CD4",
    "linfocitos cd4": "CD4",
    # Viral load
    "vl": "Viral load", "viral load": "Viral load",
    "hiv rna": "Viral load", "hiv-1 rna": "Viral load",
    "carga viral": "Viral load",
    # Tuberculosis
    "tb": "Tuberculosis", "tuberculosis": "Tuberculosis",
    "tuberculose": "Tuberculosis", "tubercolosis": "Tuberculosis",
    # HBV
    "hbv": "HBV", "hepatitis b": "HBV", "hepatitis b virus": "HBV",
    "hepatitis b viral": "HBV", "vhb": "HBV",
    # HCV
    "hcv": "HCV", "hepatitis c": "HCV",
    "hepatitis c virus": "HCV", "vhc": "HCV",
    # PrEP / PEP
    "prep": "PrEP", "pre-exposure prophylaxis": "PrEP",
    "profilaxis preexposición": "PrEP",
    "pep": "PEP", "npep": "PEP", "post-exposure prophylaxis": "PEP",
    # Opportunistic infections
    "pcp": "Pneumocystis pneumonia", "pneumocystis": "Pneumocystis pneumonia",
    "pneumocystis jirovecii pneumonia": "Pneumocystis pneumonia",
    "mac": "Mycobacterium avium complex",
    "mycobacterium avium": "Mycobacterium avium complex",
    "cmv": "Cytomegalovirus", "cytomegalovirus": "Cytomegalovirus",
    "cryptococcus": "Cryptococcal meningitis",
    "cryptococcal meningitis": "Cryptococcal meningitis",
    "toxoplasma": "Toxoplasmosis", "toxoplasmosis": "Toxoplasmosis",
    # Fixed-dose combinations
    "bic/ftc/taf": "BIC/FTC/TAF",
    "bictegravir/emtricitabine/taf": "BIC/FTC/TAF",
    "biktarvy": "BIC/FTC/TAF",
    "dtg/3tc": "DTG/3TC", "dolutegravir/lamivudine": "DTG/3TC", "dovato": "DTG/3TC",
    "taf/ftc": "TAF/FTC", "tdf/ftc": "TDF/FTC", "truvada": "TDF/FTC",
    "abc/3tc": "ABC/3TC", "epzicom": "ABC/3TC",
    "zdv/3tc": "ZDV/3TC", "combivir": "ZDV/3TC",
    "drv/r": "DRV/ritonavir", "drv/c": "DRV/cobicistat", "prezcobi": "DRV/cobicistat",
    # Drug classes
    "insti": "INSTI", "integrase inhibitor": "INSTI",
    "integrase strand transfer inhibitor": "INSTI",
    "nnrti": "NNRTI", "nrti": "NRTI",
    "pi": "Protease inhibitor", "protease inhibitor": "Protease inhibitor",
    # Clinical parameters
    "renal impairment": "Renal impairment",
    "renal insufficiency": "Renal impairment",
    "kidney disease": "Renal impairment",
    "chronic kidney disease": "Renal impairment",
    "egfr": "eGFR", "hla-b*5701": "HLA-B*5701", "hla-b 5701": "HLA-B*5701",
    "dofetilide": "Dofetilide", "metformin": "Metformin",
    "iron": "Iron supplements", "iron supplements": "Iron supplements",
    "calcium": "Calcium supplements",
}


# ── Entity normalization ───────────────────────────────────────────────────

def normalize_entity(text: str) -> str:
    """Maps a raw entity string to its canonical form using ENTITY_CANONICAL_MAP."""
    if not text:
        return text
    text_clean = text.strip()
    text_lower = text_clean.lower()
    if text_lower in ENTITY_CANONICAL_MAP:
        return ENTITY_CANONICAL_MAP[text_lower]
    for key, val in ENTITY_CANONICAL_MAP.items():
        if len(key) > 3 and key in text_lower:
            return val
    return text_clean


def is_valid_entity(text: str) -> bool:
    """Returns True if the string looks like a short medical concept, not a sentence fragment."""
    if not text or not text.strip():
        return False
    text = text.strip()
    if len(text.split()) > 6:
        return False
    if any(c in text for c in ['>', '<', '≥', '≤', '=', '%']):
        return False
    if any(text.lower().startswith(b) for b in BAD_STARTS):
        return False
    if len(text) < 2:
        return False
    return True


def normalize_relation(rel: str) -> str | None:
    """Maps a raw relation string to one of the allowed MEDICAL_RELATIONS, or None if unmappable."""
    rel = rel.strip().upper()
    if rel in MEDICAL_RELATIONS:
        return rel
    rel_map = {
        "TREAT": "TREATS", "PREVENT": "PREVENTS",
        "INTERACT": "INTERACTS_WITH", "PREFERRED": "PREFERRED_FOR",
        "AVOID": "AVOIDED_IN", "COMBINE": "COMBINED_WITH",
        "MONITOR": "MONITORING_REQUIRED", "ALTERNATIVE": "ALTERNATIVE_FOR",
        "CONTRAINDIC": "CONTRAINDICATED_IN", "ADJUST": "REQUIRES_ADJUSTMENT",
        "INDICATE": "INDICATED_WHEN", "DISCONTINUE": "DISCONTINUED_WHEN",
        "EQUIVALENT": "EQUIVALENT_TO", "PART": "PART_OF", "FREQUENC": "FREQUENCY",
    }
    for key, val in rel_map.items():
        if key in rel:
            return val
    return None


# ── Triplet extraction ─────────────────────────────────────────────────────

def extract_triplets_from_chunk(text: str, source: str, lang: str = "en") -> list:
    """
    Extracts (subject, relation, object) triplets from a clinical text chunk.
    Uses GPT-4o-mini for both EN and ES. Entities are normalized before return.
    """
    relations_str = ", ".join(MEDICAL_RELATIONS)

    prompt = f"""Extract medical knowledge triplets from this HIV guideline text.

STRICT RULES for entities:
- Subject and Object MUST be SHORT medical concepts (1-4 words MAX)
- Use canonical drug names: "Dolutegravir" not "initiating dolutegravir therapy"
- NO full sentences as entities, NO numerical thresholds
- Normalize: "DTG" → "Dolutegravir", "rifampin" → "Rifampin"

ALLOWED RELATIONS (use ONLY these exact names): {relations_str}

Good examples:
  {{"subject": "Dolutegravir", "relation": "INTERACTS_WITH", "object": "Rifampin"}}
  {{"subject": "TAF", "relation": "PREFERRED_FOR", "object": "Renal impairment"}}

Bad examples (DO NOT DO THIS):
  {{"subject": "initiating ART during pregnancy", ...}}
  {{"subject": "HIV RNA levels >200 copies/mL", ...}}

Text ({lang.upper()}):
{text[:1500]}

Return ONLY a valid JSON list, no explanation:
[{{"subject": "...", "relation": "RELATION_NAME", "object": "..."}}]
If no valid triplets found, return: []"""

    try:
        resp = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=600,
        )
        content = resp.choices[0].message.content.strip()

        match = re.search(r'\[.*\]', content, re.DOTALL)
        if not match:
            return []
        triplets = json.loads(match.group())
        if not isinstance(triplets, list):
            return []

        valid = []
        for t in triplets:
            if not isinstance(t, dict):
                continue
            subj = normalize_entity(str(t.get("subject", "")).strip())
            obj  = normalize_entity(str(t.get("object",  "")).strip())
            rel  = normalize_relation(str(t.get("relation", "")).strip())

            if not is_valid_entity(subj) or not is_valid_entity(obj):
                continue
            if subj.lower() == obj.lower():
                continue
            if rel is None:
                continue

            valid.append({
                "subject":  subj,
                "relation": rel,
                "object":   obj,
                "source":   source,
                "lang":     lang,
            })
        return valid

    except Exception as e:
        print(f"  Triplet extraction error: {e}")
        return []


# ── Graph construction ─────────────────────────────────────────────────────

def build_networkx_graph(triplets: list) -> nx.DiGraph:
    """Builds a directed NetworkX graph from a list of triplets."""
    G = nx.DiGraph()
    for t in triplets:
        subj = normalize_entity(t.get("subject", "").strip())
        obj  = normalize_entity(t.get("object",  "").strip())
        rel  = t.get("relation", "").strip().upper()
        if not subj or not obj or rel not in MEDICAL_RELATIONS:
            continue
        if subj.lower() == obj.lower():
            continue
        lang = t.get("lang", "en")
        G.add_node(subj, lang=lang)
        G.add_node(obj,  lang=lang)
        G.add_edge(subj, obj,
                   relation=rel,
                   confidence=t.get("confidence", 0.85),
                   source=t.get("source", "unknown"),
                   lang=lang,
                   weight=t.get("confidence", 0.85))
    return G


def merge_graphs(G1: nx.DiGraph, G2: nx.DiGraph) -> nx.DiGraph:
    """
    Merges two graphs (e.g. EN + ES) by union of nodes and edges.
    When an edge exists in both, keeps the one with higher confidence.
    """
    G_merged = G1.copy()
    for node, data in G2.nodes(data=True):
        if node not in G_merged:
            G_merged.add_node(node, **data)
    for u, v, data in G2.edges(data=True):
        if G_merged.has_edge(u, v):
            if data.get("confidence", 0) > G_merged[u][v].get("confidence", 0):
                G_merged[u][v].update(data)
        else:
            G_merged.add_edge(u, v, **data)
    print(f"  Merged graph: {G_merged.number_of_nodes()} nodes, {G_merged.number_of_edges()} edges")
    return G_merged


# ── Persistence ────────────────────────────────────────────────────────────

def save_graph(G: nx.DiGraph, path: str = GRAPH_CACHE_PATH):
    with open(path, "wb") as f:
        pickle.dump(G, f)
    print(f"  Graph saved: {path} ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")


def load_graph(path: str = GRAPH_CACHE_PATH) -> nx.DiGraph | None:
    if os.path.exists(path):
        with open(path, "rb") as f:
            G = pickle.load(f)
        print(f"  Graph loaded: {path} ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")
        return G
    return None