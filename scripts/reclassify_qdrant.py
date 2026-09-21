# reclassify_qdrant.py
"""
Reclassifies Qdrant chunks more intelligently:
  - noise WITH clinical keywords → uncertain
  - noise WITHOUT clinical keywords → deleted
  - keeps all clinical and uncertain chunks
"""
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Filter, FieldCondition, MatchValue, PointIdsList
)
load_dotenv()

q = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))

# Keywords that indicate clinical content
CLINICAL_KEYWORDS = [
    # Drugs
    "dolutegravir", "dtg", "bictegravir", "raltegravir",
    "tenofovir", "taf", "tdf", "emtricitabine", "ftc",
    "lamivudine", "3tc", "abacavir", "abc", "efavirenz",
    "darunavir", "lopinavir", "ritonavir", "atazanavir",
    "rilpivirine", "nevirapine", "zidovudine", "zdv",
    "cabotegravir", "lenacapavir", "fostemsavir",
    # Conditions
    "hiv", "aids", "tuberculosis", "hepatitis", "hbv", "hcv",
    "opportunistic", "pneumocystis", "toxoplasma", "cryptococcal",
    "mycobacterium", "cytomegalovirus", "candida",
    # Clinical concepts
    "antiretroviral", "art ", "arv", "virologic", "viral load",
    "cd4", "resistance", "mutation", "prophylaxis", "prep", "pep",
    "pregnancy", "renal", "egfr", "contraindicated", "preferred",
    "dose", "regimen", "treatment", "therapy", "monitoring",
    # Spanish
    "vih", "antirretroviral", "embarazo", "dosis", "tratamiento",
    "carga viral", "profilaxis", "resistencia",
]

def is_clinical_content(text: str) -> bool:
    """Returns True if text contains clinical HIV keywords."""
    text_lower = text.lower()
    return any(kw in text_lower for kw in CLINICAL_KEYWORDS)


for col in ["medical_docs_en", "medical_docs_es"]:
    print(f"\n=== Processing {col} ===")
    before = q.count(col).count

    # Get all noise chunks
    noise_points = []
    offset = None

    while True:
        results, next_offset = q.scroll(
            collection_name = col,
            limit           = 100,
            offset          = offset,
            with_payload    = True,
            with_vectors    = False,
            scroll_filter   = Filter(must=[
                FieldCondition(
                    key   = "section_type",
                    match = MatchValue(value="noise")
                )
            ])
        )
        noise_points.extend(results)
        if next_offset is None:
            break
        offset = next_offset

    print(f"  Noise chunks found: {len(noise_points)}")

    # Separate clinical noise from real noise
    clinical_noise_ids = []
    real_noise_ids     = []

    for p in noise_points:
        text = p.payload.get("original_text",
                             p.payload.get("text", ""))
        if is_clinical_content(text):
            clinical_noise_ids.append(p.id)
        else:
            real_noise_ids.append(p.id)

    print(f"  Clinical noise (→ uncertain) : {len(clinical_noise_ids)}")
    print(f"  Real noise (→ deleted)       : {len(real_noise_ids)}")

    # 1. Reclassify clinical noise → uncertain
    if clinical_noise_ids:
        # Update section_type in batches of 100
        for i in range(0, len(clinical_noise_ids), 100):
            batch = clinical_noise_ids[i:i+100]
            q.set_payload(
                collection_name = col,
                payload         = {"section_type": "uncertain"},
                points          = batch,
            )
        print(f"  ✓ Reclassified {len(clinical_noise_ids)} → uncertain")

    # 2. Delete real noise
    if real_noise_ids:
        q.delete(
            collection_name = col,
            points_selector = PointIdsList(points=real_noise_ids)
        )
        print(f"  ✓ Deleted {len(real_noise_ids)} real noise chunks")

    after = q.count(col).count
    print(f"  Result: {before} → {after} chunks (-{before-after})")

print("\nDone!")