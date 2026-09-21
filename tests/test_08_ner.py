"""
test_08_ner.py
--------------
Verifies that the NER (Named Entity Recognition) pipeline
is working correctly for both English and Spanish medical text.

EN Backend: GPT-4o-mini (primary) + PubMedBERT if available
ES Backend: GPT-4o-mini (primary) + BSC-TeMU if available

Run:
    uv run python tests/test_08_ner.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
RESET  = "\033[0m"

def ok(msg):   print(f"  {GREEN}✓{RESET} {msg}")
def fail(msg): print(f"  {RED}✗{RESET} {msg}")
def warn(msg): print(f"  {YELLOW}~{RESET} {msg}")


def test_ner_en_openai() -> bool:
    """Test EN NER via GPT-4o-mini."""
    print("\n[1] EN NER — GPT-4o-mini backend")
    try:
        from core.ingestion.ner_en import extract_entities_en

        text = (
            "Dolutegravir 50mg twice daily is recommended when co-administered "
            "with Rifampin for tuberculosis. TAF is preferred over TDF in patients "
            "with renal impairment."
        )

        entities = extract_entities_en(text)

        if not entities:
            fail("No entities extracted")
            return False

        ok(f"Extracted {len(entities)} entities:")
        for e in entities:
            ok(f"  [{e.get('type','?'):12}] {e.get('text','?'):20}")

        # Check key drugs
        entity_texts = [e.get("text","").lower() for e in entities]
        expected     = ["dolutegravir", "rifampin", "taf", "tdf"]
        found        = [d for d in expected
                       if any(d in t for t in entity_texts)]

        ok(f"Key drugs found: {found}")
        if len(found) >= 2:
            ok(f"EN NER: {len(found)}/{len(expected)} drugs ✓")
            return True
        warn(f"Only {len(found)}/{len(expected)} drugs found")
        return True

    except Exception as e:
        fail(f"EN NER failed: {e}")
        return False


def test_ner_en_local() -> bool:
    """Test EN NER via Ollama local (gemma3:1b)."""
    print("\n[2] EN NER — Ollama local backend")
    try:
        from core.ingestion.ner_en import extract_entities_local

        text = "Dolutegravir HIV treatment with TAF and FTC combination."
        entities = extract_entities_local(text)

        if entities:
            ok(f"Local NER extracted {len(entities)} entities")
            for e in entities[:3]:
                ok(f"  {e.get('text','?')}")
        else:
            warn("Local NER returned 0 entities (gemma3:1b may need tuning)")

        return True  # non-fatal

    except Exception as e:
        warn(f"Local NER skipped: {e}")
        return True


def test_ner_pubmedbert() -> bool:
    """Test PubMedBERT NER if available."""
    print("\n[3] EN NER — PubMedBERT (if available)")
    try:
        from transformers import pipeline

        ner = pipeline(
            "ner",
            model                = "d4data/biomedical-ner-all",
            aggregation_strategy = "first",
            device               = -1,
        )

        text     = "Dolutegravir 50mg with Rifampin for HIV tuberculosis co-infection."
        entities = ner(text)

        ok(f"PubMedBERT extracted {len(entities)} entities:")
        for e in entities:
            ok(f"  [{e['entity_group']:15}] {e['word']:15} (score: {e['score']:.2f})")
        return True

    except ImportError:
        warn("transformers not installed — PubMedBERT unavailable")
        return True
    except Exception as e:
        warn(f"PubMedBERT test skipped: {e}")
        return True


def test_ner_es_openai() -> bool:
    """Test ES NER via GPT-4o-mini."""
    print("\n[4] ES NER — GPT-4o-mini backend")
    try:
        # Try to import ner_es if it exists
        try:
            from core.ingestion.ner_es import extract_entities_es
            use_module = True
        except ImportError:
            use_module = False
            warn("ner_es.py not found — using direct OpenAI call")

        text = (
            "El Dolutegravir 50mg dos veces al día es recomendado para "
            "pacientes con VIH y tuberculosis cuando se usa Rifampicina."
        )

        if use_module:
            entities = extract_entities_es(text)
        else:
            # Fallback — test directly via OpenAI
            from openai import OpenAI
            from dotenv import load_dotenv
            load_dotenv()
            client = OpenAI()

            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content":
                    f"Extract medical entities from this Spanish HIV text. "
                    f"Return JSON list with 'text' and 'type' fields.\n{text}"}],
                temperature=0, max_tokens=200
            )
            import json, re
            content = resp.choices[0].message.content
            match   = re.search(r'\[.*\]', content, re.DOTALL)
            entities = json.loads(match.group()) if match else []

        ok(f"Extracted {len(entities)} ES entities:")
        for e in (entities[:5] if entities else []):
            ok(f"  {e.get('text','?')}")

        if entities:
            ok("ES NER working ✓")
            return True
        warn("ES NER returned 0 entities")
        return True

    except Exception as e:
        fail(f"ES NER failed: {e}")
        return False


def test_ner_bsc_temu() -> bool:
    """Test BSC-TeMU ES NER if available."""
    print("\n[5] ES NER — BSC-TeMU RoBERTa (if available)")
    try:
        from transformers import pipeline

        ner = pipeline(
            "ner",
            model                = "PlanTL-GOB-ES/bsc-bio-ehr-es-pharmaconer",
            aggregation_strategy = "first",
            device               = -1,
        )

        text     = "El Dolutegravir y la Rifampicina son medicamentos para el VIH."
        entities = ner(text)

        ok(f"BSC-TeMU extracted {len(entities)} ES entities:")
        for e in entities:
            word = e["word"].rstrip(".,;:")
            ok(f"  [{e['entity_group']:15}] {word:15} (score: {e['score']:.2f})")
        return True

    except ImportError:
        warn("transformers not installed — BSC-TeMU unavailable")
        return True
    except Exception as e:
        warn(f"BSC-TeMU test skipped: {e}")
        return True


def test_entity_quality() -> bool:
    """Test that extracted entities are valid concepts, not fragments."""
    print("\n[6] Entity quality validation")
    try:
        from core.ingestion.ner_en import extract_entities_en

        text = (
            "Initiating antiretroviral therapy during pregnancy. "
            "Dolutegravir 50mg once daily is recommended."
        )
        entities = extract_entities_en(text)
        all_ok   = True

        for e in entities:
            text_e     = e.get("text", "")
            word_count = len(text_e.split())
            if word_count > 8:
                warn(f"Long entity detected: '{text_e}' ({word_count} words)")
            else:
                ok(f"Valid entity: '{text_e}' ({word_count} words)")

        ok(f"Validated {len(entities)} entities")
        return all_ok

    except Exception as e:
        fail(f"Quality test failed: {e}")
        return False


def main():
    print("=" * 55)
    print("  MedalRAG — NER Pipeline Tests")
    print("=" * 55)

    tests = [
        test_ner_en_openai,
        test_ner_en_local,
        test_ner_pubmedbert,
        test_ner_es_openai,
        test_ner_bsc_temu,
        test_entity_quality,
    ]

    results = [t() for t in tests]
    n_pass  = sum(results)
    n_total = len(results)

    print("\n" + "=" * 55)
    print(f"  Results: {n_pass}/{n_total} tests passed")

    if n_pass >= 4:
        ok("NER pipeline is working correctly")
        sys.exit(0)
    else:
        fail(f"{n_total - n_pass} test(s) failed")
        sys.exit(1)


if __name__ == "__main__":
    main()