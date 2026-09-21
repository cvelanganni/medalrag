"""
loader.py — PDF loader and chunker for HIV clinical guidelines.

Detects tables to choose the best loading strategy,
filters noise, and flags clinical chunks before indexing.
"""

import os
import re
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pdfplumber


# ── Noise / Clinical Signal Lists ─────────────────────────────────────────

NOISE_SIGNALS = [
    "et al.", "doi:", "available at: http", "available at: https",
    "copyright", "acknowledgment", "conflict of interest",
    "table of contents", "https://www.", "http://www.",
    "pubmed.ncbi", "ncbi.nlm.nih.gov",
    "j infect dis", "clin infect dis", "lancet",
    "j antimicrob chemother", "am j obstet gynecol",
    "new engl j med", "ann intern med",
    "references\n", "bibliography", "medline", "embase", "cochrane",
]

CLINICAL_SIGNALS_EN = [
    "mg once daily", "mg twice daily", "first-line", "preferred regimen",
    "cells/mm3", "copies/ml", "is recommended", "should be initiated",
    "contraindicated", "drug interaction", "prophylaxis with",
    "treatment of choice", "antiretroviral", "cd4 count",
    "viral suppression", "viral load", "initiat", "discontinu",
    "dose of", "dosed at", "is indicated", "should be used",
    "alternative regimen", "switch to", "insti", "nnrti", "nrti", "pi/r",
]

CLINICAL_SIGNALS_ES = [
    "mg una vez al día", "mg dos veces al día", "de primera línea",
    "régimen preferido", "células/mm3", "copias/ml",
    "se recomienda", "debe iniciarse", "contraindicado",
    "interacción farmacológica", "profilaxis con", "tratamiento de elección",
    "antirretroviral", "recuento de cd4", "supresión viral", "carga viral",
    "iniciar", "suspender", "dosis de", "dosificado",
    "inhibidor de integrasa", "inhibidor de la proteasa",
    "régimen alternativo", "cambiar a",
]


# ── Helpers ────────────────────────────────────────────────────────────────

def has_tables(pdf_path: str) -> bool:
    """Returns True if the PDF contains any detectable tables."""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            return any(page.extract_tables() for page in pdf.pages)
    except Exception:
        return False


def smart_load(pdf_path: str) -> list:
    """
    Uses UnstructuredPDFLoader for PDFs with tables (better structure preservation),
    falls back to PyPDFLoader otherwise.
    """
    if has_tables(pdf_path):
        try:
            docs = UnstructuredPDFLoader(pdf_path, mode="elements", strategy="fast").load()
            if docs:
                return docs
        except Exception:
            pass
    return PyPDFLoader(pdf_path).load()


def is_noise(text: str) -> bool:
    """Returns True if the chunk looks like bibliographic noise or is too short."""
    text_lower = text.lower()
    if len(text.strip().split()) < 20:
        return True
    if any(s in text_lower for s in NOISE_SIGNALS):
        return True
    if text_lower.count("http") >= 2 or text_lower.count("et al") >= 2:
        return True
    if re.match(r'^[\d\s\.\,\;\:\-\(\)]+$', text.strip()):
        return True
    return False


def is_clinical(text: str, lang: str = "en") -> bool:
    """Returns True if the chunk contains clinical recommendation signals."""
    text_lower = text.lower()
    signals    = CLINICAL_SIGNALS_EN if lang == "en" else CLINICAL_SIGNALS_ES
    return any(s in text_lower for s in signals)


# ── Main ───────────────────────────────────────────────────────────────────

def load_and_chunk(
    data_dir     : str,
    lang         : str,
    chunk_size   : int = 1800,
    chunk_overlap: int = 300,
) -> list:
    """
    Loads all PDFs from a directory, splits into chunks,
    filters noise, and returns chunks with clinical metadata.

    Returns:
        list of {"text", "filename", "page", "lang", "is_clinical"}
    """
    pdf_files = list(Path(data_dir).glob("*.pdf"))
    if not pdf_files:
        print(f"  No PDF found in {data_dir}")
        return []

    print(f"  Loading {len(pdf_files)} PDFs [{lang.upper()}]...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )

    all_chunks, noise_count = [], 0

    for pdf_path in pdf_files:
        try:
            docs   = smart_load(str(pdf_path))
            chunks = splitter.split_documents(docs)
            for chunk in chunks:
                text = chunk.page_content
                if is_noise(text):
                    noise_count += 1
                    continue
                all_chunks.append({
                    "text"       : text,
                    "filename"   : pdf_path.name,
                    "page"       : chunk.metadata.get("page", "?"),
                    "lang"       : lang,
                    "is_clinical": is_clinical(text, lang),
                })
        except Exception as e:
            print(f"  Error loading {pdf_path.name}: {e}")

    clinical = sum(1 for c in all_chunks if c["is_clinical"])
    print(f"  {len(all_chunks)} chunks kept "
          f"({clinical} clinical, "
          f"{len(all_chunks) - clinical} uncertain, "
          f"{noise_count} noise filtered)")

    return all_chunks