# MedalRAG — HIV Clinical Decision Support Assistant

> A hybrid GraphRAG system for HIV/AIDS clinical guideline querying,
> combining NIH/HHS (EN) and GESIDA (ES) guidelines with knowledge
> graph reasoning, bilingual support, and clinical contradiction detection.

**CTB-UPM Madrid — Internship Project 2026**

---

## Overview

MedalRAG is a Retrieval-Augmented Generation (RAG) system designed to
assist physicians with HIV/AIDS clinical decision-making. It queries
official clinical guidelines in English (NIH/HHS) and Spanish (GESIDA)
and returns structured, source-cited recommendations.

### Key Features

- **Bilingual EN/ES** — queries and answers in English or Spanish
- **Hybrid search** — BM25 + dense semantic search via Qdrant
- **Knowledge Graph** — Neo4j graph with 7,000+ medical entities
- **GraphRAG pipeline** — PathRAG + PPR HippoRAG multi-hop reasoning
- **UMLS normalization** — entity deduplication via CUI identifiers
- **Cross-encoder reranker** — BGE/ms-marco reranking
- **Query expansion** — multiple reformulations via RRF fusion
- **Contradiction detection** — alerts when EN and ES guidelines differ
- **Multi-patient profiles** — persistent conversations per patient
- **LangSmith tracing** — optional pipeline visualization

### Architecture

```
PDF Guidelines (EN + ES)
        ↓
    Chunking + Section Classification
        ↓
   ┌────┴────┐
   ↓         ↓
Qdrant      Neo4j Knowledge Graph
(dense +    (PathRAG + PPR HippoRAG)
 sparse)         ↓ UMLS CUI normalization
   ↓
Hybrid Search + Query Expansion
        ↓
Cross-Encoder Reranking
        ↓
LLM Generation (GPT-4o / Claude / Ollama)
        ↓
Streamlit Interface
```

---

## Requirements

- **Docker Desktop** — for Qdrant, Neo4j, Ollama
- **Python 3.12+**
- **uv** package manager
- **~15GB disk space** — models + indexed data
- **OpenAI API key** — with ~$5 credit for initial indexing
- GPU recommended for Ollama inference (CPU works but slower)

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/your_lab/medalrag.git
cd medalrag
```

### 2. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and fill in at minimum:

```
OPENAI_API_KEY=sk-...   # required
```

### 3. Start Docker services

```bash
docker-compose up -d
```

Verify all services are running:

```bash
docker ps
# Expected: qdrant_lmph, neo4j_lmph, ollama_lmph
```

### 4. Install Python dependencies

```bash
pip install uv      # if not already installed
uv sync
```

### 5. Download Ollama models

```bash
# Embeddings model (required)
docker exec ollama_lmph ollama pull mxbai-embed-large

# Router model (required)
docker exec ollama_lmph ollama pull gemma3:1b

# Local LLM (optional — for offline generation)
docker exec ollama_lmph ollama pull gemma3:12b-it-qat
```

### 6. Add your guidelines

Place your PDF guidelines in the appropriate folders:

```
data/
├── en/          ← NIH/HHS English guidelines (PDF)
├── es/          ← GESIDA Spanish guidelines (PDF)
└── glossary/    ← NIH HIV glossary EN + ES (PDF)
```

> **Note:** Guidelines are not included in this repository due to
> copyright. Download them from:
>
> - **EN:** https://clinicalinfo.hiv.gov/en/guidelines
> - **ES:** https://www.gesida-seimc.org/guias-clinicas/

### 7. Build the pipeline (one-time, ~1 hour)

```bash
# Full pipeline (recommended)
uv run build_pipeline.py

# Without LLM contextualization (faster, ~20 min)
uv run build_pipeline.py --no-contextual

# Qdrant only — skip graph rebuild
uv run build_pipeline.py --skip-graph
```

### 8. Launch the interface

```bash
uv run streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## Optional Setup

### UMLS Entity Normalization

Improves graph quality by mapping entity variants to UMLS CUI identifiers
(e.g. "DTG", "Dolutegravir", "Tivicay" → CUI C3889366).

1. Create a free account at https://uts.nlm.nih.gov/uts/signup-login
2. Add your key to `.env`: `UMLS_API_KEY=your_key`
3. Run the enrichment script:

```bash
uv run enrich_graph.py
```

### LangSmith Tracing

Visualize pipeline calls and trace each query step-by-step.

1. Create a free account at https://smith.langchain.com
2. Add your key to `.env`: `LANGCHAIN_API_KEY=ls__...`
3. Set: `LANGCHAIN_TRACING_V2=true`

---

## Evaluation

Run the ablation study benchmark (RAGAs metrics):

```bash
# Quick mode — 4 versions, ~30 min
uv run evaluate.py --quick --sleep 15

# Full ablation — 10 versions, ~2 hours
uv run evaluate.py --sleep 15

# Specific versions only
uv run evaluate.py --versions v1_semantic_baseline v9_hybrid
```

Results are saved in `evaluations/` as JSON files and matplotlib figures.

---

## Project Structure

```
medalRAG/
├── app.py                      ← Streamlit interface
├── evaluate.py                 ← RAGAs benchmark + ablation study
├── build_pipeline.py           ← Data ingestion + indexing
├── enrich_graph.py             ← UMLS graph enrichment
├── generate_golden_dataset.py  ← Golden dataset generator
├── entity_normalization.py     ← Entity canonical names
├── .env.example                ← Environment template
├── docker-compose.yml          ← Docker services
├── pyproject.toml              ← Python dependencies
│
├── data/
│   ├── en/                     ← NIH/HHS guidelines (PDF)
│   ├── es/                     ← GESIDA guidelines (PDF)
│   └── glossary/               ← NIH HIV glossary EN + ES
│
├── core/
│   ├── ingestion/
│   │   ├── loader.py           ← PDF loading + chunking
│   │   ├── glossary_parser.py  ← Glossary parsing
│   │   ├── ner_en.py           ← English NER
│   │   ├── ner_es.py           ← Spanish NER (BETO/XLM-R)
│   │   └── umls_normalizer.py  ← UMLS normalization
│   ├── graph/
│   │   ├── builder.py          ← Triplet extraction
│   │   ├── neo4j_sync.py       ← Neo4j synchronization
│   │   ├── pathrag.py          ← PathRAG filtering
│   │   └── hipporag_ppr.py     ← Personalized PageRank
│   └── retrieval/
│       ├── router.py           ← SIMPLE/COMPLEX classifier
│       ├── vector_search.py    ← Qdrant search
│       ├── hybrid_search.py    ← BM25 + Dense + RRF
│       ├── query_expansion.py  ← Query reformulation + RRF
│       ├── reranker.py         ← Cross-encoder reranking
│       └── contradiction_detector.py
│
├── patients/                   ← Patient profiles (local, git-ignored)
├── conversations/              ← Conversation history (local, git-ignored)
└── evaluations/                ← Benchmark results + plots (git-ignored)
```

---

## Pipeline Components

| Component        | Implementation             | Purpose              |
| ---------------- | -------------------------- | -------------------- |
| Embedding        | mxbai-embed-large (Ollama) | Chunk vectorization  |
| Vector DB        | Qdrant (Docker)            | Semantic search      |
| Sparse search    | BM25 (rank-bm25)           | Lexical matching     |
| Reranker         | ms-marco-MiniLM-L-6-v2     | Top-K refinement     |
| Knowledge Graph  | Neo4j (Docker)             | Entity relations     |
| Graph algorithms | PathRAG + HippoRAG PPR     | Multi-hop reasoning  |
| Entity linking   | UMLS API                   | Deduplication        |
| LLM (generator)  | GPT-4o / Claude / Ollama   | Response generation  |
| LLM (router)     | gemma3:1b (Ollama)         | Query classification |
| Evaluator        | DeepSeek-chat              | RAGAs metrics        |
| Interface        | Streamlit                  | Clinical UI          |

---

## RAGAs Benchmark Results

Evaluated on 35 bilingual questions (EN + ES) across 10 pipeline versions:

| Version               | Context Recall | Faithfulness | Factual Correctness | Average |
| --------------------- | :------------: | :----------: | :-----------------: | :-----: |
| v1 Semantic baseline  |     0.550      |    0.695     |        0.258        |  0.501  |
| v3 +GraphRAG          |     0.675      |    0.716     |        0.275        |  0.555  |
| v6 +Cross-encoder     |     0.600      |    0.689     |        0.282        |  0.524  |
| v8 Full pipeline      |     0.675      |    0.711     |        0.261        |  0.549  |
| v9 +Hybrid BM25+Dense |     0.650      |    0.736     |        0.220        |  0.535  |

> **Note:** Faithfulness of 0.711 exceeds the SIGIR 2025 LiveRAG Challenge
> 3rd-place system (Faithfulness = 0.55), validating our approach on
> a challenging medical domain.

### Score Breakdown by Language (v9 Full Pipeline)

| Category          | Context Recall | Faithfulness | Factual Correctness |
| ----------------- | :------------: | :----------: | :-----------------: |
| English questions |     0.538      |    0.734     |        0.185        |
| Spanish questions |     0.857      |    0.740     |        0.283        |

---

## Supported Models

### Generator (LLM)

| Provider       | Models                                 |
| -------------- | -------------------------------------- |
| OpenAI         | gpt-4o, gpt-4o-mini                    |
| Anthropic      | claude-sonnet-4-5, claude-haiku-4-5    |
| Ollama (local) | gemma3:12b-it-qat, meditron, medllama2 |

### Embeddings (Ollama local)

| Model             | Dimensions | Purpose                        |
| ----------------- | ---------- | ------------------------------ |
| mxbai-embed-large | 1024       | Chunk + query vectorization    |
| gemma3:1b         | —          | Query routing (SIMPLE/COMPLEX) |

---

## Roadmap

Planned improvements for future versions:

- [ ] **BGE-M3** — multilingual dense + sparse embedding in a single model
- [ ] **BGE-Reranker-v2-m3** — multilingual cross-encoder reranker
- [ ] **MedCAT** — local UMLS entity linking (replaces API calls)
- [ ] **PubMedBERT NER** — specialized English medical NER
- [ ] **BSC-TeMU RoBERTa** — specialized Spanish medical NER
- [ ] **Neo4j CUI primary key** — 100% duplicate-free graph
- [ ] **MedGraphRAG L2** — PubMed papers as intermediate graph layer
- [ ] **Contextual Retrieval** — LLM-enriched chunk indexing

---

## Clinical Safety Notice

> ⚠️ **MedalRAG is a clinical decision support tool only.**
>
> All recommendations must be validated by a qualified physician.
> This system does not replace professional medical judgment.
> Guidelines cited are from official NIH/HHS and GESIDA sources
> but may not reflect the most recent updates.
>
> **Not intended for direct patient use.**

---

## References

- Guo et al. (2024) — _LightRAG: Simple and Fast Retrieval-Augmented Generation_
- Xiong et al. (2024) — _MedRAG: Benchmarking Retrieval-Augmented Generation for Biomedical NLP_
- Gutierrez et al. (2025) — _HippoRAG 2: From Resource to Omnivore_
- Chen et al. (2025) — _PathRAG: Pruning Graph-based RAG with Relational Paths_
- CTB-UPM (2026) — _Evaluating Spanish Medical NER: Encoder-only vs LLM approaches_
- Cofala & Xion (2025) — _RAGtifier: SIGIR LiveRAG 2025 Challenge_

---

## Sources

- LightRAG : https://arxiv.org/pdf/2410.05779
- PathRAG : https://arxiv.org/pdf/2502.14902
- HippoRAG 2 : https://arxiv.org/pdf/2502.14802
- HIV Guidelines : https://clinicalinfo.hiv.gov/en/guidelines-search?guidelines-search=HIV
- GeSIDA Guidelines : https://gesida-seimc.org/category/guias-clinicas/antirretroviral-historial/
- Entity Recognition (for spanish) : https://www.techscience.com/cmc/v87n3/66969
- RAGAs : https://arxiv.org/pdf/2309.15217
- MedRAG : https://arxiv.org/pdf/2502.04413
-

## License

This project was developed during an internship at
**CTB-UPM (Centro de Tecnología Biomédica), Universidad Politécnica de Madrid, Spain**.

For research and educational use only.

---

_Built with ❤️ at CTB-UPM Madrid — 2026_
