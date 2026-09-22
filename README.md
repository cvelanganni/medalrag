# MedalRAG

**Hybrid RAG chatbot for HIV clinical decision support.**  
MedalRAG retrieves passages from NIH/HHS and GESIDA guidelines, reasons over a biomedical knowledge graph, and generates structured, sourced responses in the language of the query (English, Spanish, French, or any other language).

Built during a research internship at the [MEDAL Lab](https://medal.ctb.upm.es), CTB-UPM, Madrid (April–August 2026).

---

## What it does

A clinician types a question about an HIV patient. MedalRAG:

1. **Guards** the input, rejects out-of-scope queries
2. **Retrieves** relevant passages from US (NIH/HHS) and Spanish (GESIDA) guidelines using hybrid BM25 + dense search with query expansion
3. **Explores** a biomedical knowledge graph (~2,500 nodes, ~5,000 edges depending on the guidelines used) via PathRAG and HippoRAG Personalized PageRank
4. **Re-ranks** passages with a cross-encoder
5. **Generates** a structured response (NIH/HHS vs GESIDA side by side) with page-level citations, using GPT-4o, Claude, or DeepSeek
6. **Replies in the query language**, if the clinician asks in French, the response is in French; in Spanish, it is in Spanish; and so on

**Benchmark results** (GPT-4o-mini evaluator, n=35):

| Metric | Score |
|---|---|
| Context Recall | **0.865** |
| Faithfulness | **0.722** |
| MIRAGE (HIV MCQs) | **80%** (16/20) |

---

## Architecture

```
                    ┌─────────────────────────────────┐
                    │           Chainlit UI           │
                    └────────────────┬────────────────┘
                                     │ question (any language)
                    ┌────────────────▼────────────────┐
                    │    Agent 1 — Clinical Guardrail │
                    │    (GPT-4o-mini, JSON mode)     │
                    └────────────────┬────────────────┘
                                     │
              ┌──────────────────────┼────────────────────┐
              │                      │                    │
   ┌──────────▼─────────┐ ┌──────────▼─────────┐ ┌────────▼────────┐
   │  Hybrid Search EN  │ │  Hybrid Search ES  │ │   GraphRAG      │
   │  BM25 + BGE-M3     │ │  BM25 + BGE-M3     │ │  PathRAG + PPR  │
   │  + Query Expansion │ │  + Query Expansion │ │  Neo4j / NX     │
   └──────────┬─────────┘ └──────────┬─────────┘ └─────────┬───────┘
              │                      │                     │
              └──────────────────────┼─────────────────────┘
                                     │
                    ┌────────────────▼────────────────┐
                    │  Cross-encoder Re-ranker        │
                    │  (ms-marco-MiniLM-L-6-v2)       │
                    └────────────────┬────────────────┘
                                     │
                    ┌────────────────▼────────────────┐
                    │    Agent 2 — Grounding Check    │
                    └────────────────┬────────────────┘
                                     │
                    ┌────────────────▼────────────────┐
                    │    LLM Generation               │
                    │    GPT-4o / Claude / DeepSeek   │
                    └────────────────┬────────────────┘
                                     │
                    ┌────────────────▼────────────────┐
                    │  Structured response + sources  │
                    │  NIH/HHS ↔ GESIDA + PDF pages   │
                    └─────────────────────────────────┘
```

---

## Repository structure

```
medalRAG/
├── app.py                      # Chainlit interface — main entry point
├── build_pipeline.py           # Indexing pipeline (Qdrant + Neo4j)
├── enrich_graph.py             # Optional: UMLS CUI enrichment
├── evaluate.py                 # Ablation study & RAGAs benchmark
├── evaluate_mirage.py          # MIRAGE HIV benchmark
│
├── core/
│   ├── graph/
│   │   ├── builder.py          # Triplet extraction, entity normalization
│   │   ├── pathrag.py          # PathRAG relational path pruning
│   │   ├── hipporag_ppr.py     # HippoRAG Personalized PageRank
│   │   └── neo4j_sync.py       # Sync NetworkX → Neo4j
│   ├── retrieval/
│   │   ├── hybrid_search.py    # BM25 + Dense + RRF fusion
│   │   ├── query_expansion.py  # Query expansion + RRF
│   │   ├── reranker.py         # Cross-encoder re-ranking
│   │   ├── router.py           # Query complexity classifier
│   │   └── graph_search.py     # Neo4j graph search helpers
│   └── ingestion/
│       ├── loader.py           # PDF loading, chunking, noise filtering
│       ├── ner_en.py           # NER for English (GPT-4o-mini)
│       ├── ner_es.py           # NER for Spanish (fallback to EN)
│       └── glossary_parser.py  # NIH HIV glossary parser (v2 perspective)
│
├── tests/                      # Pytest test suite (9 tests)
├── scripts/                    # Utility scripts
├── evaluations/                # Benchmark results (JSON + PNG)
├── lib/                        # vis-network for graph visualization
├── public/                     # Chainlit static assets
│
├── golden_dataset_v2.json      # Official evaluation dataset (35 Q&A pairs)
├── golden_dataset_expert.json  # Expert-annotated complex cases
├── golden_dataset_short/medium/long.json  # Sensitivity analysis datasets
│
├── docker-compose.yml          # Qdrant + Neo4j + Ollama + Infinity reranker
├── pyproject.toml              # Dependencies (uv)
├── .env.example                # Environment variables template
└── .python-version             # Python 3.13
```

---

## Prerequisites

| Tool | Version | Purpose |
|---|---|---|
| Python | 3.13 | Runtime |
| [uv](https://docs.astral.sh/uv/) | latest | Package manager |
| [Docker](https://www.docker.com/) | latest | Qdrant + Neo4j + Ollama + Infinity reranker |
| [Ollama](https://ollama.com/) | latest | Local BGE-M3 embeddings |
| OpenAI API key | — | Generation, NER, triplet extraction |

---

## Setup

### 1. Clone and install dependencies

```bash
git clone https://github.com/cvelanganni/medalrag.git
cd medalrag
uv sync
```

### 2. Configure environment

```bash
cp .env.example .env
```

Fill in `.env` — at minimum:
```
OPENAI_API_KEY=sk-...
NEO4J_PASSWORD=your_password
AUTH_PASSWORD=your_password
CHAINLIT_AUTH_SECRET=   # python -c "import secrets; print(secrets.token_hex(32))"
DATABASE_URL=postgresql://user:password@localhost:5432/medalrag
```

### 3. Start Docker services

```bash
docker compose up -d
```

This starts Qdrant on `localhost:6333`, Neo4j on `localhost:7474`,
Ollama on `localhost:11434`, and Infinity reranker on `localhost:7997`.
PostgreSQL must be set up separately and configured via DATABASE_URL in .env.

### 4. Pull the embedding model

```bash
ollama pull bge-m3
```

### 5. Download guidelines

Place PDF files in the following directories:

```
data/en/    → NIH/HHS guidelines  (https://clinicalinfo.hiv.gov/en/guidelines)
data/es/    → GESIDA guidelines   (https://gesida-seimc.org/guias-clinicas/)
```

The pipeline was built with these documents:

**English (NIH/HHS + WHO):**
- [Adult & Adolescent ARV Guidelines](https://clinicalinfo.hiv.gov/en/guidelines/adult-and-adolescent-arv)
- [Adult & Adolescent OI Guidelines](https://clinicalinfo.hiv.gov/en/guidelines/adult-and-adolescent-opportunistic-infection)
- [Pediatric ARV Guidelines](https://clinicalinfo.hiv.gov/en/guidelines/pediatric-arv)
- [Pediatric OI Guidelines](https://clinicalinfo.hiv.gov/en/guidelines/pediatric-opportunistic-infection)
- [Perinatal Guidelines](https://clinicalinfo.hiv.gov/en/guidelines/perinatal)
- [CDC NPEP Guidelines](https://stacks.cdc.gov/view/cdc/38856)
- [WHO Tuberculosis Guidelines](https://www.who.int/publications/i/item/9789240007048)
- [WHO Hepatitis B & C Guidelines](https://www.who.int/publications/i/item/9789241549981)
- [WHO Contraceptive Eligibility for Women with HIV](https://www.who.int/publications/i/item/9789241549172)
- [HIV Glossary EN](https://clinicalinfo.hiv.gov/sites/g/files/mnhszr391/files/glossary/Glossary-English_HIVinfo.pdf)
- [HIV Glossary ES](https://clinicalinfo.hiv.gov/sites/g/files/mnhszr391/files/glossary/Glossary-Spanish_HIVinfo.pdf)
  
**Spanish (GESIDA):**
- [Guía TAR Adultos (2022)](https://gesida-seimc.org/guias-clinicas/)
- [Documento TB en VIH](https://gesida-seimc.org/guias-clinicas/)
- [Documento VIH y Embarazo](https://gesida-seimc.org/guias-clinicas/)
- [Documento Adherencia (2020)](https://gesida-seimc.org/guias-clinicas/)
- [Documento Riesgo Cardiovascular en VIH](https://gesida-seimc.org/guias-clinicas/)
- [Documento Profilaxis Postexposición VIH/VHB/VHC](https://gesida-seimc.org/guias-clinicas/)
- [Documento Salud Pública y VIH](https://gesida-seimc.org/guias-clinicas/)
- [Documento Alteraciones Neurológicas](https://gesida-seimc.org/guias-clinicas/)

---

## Build the pipeline

Run once to index all guidelines and build the knowledge graph.

```bash
# Full pipeline — Qdrant + graph (~$1 in API costs for LLM contextualization)
uv run build_pipeline.py

# Graph only (skip Qdrant re-indexing)
uv run build_pipeline.py --skip-qdrant

# Qdrant only (skip graph building)
uv run build_pipeline.py --skip-graph

# Disable LLM contextualization (faster, cheaper, slightly lower quality)
uv run build_pipeline.py --no-contextual
```

Expected output after a full build:
```
✓ Build complete!
  EN chunks  : ~9800  (varies with guidelines used)
  ES chunks  : ~1400  (varies with guidelines used)
  Graph nodes: ~2500  (varies with guidelines used)
  Graph edges: ~5000  (varies with guidelines used)
```

---

## Run the app

```bash
uv run chainlit run app.py
```

Open `http://localhost:8000` and log in with the credentials set in `.env`.

### Patient commands

```
/new_patient Sofia | 32 | 450 | 85000 | HIV+ naive HBsAg+ eGFR 58
/select
/delete_patient
```

### Pipeline settings

All pipeline components can be toggled from the UI settings panel:

| Toggle | Description |
|---|---|
| LLM Engine | GPT-4o, Claude Sonnet, DeepSeek |
| Hybrid BM25 + Dense | Hybrid search with RRF fusion |
| Query Expansion | 3 reformulations per query via GPT-4o-mini |
| GraphRAG + Neo4j | Knowledge graph retrieval |
| PathRAG Pruning | Relational path filtering |
| HippoRAG PPR | Personalized PageRank for multi-hop reasoning |
| Cross-Encoder Reranker | Re-ranking with ms-marco-MiniLM |
| GESIDA Guidelines | Spanish guidelines in retrieval |
| HyDE | Hypothetical Document Embedding |

### Docker services

The `docker-compose.yml` starts four services:

| Service | Container | Port | Purpose |
|---|---|---|---|
| Qdrant | `qdrant_lmph` | `6333` | Vector database for chunk storage and semantic search |
| Ollama | `ollama_lmph` | `11434` | Local model server (BGE-M3 embeddings, gemma3:1b router) |
| Neo4j | `neo4j_lmph` | `7474` (HTTP), `7687` (Bolt) | Knowledge graph storage and visualization |
| Infinity | `infinity_reranker` | `7997` | Cross-encoder re-ranking (BAAI/bge-reranker-v2-m3) |

All services bind to `127.0.0.1` only and persist data in Docker volumes. After `docker compose up -d`:

```bash
# Pull embedding model into Ollama
docker exec ollama_lmph ollama pull bge-m3

# Neo4j browser (optional — interactive graph visualization)
# Open http://localhost:7474, login: neo4j / your NEO4J_PASSWORD
```

---

## Evaluation

### RAGAs ablation study

```bash
# Full ablation (v1 → v9), ~2h, ~$5 in API costs
uv run evaluate.py

# Quick mode (v1, v3, v8, v9 only)
uv run evaluate.py --quick

# Specific versions with a custom evaluator
uv run evaluate.py --versions v1_semantic_baseline v9_hybrid --evaluator gpt4o-mini

# Available evaluators: deepseek, gpt4o-mini, gpt4o, claude-haiku, claude-sonnet
```

### MIRAGE benchmark

```bash
uv run evaluate_mirage.py --dataset mirage_hiv_strict.json --max 20
```

### Run tests

```bash
uv run pytest tests/ -v
```

---

## Optional: UMLS graph enrichment

Enriches graph nodes with UMLS CUI codes and merges duplicates. Requires a free key from [https://uts.nlm.nih.gov](https://uts.nlm.nih.gov).

```bash
# Add to .env: UMLS_API_KEY=your_key
uv run enrich_graph.py

# Replace main graph with enriched version
cp graph_cache_umls.pkl graph_cache.pkl
```

---

## Implementation choices and critique

This section explains each major architectural decision, its motivation, and its limitations honestly.

### Hybrid BM25 + Dense search with RRF

The retrieval combines sparse BM25 and dense BGE-M3 embeddings fused via Reciprocal Rank Fusion (Cormack et al., 2009). The motivation is straightforward: BM25 excels at exact drug name matching (e.g. "Dolutegravir", "TAF/FTC") while dense embeddings capture semantic context ("preferred in renal impairment" without the exact term). For a medical domain where acronyms and precise drug names are critical, neither alone is sufficient.

**Limitation.** The BM25 index is rebuilt in memory at startup, which adds a few seconds of latency on the first query. For a production system, a persistent BM25 index (e.g. Elasticsearch) would be more appropriate.

### Query expansion

Three reformulations are generated via GPT-4o-mini, one using formal medical terminology, one focusing on pharmacological mechanisms, and one in clinical guideline style. Each reformulation searches independently and results are fused with RRF, giving the original query a 1.5× weight. This approach is directly inspired by the query expansion with prompting literature, and was one of the clearest contributors to Context Recall improvements in the ablation study.

**Limitation.** Query expansion adds ~0.3s and one LLM call per query. For very short questions ("What is CD4?") the reformulations add little value and slightly increase cost.

### HyDE (Hypothetical Document Embeddings)

For complex queries, the system generates a short hypothetical guideline passage before searching and averages its embedding with the query embedding. The idea, from Gao et al. (2022), [Precise Zero-Shot Dense Retrieval without Relevance Labels](https://arxiv.org/abs/2212.10496), is that "what a relevant passage would look like" is semantically closer to actual guideline passages than the raw clinical question. HyDE is activated only for queries classified as COMPLEX by the router.

**Limitation.** HyDE relies on GPT-4o-mini generating a plausible passage, which can introduce hallucinated dosages or drug names. In a safety-critical setting, this is a non-trivial risk. A domain-specific model fine-tuned on guideline-style text would reduce this risk.

### Knowledge graph and triplet extraction

Medical triplets (subject, relation, object) are extracted from each clinical chunk using GPT-4o-mini, then validated by PubMedBERT NER. The resulting graph (~2,500 nodes, ~5,000 edges depending on the guidelines used) is stored in NetworkX for computation and synced to Neo4j for interactive visualization. This approach is inspired by [MedGraphRAG (Wu et al., 2024)](https://arxiv.org/abs/2408.04187) and [MedRAG (Zhao et al., 2025)](https://arxiv.org/abs/2502.04413), which both demonstrate that knowledge graphs substantially improve reasoning on multi-step clinical questions such as drug-drug interactions or co-infection management.

**Limitation.** GPT-4o-mini sometimes extracts sentence fragments as entities ("patients with renal impairment") rather than clean concepts ("Renal impairment"). The `is_valid_entity` filter and `ENTITY_CANONICAL_MAP` mitigate this, but ~15% of extracted triplets are filtered out as noise. A fine-tuned NER model trained specifically on HIV terminology would improve precision.

### PathRAG for triplet pruning

Once raw triplets are retrieved from Neo4j, [PathRAG (Chen et al., 2025)](https://arxiv.org/abs/2502.14902) filters them to keep only triplets on shortest relational paths between seed entities. The intuition is that GraphRAG and LightRAG often return redundant neighborhood information that adds noise to the LLM context. PathRAG replaces this neighborhood expansion with a more surgical path-based selection, which reduces token consumption and improves response coherence.

**Limitation.** The implementation here is a simplified version of the full PathRAG approach: flow-based pruning from the paper is approximated by shortest-path filtering with a confidence threshold. The original method uses a more sophisticated max-flow algorithm that was not re-implemented here. Safety-critical relations (INTERACTS_WITH, CONTRAINDICATED_IN) are always preserved regardless of the path score, which is a deliberate design choice for a clinical system.

### HippoRAG 2 Personalized PageRank

PPR propagates probability mass from seed entities outward through the knowledge graph, surfacing multi-hop connections that direct retrieval misses. For example, a query about "DTG + TB co-infection" seeds "Dolutegravir" and "Rifampin" as nodes, and PPR can surface the intermediate entity "UGT1A1 induction" that links them. This directly implements the neurobiologically inspired approach from [Gutierrez et al. (2025)](https://arxiv.org/abs/2502.14802), which outperforms standard RAG on multi-hop QA tasks by +7% on associative memory benchmarks.

**Limitation.** The full HippoRAG 2 implementation includes dense-sparse integration (passage nodes in the graph), triple filtering via recognition memory, and query-to-triple linking. MedalRAG implements a simplified version: only phrase nodes are used as seeds, and triple filtering is handled by PathRAG rather than a dedicated LLM filter step. The gains in the ablation study (v4 → v5: CR +0.003) suggest the simplified version still adds signal, but the gains are modest and not statistically significant at n=35.

### Cross-encoder re-ranking

Retrieved chunks are re-ranked by a cross-encoder (ms-marco-MiniLM-L-6-v2) that scores (query, chunk) pairs jointly, rather than comparing embeddings independently. Cross-encoder re-ranking has been a standard practice since [Nogueira & Cho (2019)](https://arxiv.org/abs/1901.04085) and consistently improves precision at the cost of ~50ms latency for 15 chunks.

**Limitation.** The ms-marco model is trained on web search data, not on medical text. A cross-encoder fine-tuned on HIV clinical question-passage pairs would likely improve ranking quality. This is flagged as a v2 improvement.

### Multilingual retrieval (EN + ES)

GESIDA guidelines are retrieved in Spanish alongside NIH/HHS in English. The BGE-M3 embedding model supports 100+ languages in a single embedding space, which enables cross-lingual semantic search without translation. The response is always generated in the language of the query, which is detected by the router and enforced in the system prompt.

**Limitation.** The NER pipeline for Spanish falls back to the English PubMedBERT model. The BSC-TeMU RoBERTa model (PlanTL-GOB-ES/bsc-bio-ehr-es-pharmaconer), which was intended for Spanish biomedical NER, was not installed in this version. This means triplets extracted from Spanish chunks rely on an English model that may miss Spanish-specific medical terminology variants.

### Evaluation design

The ablation study uses a golden dataset of 35 question-reference pairs (simple and complex, EN and ES), evaluated by three LLM evaluators (GPT-4o-mini, Claude Haiku, DeepSeek) using [RAGAs (Es et al., 2023)](https://arxiv.org/abs/2309.15217). MIRAGE (HIV-specific MCQs) provides a complementary clinical utility score.

**Limitation.** n=35 is a small evaluation set. Statistical tests (paired t-test, Wilcoxon, bootstrap CI) consistently show that gains between pipeline versions (v1 → v9: CR +0.020) are not statistically significant at p < 0.05. The non-determinism of LLM evaluators adds further variance, the same pipeline scores CR=0.865 with GPT-4o-mini, CR=0.752 with Claude Haiku, and CR=0.709 with DeepSeek. These evaluation variance issues are inherent to LLM-as-judge frameworks and are not specific to MedalRAG. A larger dataset and clinician-based evaluation would provide more reliable conclusions.

---

## Tech stack

| Component | Technology |
|---|---|
| Interface | Chainlit + PostgreSQL |
| Vector database | Qdrant |
| Knowledge graph | NetworkX (PPR/PathRAG) + Neo4j (visualization) |
| Embeddings | BGE-M3 (Ollama, local) |
| LLM generation | GPT-4o / Claude Sonnet / DeepSeek |
| NER | GPT-4o-mini + PubMedBERT (optional) |
| Re-ranker | ms-marco-MiniLM-L-6-v2 |
| Evaluation | RAGAs + MIRAGE |
| Package manager | uv |

---

## Perspectives

- **DrugBank integration** — structured pharmacological data for precise drug-drug interaction queries
- **Structured PDF parser** — better extraction of dosing tables and recommendation grids from guidelines
- **Spanish NER** — integrate BSC-TeMU RoBERTa (PlanTL-GOB-ES/bsc-bio-ehr-es-pharmaconer) for native Spanish biomedical entity extraction
- **Clinical validation** — formal evaluation by HIV-specialist physicians (the most meaningful evaluator)
- **Local deployment** — replace proprietary APIs with open-source models (Llama, Gemma) for GDPR-compliant use in clinical settings
- **Extend to other diseases** — the pipeline is disease-agnostic; hepatitis B/C, tuberculosis, or oncology guidelines could be integrated with minimal changes

---

## References

| Paper | Link |
|---|---|
| Gutierrez et al. (2025) — HippoRAG 2: From RAG to Memory | [arXiv:2502.14802](https://arxiv.org/abs/2502.14802) |
| Zhao et al. (2025) — MedRAG: KG-elicited Reasoning for Healthcare Copilot | [arXiv:2502.04413](https://arxiv.org/abs/2502.04413) |
| Chen et al. (2025) — PathRAG: Pruning Graph-based RAG with Relational Paths | [arXiv:2502.14902](https://arxiv.org/abs/2502.14902) |
| Es et al. (2023) — RAGAs: Automated Evaluation of RAG | [arXiv:2309.15217](https://arxiv.org/abs/2309.15217) |
| Gao et al. (2022) — Precise Zero-Shot Dense Retrieval without Relevance Labels (HyDE) | [arXiv:2212.10496](https://arxiv.org/abs/2212.10496) |
| Wu et al. (2024) — Medical Graph RAG: Towards Safe Medical LLM via GraphRAG | [arXiv:2408.04187](https://arxiv.org/abs/2408.04187) |
| Nogueira & Cho (2019) — Passage Re-ranking with BERT | [arXiv:1901.04085](https://arxiv.org/abs/1901.04085) |
| Cormack et al. (2009) — Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods | [ACM SIGIR](https://dl.acm.org/doi/10.1145/1571941.1572114) |

---

## Acknowledgements

This project was developed during a 4-month research internship at the MEDAL Lab, Centro de Tecnología Biomédica, Universidad Politécnica de Madrid (April–August 2026).

Special thanks to:

- **Ernestina Menasalvas Ruiz**, research supervisor and lab director, for her guidance and scientific direction throughout the internship
- **Borja Jordán de Urríes Ruiz**, for his technical feedback and support during the development
- **Víctor Rodríguez Melgar**, for his contributions to implementation reviews and progress discussions
- **David Gómez Ortiz**, for his valuable advice and discussions throughout the project
- The entire **MEDAL Lab team**, for welcoming me into the lab and providing a stimulating research environment

---

*Built at MEDAL Lab (Medical Data Analytics Laboratory), Centro de Tecnología Biomédica, Universidad Politécnica de Madrid.*  
*Supervised by Ernestina Menasalvas Ruiz.*