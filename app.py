"""
MedalRAG — Chainlit Interface
HIV Clinical Decision Support System
Multi-provider LLMs + RAG hybrid pipeline + Knowledge Graph
"""

import os
import json
import pickle
import uuid
import time
import re
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

import chainlit as cl
import chainlit.data as cl_data
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
from chainlit.input_widget import Select, Switch
from chainlit.types import ThreadDict

# Optional LangSmith tracing
if os.getenv("LANGSMITH_API_KEY"):
    os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGSMITH_TRACING", "true")
    os.environ["LANGCHAIN_API_KEY"]     = os.getenv("LANGSMITH_API_KEY")
    os.environ["LANGCHAIN_PROJECT"]     = os.getenv("LANGSMITH_PROJECT", "medalrag-hiv")
    os.environ["LANGCHAIN_ENDPOINT"]    = os.getenv("LANGSMITH_ENDPOINT", "https://eu.api.smith.langchain.com")

@cl.data_layer
def get_data_layer():
    return SQLAlchemyDataLayer(
        conninfo=os.getenv("DATABASE_URL"),
        ssl_require=False
    )

import httpx
import numpy as np
from openai import AsyncOpenAI
import anthropic

from core.graph.builder import normalize_entity as get_canonical_name
from core.graph.pathrag import pathrag_filter
from core.graph.hipporag_ppr import get_ppr_triplets
from core.retrieval.reranker import rerank_chunks
from core.retrieval.hybrid_search import hybrid_search_simple
from core.retrieval.query_expansion import expand_and_search_simple
from core.retrieval.router import route_query
from qdrant_client import QdrantClient
from neo4j import GraphDatabase


# ── Clients ────────────────────────────────────────────────────────────────

openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

deepseek_client = AsyncOpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY", ""),
    base_url="https://api.deepseek.com"
) if os.getenv("DEEPSEEK_API_KEY") else None

anthropic_client = anthropic.AsyncAnthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY", "")
) if os.getenv("ANTHROPIC_API_KEY") else None

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL", "http://localhost:6333"),
    api_key=os.getenv("QDRANT_API_KEY") or None,
)

neo4j_driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI", "bolt://localhost:7687"),
    auth=(
        os.getenv("NEO4J_USERNAME", "neo4j"),
        os.getenv("NEO4J_PASSWORD", "")
    ),
)

COLLECTION_EN = os.getenv("QDRANT_COLLECTION_EN", "medical_docs_en")
COLLECTION_ES = os.getenv("QDRANT_COLLECTION_ES", "medical_docs_es")
NEO4J_DB      = os.getenv("NEO4J_DATABASE", "neo4j")
PROFILES_FILE = "patient_profiles.json"
AUTH_USER     = os.getenv("AUTH_USER", "doctor")
AUTH_PASSWORD = os.getenv("AUTH_PASSWORD", "medalrag2026")

DEFAULT_CONFIG = {
    "model_provider": "OpenAI (gpt-4o)",
    "show_thinking": True,
    "hybrid": True, "expansion": True, "graph": True,
    "pathrag": True, "ppr": True, "reranker": True,
    "es": True, "hyde": True,
}


# ── Authentication ─────────────────────────────────────────────────────────

@cl.password_auth_callback
def auth(username: str, password: str):
    if username in [AUTH_USER, "medecin"] and password in [AUTH_PASSWORD, "medalrag2026"]:
        return cl.User(identifier=username, metadata={"role": "doctor"})
    return None


# ── Verification Agents ────────────────────────────────────────────────────

async def verify_query_agent(query: str, patient_ctx: str = "", recent_history: str = "") -> dict:
    """
    Agent 1 — Clinical guardrail.
    Checks if the query is HIV/AIDS related before running the pipeline.
    Returns {"is_clinical": bool, "confidence": float, "reason": str}.
    """
    system_prompt = """You are a strict Medical Triage & Clinical Safety Agent for MedalRAG (an HIV/AIDS clinical decision support system).

Determine if the input query is related to clinical medicine, healthcare decision-making, patient assessment, pharmacology, or HIV care.

CRITICAL INCLUSION RULES (Classify as is_clinical: true):
1. Any inquiry regarding antiretroviral therapy (ART/ARV), opportunistic infections, or co-infections (HBV, HCV, TB).
2. Patient assessment, lab tests (CD4, viral load, eGFR, liver enzymes, HLA-B*5701).
3. Reproductive health, pregnancy planning, contraception, vertical transmission risk, or teratogenicity in people living with HIV.
4. Follow-up clinical questions referencing a previous answer or the active patient's background.

CRITICAL EXCLUSIONS (Classify as is_clinical: false):
- Cooking, food recipes, baking.
- General programming/coding, gaming, sports, entertainment.
- Unrelated non-medical general chit-chat.

Respond ONLY with a JSON object:
{
  "is_clinical": true/false,
  "confidence": 0.0-1.0,
  "reason": "short explanation"
}"""

    user_payload = (
        f"Active Patient Record: {patient_ctx or 'None'}\n"
        f"Recent Conversation Summary: {recent_history or 'None'}\n"
        f"User Inquiry: \"{query}\""
    )

    try:
        res = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_payload}
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        return json.loads(res.choices[0].message.content)
    except Exception:
        return {"is_clinical": True, "confidence": 1.0, "reason": "Bypass on fallback"}


async def contextualize_query(query: str, patient_ctx: str, recent_history: str = "") -> str:
    """
    Rewrites short follow-up questions into standalone clinical search queries.
    Skips rewriting if the query is already long and explicit enough.
    """
    words = query.strip().split()
    if len(words) > 8 and any(k in query.lower() for k in ["hiv", "vih", "art", "arv", "hbv", "vhb", "cd4"]):
        return query
    if not patient_ctx and not recent_history:
        return query

    prompt = f"""You are a Clinical Query Contextualizer for an HIV Decision Support Assistant.
Rewrite the short follow-up inquiry into a concise standalone medical search query for HIV guideline retrieval.

Patient Context: {patient_ctx or 'None'}
Recent Context: {recent_history or 'None'}
User Inquiry: "{query}"

Output ONLY 1 concise search query in English focused on main clinical keywords (drugs, conditions, renal status)."""

    try:
        res = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        rewritten = res.choices[0].message.content.strip().strip('"')
        return rewritten if rewritten else query
    except Exception:
        return query


async def verify_documents_agent(query: str, retrieved_docs: list) -> dict:
    """
    Agent 2 — Grounding verifier.
    Checks if retrieved chunks are relevant enough before generating.
    Returns {"relevant": bool, "relevance_score": float, "reason": str}.
    """
    if not retrieved_docs:
        return {"relevant": False, "relevance_score": 0.0, "reason": "No passages retrieved"}

    sample_context = "\n---\n".join([
        doc.payload.get("original_text", doc.payload.get("text", ""))[:400]
        for doc in retrieved_docs[:4]
    ])

    system_prompt = """You are a Grounding Verification Agent for an HIV/AIDS clinical decision system.
Assess whether the retrieved guideline excerpts are TOPICALLY RELEVANT to the domain of the inquiry (e.g., HIV treatment, ARV regimens, opportunistic infections, HBV/HCV co-infections, renal dosing, reproductive health, or patient monitoring).

CRITICAL: The excerpts DO NOT need to contain all exact numbers (e.g. CD4 or eGFR values) or the full answer by themselves. They only need to provide relevant medical/pharmacological background on the topic.

Respond ONLY with a JSON object:
{
  "relevant": true/false,
  "relevance_score": 0.0-1.0,
  "reason": "brief explanation"
}"""

    try:
        res = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": f"Query: {query}\n\nRetrieved Guidelines Excerpts:\n{sample_context}"}
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        return json.loads(res.choices[0].message.content)
    except Exception:
        return {"relevant": True, "relevance_score": 1.0, "reason": "Bypass on fallback"}


# ── Graph Cache ────────────────────────────────────────────────────────────

_gc = {}

def init_graph():
    """Loads the NetworkX graph and entity embeddings into memory (once per session)."""
    global _gc
    if _gc:
        return _gc

    g = None
    if os.path.exists("graph_cache.pkl"):
        with open("graph_cache.pkl", "rb") as f:
            g = pickle.load(f)

    names, mat = [], None
    if os.path.exists("entity_embeddings_cache.pkl"):
        with open("entity_embeddings_cache.pkl", "rb") as f:
            cache = pickle.load(f)
        names = list(cache.keys())
        m     = np.array([cache[n] for n in names])
        mat   = m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-8)

    # Build name lookup from Neo4j or fallback to graph nodes
    nl = {}
    try:
        with neo4j_driver.session(database=NEO4J_DB) as s:
            nl = {r["name"].lower(): r["name"]
                  for r in s.run("MATCH (n:MedicalEntity) RETURN n.name AS name")}
    except Exception:
        pass
    if not nl and g:
        nl = {n.lower(): n for n in g.nodes()}

    _gc = {"ppr": g, "names": names, "mat": mat, "nl": nl}
    return _gc


# ── Patient Profiles ───────────────────────────────────────────────────────

def load_profiles() -> dict:
    if os.path.exists(PROFILES_FILE):
        with open(PROFILES_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_profiles(p: dict):
    with open(PROFILES_FILE, "w", encoding="utf-8") as f:
        json.dump(p, f, indent=2, ensure_ascii=False)

async def set_chat_title(patient_name: str):
    try:
        thread_id = cl.context.session.thread_id
        if thread_id and cl_data._data_layer:
            await cl_data._data_layer.update_thread(thread_id, name=f"Patient: {patient_name}")
    except Exception:
        pass


# ── Pipeline Helpers ───────────────────────────────────────────────────────

async def embed_async(text: str) -> list:
    """Async BGE-M3 embedding via Ollama."""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(
                f"{OLLAMA_URL}/api/embed",
                json={"model": "bge-m3", "input": text[:8000]}
            )
            return r.json().get("embeddings", [[0.] * 1024])[0]
    except Exception:
        return [0.] * 1024


async def resolve_async(entity: str, gc: dict):
    """Resolves an entity string to its canonical graph name using cosine similarity (threshold 0.82)."""
    names, mat, nl = gc["names"], gc["mat"], gc["nl"]
    if not names or mat is None:
        return None
    c     = get_canonical_name(entity.strip())
    cl_low = c.lower()
    if cl_low in nl:
        return nl[cl_low]
    try:
        e = np.array(await embed_async(c))
        n = np.linalg.norm(e)
        if n > 0:
            s  = mat @ (e / n)
            bi = int(np.argmax(s))
            if float(s[bi]) >= 0.82:
                return names[bi]
    except Exception:
        pass
    return None


def extract_entities(q: str) -> list:
    """Extracts seed entities from the query using the canonical entity map."""
    from core.graph.builder import ENTITY_CANONICAL_MAP
    ql = q.lower()
    return list(set(v for k, v in ENTITY_CANONICAL_MAP.items() if len(k) >= 3 and k.lower() in ql))


def get_triplets(resolved_entities: list) -> list:
    """Fetches graph triplets from Neo4j for the given seed entities (PathRAG-style)."""
    if not resolved_entities:
        return []
    seen, trips = set(), []
    try:
        with neo4j_driver.session(database=NEO4J_DB) as s:
            for rec in s.run("""
                MATCH (n:MedicalEntity)-[r]->(m:MedicalEntity)
                WHERE n.name IN $names
                RETURN n.name AS subject, type(r) AS relation, m.name AS object LIMIT 80
            """, names=resolved_entities):
                t = dict(rec)
                k = (t["subject"], t["relation"], t["object"])
                if k not in seen:
                    seen.add(k)
                    trips.append(t)
    except Exception:
        pass
    return trips


def fmt_triplets(trips: list) -> str:
    return "\n".join(f"- {t['subject']} --[{t['relation']}]--> {t['object']}" for t in trips) if trips else ""


def extract_and_clean_followup(resp: str):
    """
    Extracts follow-up questions (lines starting with '?') from the LLM response
    and returns them separately so they can be rendered as action buttons.
    """
    followups, clean_lines = [], []
    for line in resp.splitlines():
        s = line.strip()
        if s.startswith("?") and len(s) > 5:
            c = s.lstrip("?").strip()
            if c and c[0].isdigit():
                c = c[1:].strip().lstrip(".").strip()
            if len(c) > 5:
                followups.append(c)
        else:
            clean_lines.append(line)
    return followups[:3], "\n".join(clean_lines).strip()


def build_graph_html(trips: list):
    """Renders the triplets as an interactive pyvis graph (returns temp HTML path)."""
    try:
        from pyvis.network import Network
        if not trips:
            return None
        import tempfile
        net = Network(height="580px", width="100%", bgcolor="#1A1B1E", font_color="white", directed=True)
        net.force_atlas_2based()
        color_map = {
            "Dolutegravir": "#4E79A7", "HIV infection": "#E15759",
            "ART": "#F28E2B", "CD4": "#76B7B2", "default": "#BAB0AC"
        }
        nodes_added = set()
        for t in trips:
            for node in [t["subject"], t["object"]]:
                if node not in nodes_added:
                    net.add_node(node, label=node, title=node,
                                 color=color_map.get(node, color_map["default"]), size=18)
                    nodes_added.add(node)
            net.add_edge(t["subject"], t["object"], label=t["relation"], color="#555555")
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode="w", encoding="utf-8")
        net.save_graph(tmp.name)
        return tmp.name
    except Exception:
        return None


# ── Pipeline Settings Widget ───────────────────────────────────────────────

async def send_pipeline_settings():
    settings = await cl.ChatSettings([
        Select(
            id="model_provider",
            label="LLM Engine",
            values=[
                "OpenAI (gpt-4o)",
                "Anthropic (claude-3-5-sonnet)",
                "DeepSeek (deepseek-reasoner)",
                "DeepSeek (deepseek-chat)"
            ],
            initial_index=0,
        ),
        Switch(id="show_thinking", label="Show Real-time Reasoning / CoT", initial=True),
        Switch(id="hybrid",        label="Hybrid BM25 + Dense Search",      initial=True),
        Switch(id="expansion",     label="Query Expansion & Decomposition",  initial=True),
        Switch(id="graph",         label="GraphRAG + Neo4j Subgraphs",       initial=True),
        Switch(id="pathrag",       label="PathRAG Relation Pruning",         initial=True),
        Switch(id="ppr",           label="HippoRAG Personalized PageRank",   initial=True),
        Switch(id="reranker",      label="Cross-Encoder Reranker",           initial=True),
        Switch(id="es",            label="GESIDA Guidelines (Spanish)",      initial=True),
        Switch(id="hyde",          label="HyDE (Hypothetical Embeddings)",   initial=True),
    ]).send()
    return dict(settings)


# ── Chainlit Lifecycle ─────────────────────────────────────────────────────

@cl.on_chat_start
async def on_start():
    init_graph()
    config = await send_pipeline_settings()
    cl.user_session.set("config",      config)
    cl.user_session.set("profiles",    load_profiles())
    cl.user_session.set("active_pid",  None)
    cl.user_session.set("chat_history", [])

    await cl.Message(content=(
        "## MedalRAG — HIV Clinical Decision Support Assistant\n\n"
        "Commands: `/new_patient Name | Age | CD4 | VL | ARV` · `/select` · `/delete_patient`"
    )).send()

    profiles = cl.user_session.get("profiles", {})
    if profiles:
        actions = [
            cl.Action(name="select_patient", value=pid,
                      label=f"{p['name']} — CD4: {p['cd4_count']}",
                      payload={"pid": pid})
            for pid, p in profiles.items()
        ]
        actions.append(cl.Action(name="new_patient",          value="new", label="New Patient Record",    payload={}))
        actions.append(cl.Action(name="delete_patient_prompt", value="del", label="Delete Patient Record", payload={}))
        await cl.Message(content="**Select an active patient file or create a new profile:**", actions=actions).send()
    else:
        await cl.Message(content=(
            "No patient profiles registered yet.\n"
            "```text\n/new_patient Sofia Martinez | 32 | 450 | 85000 | HIV+ naive HBsAg+ eGFR 58\n```"
        )).send()


@cl.on_chat_resume
async def on_resume(thread: ThreadDict):
    init_graph()
    config = await send_pipeline_settings()
    cl.user_session.set("config",     config)
    cl.user_session.set("profiles",   load_profiles())
    cl.user_session.set("active_pid", None)

    # Restore last conversation turns from persisted thread
    chat_history = []
    for message in thread.get("steps", []):
        if message.get("type") == "user_message":
            output = message.get("output", "")
            if output and not output.startswith("/"):
                chat_history.append({"role": "user", "content": output})
        elif message.get("type") == "assistant_message":
            output = message.get("output", "")
            if output and len(output) > 100:
                chat_history.append({"role": "assistant", "content": output[:1500]})
    cl.user_session.set("chat_history", chat_history)


@cl.on_settings_update
async def on_settings(settings: dict):
    cl.user_session.set("config", settings)
    model = settings.get("model_provider", "Default")
    await cl.Message(content=f"**Configuration updated:** Active LLM engine: `{model}`").send()


# ── Patient Action Callbacks ───────────────────────────────────────────────

@cl.action_callback("select_patient")
async def select_patient(action: cl.Action):
    pid = action.payload.get("pid") or action.value
    profiles = cl.user_session.get("profiles", {})
    if pid in profiles:
        cl.user_session.set("active_pid", pid)
        p = profiles[pid]
        await set_chat_title(p["name"])
        await cl.Message(content=(
            f"### Selected Patient: {p['name']}\n"
            f"| Parameter | Value |\n|---|---|\n"
            f"| Age | {p['age']} y/o |\n"
            f"| CD4 Count | {p['cd4_count']} cells/mm³ |\n"
            f"| Viral Load | {p.get('viral_load', 'Unknown')} |\n"
            f"| ARV History | {p.get('arv_history', 'Unknown')} |\n\n"
            f"Ask your clinical question regarding this patient below ↓"
        )).send()


@cl.action_callback("new_patient")
async def new_patient_btn(action: cl.Action):
    await cl.Message(content=(
        "```text\n/new_patient Name | Age | CD4 | Viral Load | ARV History\n```\n"
        "Example: `/new_patient Sofia | 32 | 450 | 85000 | HIV+ naive HBsAg+ eGFR 58`"
    )).send()


@cl.action_callback("delete_patient_prompt")
async def delete_patient_btn(action: cl.Action):
    await _cmd_delete("")


@cl.action_callback("execute_delete")
async def execute_delete(action: cl.Action):
    pid = action.payload.get("pid") or action.value
    profiles = cl.user_session.get("profiles", {})
    if pid in profiles:
        deleted_name = profiles[pid]["name"]
        del profiles[pid]
        save_profiles(profiles)
        cl.user_session.set("profiles", profiles)
        if cl.user_session.get("active_pid") == pid:
            cl.user_session.set("active_pid", None)
        await cl.Message(content=f"Patient record **{deleted_name}** has been deleted.").send()


@cl.action_callback("followup")
async def on_followup(action: cl.Action):
    q = action.payload.get("question") or action.value
    await on_message(cl.Message(content=q))


# ── Main Message Handler ───────────────────────────────────────────────────

@cl.on_message
async def on_message(message: cl.Message):
    content = message.content.strip()

    # Handle slash commands
    if content.startswith("/new_patient"):
        await _cmd_new_patient(content)
        return
    if content.startswith("/delete_patient"):
        await _cmd_delete(content.replace("/delete_patient", "").strip())
        return
    if content.startswith("/select"):
        await _cmd_select()
        return

    profiles     = cl.user_session.get("profiles", {})
    active_pid   = cl.user_session.get("active_pid")
    config       = cl.user_session.get("config", DEFAULT_CONFIG.copy())
    chat_history = cl.user_session.get("chat_history", [])
    patient      = profiles.get(active_pid, {}) if active_pid else {}

    patient_ctx = ""
    if patient:
        patient_ctx = (
            f"Patient: {patient['name']}, Age: {patient['age']}, "
            f"CD4: {patient['cd4_count']} cells/mm³, "
            f"VL: {patient.get('viral_load', 'Unknown')}, "
            f"ARV: {patient.get('arv_history', 'Unknown')}"
        )

    recent_history = ""
    if chat_history:
        recent_history = " | ".join([f"{m['role']}: {m['content'][:140]}" for m in chat_history[-3:]])

    gc = init_graph()

    model_selected = config.get("model_provider", "OpenAI (gpt-4o)")
    use_hybrid     = config.get("hybrid",    True)
    use_expand     = config.get("expansion", True)
    use_graph      = config.get("graph",     True)
    use_pathrag    = config.get("pathrag",   True)
    use_ppr        = config.get("ppr",       True)
    use_reranker   = config.get("reranker",  True)
    use_es         = config.get("es",        True)
    use_hyde       = config.get("hyde",      True)

    elements        = []
    element_buttons = []
    start_time      = time.time()

    async with cl.Step(name="Thinking...", type="llm") as thought_step:

        # Step 1 — Guardrail agent
        await thought_step.stream_token("Agent 1: Clinical Triage & Safety Verification...\n")
        guard_result = await verify_query_agent(content, patient_ctx=patient_ctx, recent_history=recent_history)
        is_clinical  = guard_result.get("is_clinical", True)
        confidence   = guard_result.get("confidence", 1.0)
        guard_reason = guard_result.get("reason", "")
        await thought_step.stream_token(
            f"- Status: {'Clinical Query Accepted' if is_clinical else 'Non-Clinical Query Blocked'} (Confidence: {confidence:.2f})\n"
            f"- Assessment: {guard_reason}\n\n"
        )

        if not is_clinical:
            elapsed = max(1, round(time.time() - start_time))
            thought_step.name = f"Thought for {elapsed}s"
            await thought_step.update()
            patient_name = patient.get("name", "the patient")
            await cl.Message(content=(
                "**Out-of-Scope Query Detected**\n\n"
                "MedalRAG is specialized in **HIV/AIDS care, opportunistic infections, and antiretroviral regimens**.\n\n"
                f"*{guard_reason}*\n\n"
                f"Please submit a clinical query regarding **{patient_name}**."
            )).send()
            return

        # Step 2 — Routing & query contextualization
        route    = route_query(content)
        cplx     = route.get("complexity", "STANDARD")
        do_hyde  = use_hyde and route.get("use_hyde", cplx == "COMPLEX")
        await thought_step.stream_token(f"Clinical Routing: Complexity: `{cplx}`\n\n")

        search_query = await contextualize_query(content, patient_ctx, recent_history)
        if search_query != content:
            await thought_step.stream_token(f"Contextualized Search Query: `{search_query}`\n\n")

        # Step 3 — Retrieve NIH/HHS guidelines (EN)
        await thought_step.stream_token("Searching US Clinical Guidelines (NIH/HHS)...\n")
        if use_expand:
            chunks_en = await cl.make_async(expand_and_search_simple)(
                query=search_query, collection=COLLECTION_EN, limit=15, use_hyde=do_hyde, lang="en"
            )
        elif use_hybrid:
            chunks_en = await cl.make_async(hybrid_search_simple)(
                query=search_query, collection=COLLECTION_EN, limit=15, use_hyde=do_hyde
            )
        else:
            chunks_en = []
        if use_reranker and chunks_en:
            chunks_en = await cl.make_async(rerank_chunks)(search_query, chunks_en, top_k=8)
        await thought_step.stream_token(f"- Retrieved {len(chunks_en)} US guideline passages.\n\n")

        # Step 4 — Retrieve GESIDA guidelines (ES)
        chunks_es = []
        if use_es:
            await thought_step.stream_token("Searching Spanish Guidelines (GESIDA)...\n")
            if use_expand:
                chunks_es = await cl.make_async(expand_and_search_simple)(
                    query=search_query, collection=COLLECTION_ES, limit=8, use_hyde=do_hyde, lang="es"
                )
            elif use_hybrid:
                chunks_es = await cl.make_async(hybrid_search_simple)(
                    query=search_query, collection=COLLECTION_ES, limit=8, use_hyde=do_hyde
                )
            if use_reranker and chunks_es:
                chunks_es = await cl.make_async(rerank_chunks)(search_query, chunks_es, top_k=5)
            await thought_step.stream_token(f"- Retrieved {len(chunks_es)} GESIDA guideline passages.\n\n")

        # Step 5 — Grounding agent
        await thought_step.stream_token("Agent 2: Grounding & Retrieval Verification...\n")
        doc_eval     = await verify_documents_agent(search_query, chunks_en + chunks_es)
        docs_relevant = doc_eval.get("relevant", True)
        doc_score    = doc_eval.get("relevance_score", 1.0)
        doc_reason   = doc_eval.get("reason", "")
        await thought_step.stream_token(
            f"- Status: {'Sufficient Evidence Found' if docs_relevant else 'Low Retrieval Relevance'} (Score: {doc_score:.2f})\n"
            f"- Detail: {doc_reason}\n\n"
        )

        if not docs_relevant or doc_score < 0.25:
            elapsed = max(1, round(time.time() - start_time))
            thought_step.name = f"Thought for {elapsed}s"
            await thought_step.update()
            await cl.Message(content=(
                "**No Clinically Relevant Guidelines Found**\n\n"
                "The retrieved passages do not contain adequate recommendations for this inquiry.\n\n"
                f"*{doc_reason}*\n\n"
                "Please rephrase or provide additional clinical details."
            )).send()
            return

        # Step 6 — GraphRAG exploration
        entities, triplets = [], []
        if use_graph:
            await thought_step.stream_token("Exploring GraphRAG Knowledge Base & Subgraphs...\n")
            try:
                entities = extract_entities(search_query + " " + content)
                resolved = list(set(filter(None, [
                    await resolve_async(e, gc) for e in entities
                ])))

                raw = []
                try:
                    raw = await cl.make_async(get_triplets)(resolved)
                except Exception:
                    pass

                if use_pathrag and raw:
                    try:
                        triplets = pathrag_filter(raw, resolved)
                    except Exception:
                        triplets = raw
                else:
                    triplets = raw

                if use_ppr and gc.get("ppr") and resolved:
                    try:
                        ppr_t = await cl.make_async(get_ppr_triplets)(gc["ppr"], resolved, top_k=10)
                        seen  = {(t["subject"], t["relation"], t["object"]) for t in triplets}
                        triplets += [t for t in ppr_t if (t["subject"], t["relation"], t["object"]) not in seen]
                    except Exception:
                        pass

                await thought_step.stream_token(
                    f"- Extracted entities: `{', '.join(entities) or 'None'}`\n"
                    f"- Validated graph triplets: {len(triplets)}\n\n"
                )

                if entities and triplets:
                    try:
                        html_path = await cl.make_async(build_graph_html)(triplets)
                        if html_path:
                            elements.append(cl.Html(name="Biomedical Knowledge Graph", path=html_path, display="side"))
                            element_buttons.append("**Biomedical Knowledge Graph**")
                    except Exception:
                        pass

            except Exception as e:
                await thought_step.stream_token(f"- GraphRAG Warning: {str(e)}\n\n")

        # Step 7 — Attach source PDFs to sidebar
        for p in chunks_en[:4]:
            fname    = p.payload.get("filename", "")
            page_num = p.payload.get("page", 0)
            pdf_path = os.path.join("data", "en", os.path.basename(fname))
            pdf_name = f"US: {os.path.basename(fname)} p.{page_num+1}"
            if os.path.isfile(pdf_path):
                elements.append(cl.Pdf(name=pdf_name, path=pdf_path, display="side", page=page_num+1))
                element_buttons.append(f"**{pdf_name}**")

        for p in chunks_es[:2]:
            fname    = p.payload.get("filename", "")
            page_num = p.payload.get("page", 0)
            pdf_path = os.path.join("data", "es", os.path.basename(fname))
            pdf_name = f"ES: {os.path.basename(fname)} p.{page_num+1}"
            if os.path.isfile(pdf_path):
                elements.append(cl.Pdf(name=pdf_name, path=pdf_path, display="side", page=page_num+1))
                element_buttons.append(f"**{pdf_name}**")

        # Step 8 — Build generation prompt
        ctx_en    = "\n\n".join([p.payload.get("original_text", p.payload.get("text", "")) for p in chunks_en])
        ctx_es    = "\n\n".join([p.payload.get("original_text", p.payload.get("text", "")) for p in chunks_es])
        graph_txt = fmt_triplets(triplets)

        system = """You are an expert HIV/AIDS clinical decision support assistant.
CRITICAL LANGUAGE INSTRUCTION:
- Always answer ENTIRELY in the EXACT SAME LANGUAGE as the user's inquiry.
- Do not use decorative emojis. Maintain a serious, clear medical tone.

Structure:
## Clinical Recommendations
  NIH/HHS: [recommendations + citation]
  GESIDA: [recommendations + citation]
## Monitoring & Special Considerations
## Missing Critical Information

Then EXACTLY 3 follow-up clinical questions in the user's language:
? 1. [most urgent priority]
? 2. [second priority]
? 3. [third priority]

Brief medical disclaimer in the user's language."""

        user_prompt = (
            f"Patient Context: {patient_ctx or 'No patient record selected'}\n\n"
            + (f"Recent Conversation:\n{recent_history}\n\n" if recent_history else "")
            + f"NIH/HHS Guidelines Context:\n{ctx_en or 'No English sources available.'}\n\n"
            f"GESIDA Guidelines Context:\n{ctx_es or 'No Spanish sources available.'}\n\n"
            + (f"Biomedical Knowledge Graph:\n{graph_txt}\n\n" if graph_txt else "")
            + f"User Inquiry (Reply in this language): {content}"
        )

        await thought_step.stream_token(f"Synthesizing clinical response using {model_selected}...\n")

        # Stream DeepSeek reasoning (CoT) into the thought step
        deepseek_stream = None
        if "DeepSeek" in model_selected and deepseek_client and "reasoner" in model_selected:
            deepseek_stream = await deepseek_client.chat.completions.create(
                model="deepseek-reasoner",
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user_prompt}],
                stream=True,
            )
            async for chunk in deepseek_stream:
                delta     = chunk.choices[0].delta
                reasoning = getattr(delta, "reasoning_content", None)
                if reasoning:
                    await thought_step.stream_token(reasoning)
                if delta.content:
                    break

        elapsed           = max(1, round(time.time() - start_time))
        thought_step.name = f"Thought for {elapsed}s"
        await thought_step.update()

    # ── Stream final response ──────────────────────────────────────────────

    header_content = ""
    if element_buttons:
        header_content = (
            "**Source Documents & Subgraphs (Click to reopen in sidebar panel):**\n"
            + " · ".join(element_buttons)
            + "\n\n---\n\n"
        )

    response_msg = cl.Message(content=header_content, elements=elements)
    await response_msg.send()
    full_response = ""

    if "DeepSeek" in model_selected and deepseek_client:
        if deepseek_stream:
            async for chunk in deepseek_stream:
                text_chunk = chunk.choices[0].delta.content or ""
                if text_chunk:
                    full_response += text_chunk
                    await response_msg.stream_token(text_chunk)
        else:
            stream = await deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user_prompt}],
                stream=True,
            )
            async for chunk in stream:
                text_chunk = chunk.choices[0].delta.content or ""
                full_response += text_chunk
                await response_msg.stream_token(text_chunk)

    elif "Anthropic" in model_selected and anthropic_client:
        async with anthropic_client.messages.stream(
            model="claude-3-5-sonnet-20241022",
            max_tokens=4096,
            system=system,
            messages=[{"role": "user", "content": user_prompt}],
        ) as stream:
            async for text_chunk in stream.text_stream:
                full_response += text_chunk
                await response_msg.stream_token(text_chunk)

    else:
        stream = await openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user_prompt}],
            temperature=0,
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            full_response += delta
            await response_msg.stream_token(delta)

    followups, cleaned_response = extract_and_clean_followup(full_response)

    response_msg.content = header_content + cleaned_response
    chat_history.append({"role": "user",      "content": content})
    chat_history.append({"role": "assistant",  "content": cleaned_response})
    cl.user_session.set("chat_history", chat_history[-6:])

    if followups:
        response_msg.actions = [
            cl.Action(name="followup", value=fq, label=fq, payload={"question": fq})
            for fq in followups
        ]
    await response_msg.update()


# ── Command Handlers ───────────────────────────────────────────────────────

async def _cmd_new_patient(content: str):
    parts = content.replace("/new_patient", "").strip().split("|")
    if len(parts) < 3:
        await cl.Message(content="Required format: `/new_patient Name | Age | CD4 | Viral Load | ARV History`").send()
        return

    def si(s):
        try:    return int(s.strip())
        except: return 0

    name     = parts[0].strip()
    pid      = str(uuid.uuid4())[:8]
    profiles = cl.user_session.get("profiles", {})
    profiles[pid] = {
        "name":        name,
        "age":         si(parts[1]),
        "cd4_count":   si(parts[2]),
        "viral_load":  parts[3].strip() if len(parts) > 3 else "Unknown",
        "arv_history": parts[4].strip() if len(parts) > 4 else "Unknown",
        "created_at":  datetime.now().isoformat(),
    }
    save_profiles(profiles)
    cl.user_session.set("profiles",   profiles)
    cl.user_session.set("active_pid", pid)
    await set_chat_title(name)
    p = profiles[pid]
    await cl.Message(content=(
        f"**Patient record initialized:**\n\n"
        f"| Parameter | Value |\n|---|---|\n"
        f"| Name | {p['name']} |\n| Age | {p['age']} y/o |\n"
        f"| CD4 Count | {p['cd4_count']} cells/mm³ |\n"
        f"| Viral Load | {p['viral_load']} |\n| ARV History | {p['arv_history']} |\n\n"
        f"You can now ask clinical inquiries for **{p['name']}**."
    )).send()


async def _cmd_delete(name_to_delete: str):
    profiles = cl.user_session.get("profiles", {})
    if not profiles:
        await cl.Message(content="No registered patient records to delete.").send()
        return

    if name_to_delete:
        matched_pid = next(
            (pid for pid, p in profiles.items() if name_to_delete.lower() in p["name"].lower()),
            None
        )
        if matched_pid:
            deleted_name = profiles[matched_pid]["name"]
            del profiles[matched_pid]
            save_profiles(profiles)
            cl.user_session.set("profiles", profiles)
            if cl.user_session.get("active_pid") == matched_pid:
                cl.user_session.set("active_pid", None)
            await cl.Message(content=f"Patient record **{deleted_name}** has been deleted.").send()
        else:
            await cl.Message(content=f"No patient found matching '{name_to_delete}'.").send()
        return

    actions = [
        cl.Action(name="execute_delete", value=pid,
                  label=f"Delete {p['name']}", payload={"pid": pid})
        for pid, p in profiles.items()
    ]
    await cl.Message(content="**Select a patient profile to delete:**", actions=actions).send()


async def _cmd_select():
    profiles = cl.user_session.get("profiles", {})
    if not profiles:
        await cl.Message(content="No registered patient records. Use `/new_patient`.").send()
        return
    actions = [
        cl.Action(name="select_patient", value=pid,
                  label=f"{p['name']} — CD4: {p['cd4_count']}", payload={"pid": pid})
        for pid, p in profiles.items()
    ]
    actions.append(cl.Action(name="new_patient",           value="new", label="New Patient Record",    payload={}))
    actions.append(cl.Action(name="delete_patient_prompt", value="del", label="Delete Patient Record", payload={}))
    await cl.Message(content="**Select a patient file:**", actions=actions).send()