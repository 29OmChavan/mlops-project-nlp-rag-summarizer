"""
FastAPI Production Service for Project 03 (v2.0.0):
6-Page Interactive PEFT LoRA Summarization, First-Principles LLM & Hybrid RAG Platform.
Provides:
- GET  /                  : 6-Page Interactive Web Application (defaults to Studio)
- GET  /intro             : Architecture & Engineering Walkthrough ("How I Built This From Scratch")
- GET  /studio            : Multi-Persona LoRA Summarizer & Hybrid RAG Duel Studio
- GET  /chunk-visualizer  : Qdrant 2D Vector Space Explorer & Live Document Ingestion
- GET  /prompt-playground : Custom Prompt Template Sandbox & First-Principles BPE/LoRA/Attention Lab
- GET  /evaluation-hub    : ROUGE/BLEU Gates, Hybrid RRF Benchmark & Telemetry Audit Log
- GET  /health            : Readiness & artifact health check (HTML in browser / JSON for API)
- GET  /metrics           : LoRA ROUGE/BLEU report, RAG benchmark & LLM Monitor telemetry (HTML/JSON)
- POST /summarize         : Summarizes technical dialogues with persona, compression & claim attributions
- POST /rag-qa            : Answers domain questions via Qdrant + BM25 RRF Hybrid RAG with [DOC-ID] citations
- POST /api/rag-duel      : Side-by-side Naive Dense vs. Pure BM25 vs. Hybrid RRF retrieval comparison
- GET  /api/kb-chunks     : Returns indexed knowledge base chunks + 2D SVD coordinates
- POST /api/ingest-chunk  : Live custom document ingestion into Qdrant + BM25
- POST /api/prompt-playground : Custom prompt template execution with faithfulness scoring
- POST /api/first-principles  : Interactive BPE Tokenizer, Causal Attention & LoRA Rank Lab
- POST /train             : Triggers full 6-stage training & evaluation pipeline
"""
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, HTTPException, Request, Header, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
import uvicorn

from src.nlp_rag_system.utils.common import load_json
from src.nlp_rag_system.pipeline.prediction_pipeline import PredictionPipeline
from src.nlp_rag_system.components.nlp_rag_suite import (
    NLPRagInteractiveSuite,
    SAMPLE_PRESETS,
)

app = FastAPI(
    title="PEFT LoRA Summarizer & Hybrid RAG Suite",
    description=(
        "Incident log summarization with Flan-T5 LoRA adapters, Qdrant + BM25 RRF search, "
        "BPE/LoRA math checks, and grounding overlap logs."
    ),
    version="2.0.0",
    docs_url="/swagger-raw",
)

templates = Jinja2Templates(directory="templates")
_predictor: Optional[PredictionPipeline] = None
_suite: Optional[NLPRagInteractiveSuite] = None


def get_predictor() -> PredictionPipeline:
    global _predictor
    if _predictor is None:
        _predictor = PredictionPipeline()
    return _predictor


def get_suite() -> NLPRagInteractiveSuite:
    global _suite
    if _suite is None:
        _suite = NLPRagInteractiveSuite(get_predictor())
    return _suite


def _wants_html(request: Request, format_param: Optional[str]) -> bool:
    if format_param and format_param.lower() == "json":
        return False
    if format_param and format_param.lower() == "html":
        return True
    accept = request.headers.get("accept", "")
    return "text/html" in accept.lower()


def _load_recent_telemetry(limit: int = 20) -> List[Dict[str, Any]]:
    log_path = Path("artifacts/monitoring/llm_telemetry_logs.jsonl")
    if not log_path.exists():
        return []
    rows = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
    return list(reversed(rows[-limit:]))


def _build_ui_context(active_page: str = "studio") -> Dict[str, Any]:
    eval_path = Path("artifacts/model_evaluation/evaluation_report.json")
    rag_path = Path("artifacts/rag_pipeline/rag_benchmark_report.json")
    mon_path = Path("artifacts/monitoring/monitoring_summary.json")
    lora_ready = Path("artifacts/model_trainer/lora_adapter/adapter_config.json").exists()
    eval_ready = eval_path.exists()

    eval_data = load_json(eval_path) if eval_ready else {}
    rag_data = load_json(rag_path) if rag_path.exists() else {}
    mon_data = load_json(mon_path) if mon_path.exists() else {}
    health_data = {
        "status": "healthy" if (lora_ready and eval_ready) else "degraded",
        "lora_adapter_ready": lora_ready,
        "evaluation_gate_ready": eval_ready,
        "version": "2.0.0",
    }

    return {
        "active_page": active_page,
        "eval_data": eval_data,
        "rag_data": rag_data,
        "mon_data": mon_data,
        "health_data": health_data,
        "telemetry_logs": _load_recent_telemetry(15),
        "sample_presets": SAMPLE_PRESETS,
    }


class SummarizationRequest(BaseModel):
    dialogue: str = Field(
        ...,
        min_length=15,
        description="Technical dialogue or incident transcript to summarize",
    )
    max_new_tokens: int = Field(default=80, ge=16, le=200)
    persona: str = Field(
        default="executive_brief",
        description="Summary persona: executive_brief | sre_deep_dive | compliance_sla_audit",
    )


class RAGQueryRequest(BaseModel):
    query: str = Field(..., min_length=5, description="Domain technical question")
    top_k: int = Field(default=3, ge=1, le=6)
    category_filter: Optional[str] = Field(default=None)


class ChunkIngestRequest(BaseModel):
    title: str = Field(..., min_length=4, description="Document title")
    category: str = Field(default="mlops_architecture", description="Domain category tag")
    source: str = Field(default="SRE Architecture Runbook", description="Citation source")
    chunk_text: str = Field(..., min_length=20, description="Technical paragraph to index")


class PromptPlaygroundRequest(BaseModel):
    query: str = Field(..., min_length=5)
    prompt_template: str = Field(
        default=(
            "You are a Staff MLOps Architect writing a {persona}.\n"
            "Grounded Context:\n{context}\n\n"
            "Question: {query}\n"
            "Provide a concise answer with inline [DOC-ID] citations:"
        )
    )
    persona: str = Field(default="executive_brief")
    top_k: int = Field(default=3, ge=1, le=6)


class FirstPrinciplesRequest(BaseModel):
    sample_text: str = Field(
        default="mlops pipeline dvc mlops pipeline dvc lora rag attention"
    )
    vocab_size: int = Field(default=268, ge=257, le=320)
    d_model: int = Field(default=32, ge=8, le=128)
    num_heads: int = Field(default=4, ge=1, le=8)
    lora_r: int = Field(default=8, ge=1, le=64)
    lora_alpha: float = Field(default=16.0, ge=1.0, le=128.0)


@app.get("/health", tags=["Operations"])
async def health_check(request: Request, format: Optional[str] = Query(default=None)):
    lora_ready = Path("artifacts/model_trainer/lora_adapter/adapter_config.json").exists()
    eval_ready = Path("artifacts/model_evaluation/evaluation_report.json").exists()
    payload = {
        "status": "healthy" if (lora_ready and eval_ready) else "degraded",
        "lora_adapter_ready": lora_ready,
        "evaluation_gate_ready": eval_ready,
        "version": "2.0.0",
    }
    if _wants_html(request, format):
        ctx = _build_ui_context(active_page="health")
        return templates.TemplateResponse(request=request, name="index.html", context=ctx)
    return JSONResponse(content=payload)


@app.get("/metrics", tags=["Operations"])
async def get_metrics(request: Request, format: Optional[str] = Query(default=None)):
    eval_path = Path("artifacts/model_evaluation/evaluation_report.json")
    rag_path = Path("artifacts/rag_pipeline/rag_benchmark_report.json")
    mon_path = Path("artifacts/monitoring/monitoring_summary.json")

    if not eval_path.exists():
        raise HTTPException(
            status_code=404, detail="Evaluation report not found. Run /train first."
        )

    payload = {
        "lora_evaluation": load_json(eval_path),
        "rag_benchmark": load_json(rag_path) if rag_path.exists() else {},
        "llm_monitoring": load_json(mon_path) if mon_path.exists() else {},
    }
    if _wants_html(request, format):
        ctx = _build_ui_context(active_page="evaluation")
        return templates.TemplateResponse(request=request, name="index.html", context=ctx)
    return JSONResponse(content=payload)


@app.post("/summarize", tags=["LLMOps Inference"])
async def summarize_endpoint(
    payload: SummarizationRequest,
    x_api_key: Optional[str] = Header(default=None),
):
    try:
        suite = get_suite()
        return suite.summarize_advanced(
            dialogue=payload.dialogue,
            max_new_tokens=payload.max_new_tokens,
            persona=payload.persona,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rag-qa", tags=["LLMOps Inference"])
async def rag_qa_endpoint(
    payload: RAGQueryRequest,
    x_api_key: Optional[str] = Header(default=None),
):
    try:
        suite = get_suite()
        return suite.compare_retrieval_modes(
            query=payload.query,
            top_k=payload.top_k,
            category_filter=payload.category_filter,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/rag-duel", tags=["LLMOps Inference"])
async def rag_duel_endpoint(payload: RAGQueryRequest):
    try:
        return get_suite().compare_retrieval_modes(
            query=payload.query,
            top_k=payload.top_k,
            category_filter=payload.category_filter,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/kb-chunks", tags=["Knowledge Base"])
async def kb_chunks_endpoint():
    try:
        return get_suite().get_knowledge_base_projection()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ingest-chunk", tags=["Knowledge Base"])
async def ingest_chunk_endpoint(payload: ChunkIngestRequest):
    try:
        return get_suite().ingest_custom_chunk(
            title=payload.title,
            category=payload.category,
            source=payload.source,
            chunk_text=payload.chunk_text,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/prompt-playground", tags=["Prompt Lab"])
async def prompt_playground_endpoint(payload: PromptPlaygroundRequest):
    try:
        return get_suite().run_prompt_playground(
            query=payload.query,
            prompt_template=payload.prompt_template,
            persona=payload.persona,
            top_k=payload.top_k,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/first-principles", tags=["First-Principles Lab"])
async def first_principles_endpoint(payload: FirstPrinciplesRequest):
    try:
        return get_suite().run_first_principles_lab(
            sample_text=payload.sample_text,
            vocab_size=payload.vocab_size,
            d_model=payload.d_model,
            num_heads=payload.num_heads,
            lora_r=payload.lora_r,
            lora_alpha=payload.lora_alpha,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/presets", tags=["UI"])
async def presets_endpoint():
    return SAMPLE_PRESETS


@app.post("/train", tags=["Pipeline"])
async def trigger_training():
    try:
        from src.nlp_rag_system.pipeline.training_pipeline import TrainingPipeline

        res = TrainingPipeline().run_pipeline()
        return {"status": "Pipeline Completed", "results": res}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/", response_class=HTMLResponse, tags=["UI"])
async def home_ui(request: Request):
    return templates.TemplateResponse(
        request=request, name="index.html", context=_build_ui_context("studio")
    )


@app.get("/intro", response_class=HTMLResponse, tags=["UI"])
async def intro_ui(request: Request):
    return templates.TemplateResponse(
        request=request, name="index.html", context=_build_ui_context("intro")
    )


@app.get("/studio", response_class=HTMLResponse, tags=["UI"])
async def studio_ui(request: Request):
    return templates.TemplateResponse(
        request=request, name="index.html", context=_build_ui_context("studio")
    )


@app.get("/chunk-visualizer", response_class=HTMLResponse, tags=["UI"])
async def chunk_visualizer_ui(request: Request):
    return templates.TemplateResponse(
        request=request, name="index.html", context=_build_ui_context("chunks")
    )


@app.get("/prompt-playground", response_class=HTMLResponse, tags=["UI"])
async def prompt_playground_ui(request: Request):
    return templates.TemplateResponse(
        request=request, name="index.html", context=_build_ui_context("playground")
    )


@app.get("/evaluation-hub", response_class=HTMLResponse, tags=["UI"])
async def evaluation_hub_ui(request: Request):
    return templates.TemplateResponse(
        request=request, name="index.html", context=_build_ui_context("evaluation")
    )


@app.get("/help", response_class=HTMLResponse, tags=["UI"])
async def help_ui(request: Request):
    return templates.TemplateResponse(
        request=request, name="index.html", context=_build_ui_context("help")
    )


@app.get("/docs", response_class=HTMLResponse, tags=["UI"])
async def docs_ui(request: Request):
    return templates.TemplateResponse(
        request=request, name="index.html", context=_build_ui_context("docs")
    )


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=False)
