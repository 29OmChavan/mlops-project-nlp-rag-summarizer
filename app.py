"""
FastAPI Production Service for Project 03: LoRA Text Summarization & Hybrid RAG QA.
Provides:
- GET  /          : Interactive Summarization & Hybrid RAG Web UI
- GET  /health    : Readiness & artifact health check
- GET  /metrics   : Returns LoRA ROUGE/BLEU evaluation report, RAG benchmark, and LLM Monitor telemetry
- POST /summarize : Summarizes technical dialogues using PEFT LoRA fine-tuned Flan-T5
- POST /rag-qa    : Answers domain questions using Qdrant + BM25 RRF Hybrid RAG with [DOC-ID] citations
- POST /train     : Triggers full 6-stage training & evaluation pipeline
"""
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, HTTPException, Request, Header
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
import uvicorn

from src.nlp_rag_system.utils.common import load_json
from src.nlp_rag_system.pipeline.prediction_pipeline import PredictionPipeline
from src.nlp_rag_system.pipeline.training_pipeline import TrainingPipeline

app = FastAPI(
    title="Production MLOps Project 03: LoRA Summarization & Hybrid RAG LLMOps API",
    description=(
        "PEFT LoRA Fine-Tuned Transformer (google/flan-t5-small), Qdrant + BM25 Hybrid RAG "
        "with Reciprocal Rank Fusion & Citations, ROUGE/BLEU Gate, and LLM Hallucination Monitor."
    ),
    version="1.0.0",
)

templates = Jinja2Templates(directory="templates")
_predictor: Optional[PredictionPipeline] = None


def get_predictor() -> PredictionPipeline:
    global _predictor
    if _predictor is None:
        _predictor = PredictionPipeline()
    return _predictor


class SummarizationRequest(BaseModel):
    dialogue: str = Field(
        ...,
        min_length=15,
        description="Technical dialogue or incident transcript to summarize",
    )
    max_new_tokens: int = Field(default=80, ge=16, le=200)

    model_config = {
        "json_schema_extra": {
            "example": {
                "dialogue": (
                    "SRE Lead: Our FastAPI risk prediction container on AWS EC2 returned HTTP 500. "
                    "ML Engineer: Joblib failed to unpickle RiskDomainFeatureGenerator from __main__. "
                    "We moved it into the installable src package with pip install -e . and resolved the outage."
                ),
                "max_new_tokens": 80,
            }
        }
    }


class RAGQueryRequest(BaseModel):
    query: str = Field(..., min_length=5, description="Domain technical question")
    top_k: int = Field(default=3, ge=1, le=6)
    category_filter: Optional[str] = Field(
        default=None,
        description="Optional metadata category filter (e.g., 'llm_finetuning', 'rag_optimization', 'data_drift')",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "query": "How does LoRA parameterize weight updates and why is matrix B initialized to zero?",
                "top_k": 3,
                "category_filter": None,
            }
        }
    }


@app.get("/health", tags=["Operations"])
async def health_check():
    lora_ready = Path("artifacts/model_trainer/lora_adapter/adapter_config.json").exists()
    eval_ready = Path("artifacts/model_evaluation/evaluation_report.json").exists()
    return {
        "status": "healthy" if (lora_ready and eval_ready) else "degraded",
        "lora_adapter_ready": lora_ready,
        "evaluation_gate_ready": eval_ready,
        "version": "1.0.0",
    }


@app.get("/metrics", tags=["Operations"])
async def get_metrics():
    eval_path = Path("artifacts/model_evaluation/evaluation_report.json")
    rag_path = Path("artifacts/rag_pipeline/rag_benchmark_report.json")
    mon_path = Path("artifacts/monitoring/monitoring_summary.json")

    if not eval_path.exists():
        raise HTTPException(status_code=404, detail="Evaluation report not found. Run /train first.")

    return {
        "lora_evaluation": load_json(eval_path),
        "rag_benchmark": load_json(rag_path) if rag_path.exists() else {},
        "llm_monitoring": load_json(mon_path) if mon_path.exists() else {},
    }


@app.post("/summarize", tags=["LLMOps Inference"])
async def summarize_endpoint(
    payload: SummarizationRequest,
    x_api_key: Optional[str] = Header(default=None),
):
    try:
        predictor = get_predictor()
        return predictor.summarize(
            dialogue=payload.dialogue, max_new_tokens=payload.max_new_tokens
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rag-qa", tags=["LLMOps Inference"])
async def rag_qa_endpoint(
    payload: RAGQueryRequest,
    x_api_key: Optional[str] = Header(default=None),
):
    try:
        predictor = get_predictor()
        return predictor.answer_question_rag(
            query=payload.query,
            top_k=payload.top_k,
            category_filter=payload.category_filter,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/train", tags=["Pipeline"])
async def trigger_training():
    try:
        res = TrainingPipeline().run_pipeline()
        return {"status": "Pipeline Completed", "results": res}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/", response_class=HTMLResponse, tags=["UI"])
async def home_ui(request: Request):
    eval_path = Path("artifacts/model_evaluation/evaluation_report.json")
    rag_path = Path("artifacts/rag_pipeline/rag_benchmark_report.json")
    eval_data = load_json(eval_path) if eval_path.exists() else None
    rag_data = load_json(rag_path) if rag_path.exists() else None
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"eval_data": eval_data, "rag_data": rag_data},
    )


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=False)
