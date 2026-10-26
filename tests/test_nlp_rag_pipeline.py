"""
Automated Unit & Integration Test Suite for Project 03 (v2.0.0):
1. First-Principles BPE Tokenizer round-trip & compression test
2. First-Principles Multi-Head Causal Self-Attention & LoRA zero-init mathematical identity test
3. End-to-End 6-stage pipeline execution, ROUGE/BLEU evaluation gate, and Hybrid RAG benchmark
4. Citation-backed Hybrid RAG retrieval accuracy (`[DOC-101]`..`[DOC-106]`)
5. LLM Hallucination Monitor faithfulness scoring & FastAPI v2.0.0 endpoints + 6-page UI routes
"""
import os
from pathlib import Path
import torch
from fastapi.testclient import TestClient

from src.nlp_rag_system.components.first_principles_llm import (
    BytePairTokenizerFromScratch,
    ScaledDotProductMultiHeadAttention,
    LoRALinearFromScratch,
)
from src.nlp_rag_system.components.llm_monitor import LLMMonitor
from src.nlp_rag_system.pipeline.training_pipeline import TrainingPipeline
from app import app

client = TestClient(app)


def test_bpe_tokenizer_from_scratch():
    sample = "mlops pipeline dvc mlops pipeline dvc lora rag"
    tok = BytePairTokenizerFromScratch()
    tok.train(sample, vocab_size=268)
    ids = tok.encode(sample)
    assert len(ids) < len(sample.encode("utf-8"))
    assert tok.decode(ids) == sample


def test_attention_and_lora_from_scratch():
    mha = ScaledDotProductMultiHeadAttention(d_model=16, num_heads=4)
    x = torch.randn(2, 6, 16)
    out, attn = mha(x, causal_mask=True)
    assert out.shape == (2, 6, 16)
    assert torch.allclose(attn[0, 0, 0, 1:], torch.zeros(5), atol=1e-6)

    lora = LoRALinearFromScratch(in_features=32, out_features=16, r=4, lora_alpha=8.0)
    inp = torch.randn(4, 32)
    assert torch.allclose(lora(inp), lora.pretrained(inp), atol=1e-6)
    assert lora.pretrained.weight.requires_grad is False
    assert lora.lora_A.requires_grad is True
    assert lora.lora_B.requires_grad is True


def test_end_to_end_pipeline_and_gates():
    if not Path("artifacts/model_evaluation/evaluation_report.json").exists():
        os.environ["DISABLE_REMOTE_MLFLOW"] = "1"
        os.environ["DISABLE_HF_PUSH"] = "1"
        results = TrainingPipeline().run_pipeline()
        assert results["evaluation"]["ci_cd_evaluation_gate_passed"] is True
        assert results["rag_benchmark"]["advanced_hybrid_rrf_rag"]["recall_at_3"] >= 0.80

    assert Path("artifacts/model_trainer/lora_adapter/adapter_config.json").exists()
    assert Path("artifacts/model_evaluation/evaluation_report.json").exists()
    assert Path("artifacts/rag_pipeline/rag_benchmark_report.json").exists()


def test_llm_monitor_faithfulness_and_hallucination_flag():
    mon = LLMMonitor()
    ctx = ["LoRA freezes pretrained weights W_0 and trains rank decomposition matrices A and B."]
    grounded_ans = "LoRA freezes pretrained weights and trains matrices A and B."
    hallucinated_ans = "Quantum blockchain hyperdrives accelerate photonic warp nacelles."

    assert mon.compute_faithfulness_score(grounded_ans, ctx) >= 0.75
    assert mon.compute_faithfulness_score(hallucinated_ans, ctx) < 0.20


def test_fastapi_v2_endpoints_and_pages():
    # JSON API health & metrics
    h = client.get("/health")
    assert h.status_code == 200
    assert h.json()["status"] == "healthy"

    m = client.get("/metrics")
    assert m.status_code == 200
    assert m.json()["lora_evaluation"]["ci_cd_evaluation_gate_passed"] is True

    # Browser HTML content negotiation on /health, /metrics, /intro, /studio, /chunk-visualizer, /prompt-playground, /help, /docs
    for route in ["/intro", "/studio", "/chunk-visualizer", "/prompt-playground", "/evaluation-hub", "/health", "/metrics", "/help", "/docs"]:
        html_res = client.get(route, headers={"Accept": "text/html"})
        assert html_res.status_code == 200
        assert "PEFT LoRA Summarizer &amp; Hybrid RAG Suite" in html_res.text or "PEFT LoRA Summarizer & Hybrid RAG Suite" in html_res.text

    # Hybrid RAG QA + 3-mode comparison
    rag_resp = client.post(
        "/rag-qa",
        json={
            "query": "How does LoRA parameterize weight updates and why is matrix B initialized to zero?",
            "top_k": 3,
        },
    )
    assert rag_resp.status_code == 200
    r_data = rag_resp.json()
    assert "[DOC-101]" in r_data["citation_tags"]
    assert r_data["faithfulness_score"] >= 0.60
    assert "mode_comparison" in r_data
    assert len(r_data["mode_comparison"]["hybrid_rrf"]) >= 1

    # Multi-persona Summarizer + compression + claim attribution
    sum_resp = client.post(
        "/summarize",
        json={
            "dialogue": (
                "SRE Lead: Our FastAPI risk prediction container on AWS EC2 returned HTTP 500. "
                "ML Engineer: Joblib failed to unpickle RiskDomainFeatureGenerator from __main__. "
                "We moved it into the installable src package with pip install -e . and resolved the outage."
            ),
            "max_new_tokens": 64,
            "persona": "sre_deep_dive",
        },
    )
    assert sum_resp.status_code == 200
    s_data = sum_resp.json()
    assert len(s_data["summary"]) > 10
    assert "compression_metrics" in s_data
    assert "structured_extraction" in s_data
    assert "claim_attributions" in s_data

    # Knowledge Base 2D Projection & Live Custom Ingestion
    kb_resp = client.get("/api/kb-chunks")
    assert kb_resp.status_code == 200
    initial_count = kb_resp.json()["total_chunks"]
    assert initial_count >= 6

    ing_resp = client.post(
        "/api/ingest-chunk",
        json={
            "title": "vLLM PagedAttention KV Cache Memory Manager",
            "category": "llm_finetuning",
            "source": "OSDI Systems Paper",
            "chunk_text": "PagedAttention partitions the KV cache into non-contiguous virtual memory blocks to eliminate fragmentation.",
        },
    )
    assert ing_resp.status_code == 200
    assert ing_resp.json()["total_indexed_chunks"] == initial_count + 1

    # Prompt Playground & First-Principles Lab
    pg_resp = client.post(
        "/api/prompt-playground",
        json={
            "query": "How does PagedAttention manage KV cache memory?",
            "prompt_template": "Role: {persona}\nContext:\n{context}\nQ: {query}",
            "persona": "executive_brief",
            "top_k": 3,
        },
    )
    assert pg_resp.status_code == 200
    assert pg_resp.json()["faithfulness_score"] >= 0.50

    fp_resp = client.post(
        "/api/first-principles",
        json={
            "sample_text": "mlops pipeline dvc mlops pipeline dvc lora rag",
            "vocab_size": 266,
            "d_model": 32,
            "num_heads": 4,
            "lora_r": 8,
            "lora_alpha": 16.0,
        },
    )
    assert fp_resp.status_code == 200
    fp_data = fp_resp.json()
    assert fp_data["bpe_lab"]["lossless_roundtrip_verified"] is True
    assert fp_data["lora_lab"]["zero_init_identity_verified"] is True
    assert fp_data["attention_lab"]["causal_mask_verified"] is True
