"""
Automated Unit & Integration Test Suite for Project 03:
1. First-Principles BPE Tokenizer round-trip & compression test
2. First-Principles Multi-Head Causal Self-Attention & LoRA zero-init mathematical identity test
3. End-to-End 6-stage pipeline execution, ROUGE/BLEU evaluation gate, and Hybrid RAG benchmark
4. Citation-backed Hybrid RAG retrieval accuracy (`[DOC-101]`..`[DOC-106]`)
5. LLM Hallucination Monitor faithfulness scoring & FastAPI endpoints (`/health`, `/metrics`, `/summarize`, `/rag-qa`)
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
    # Upper triangle above diagonal must be 0 after causal masking + softmax
    assert torch.allclose(attn[0, 0, 0, 1:], torch.zeros(5), atol=1e-6)

    lora = LoRALinearFromScratch(in_features=32, out_features=16, r=4, lora_alpha=8.0)
    inp = torch.randn(4, 32)
    # At step 0, B = 0 => lora(inp) == pretrained(inp)
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


def test_fastapi_endpoints():
    h = client.get("/health")
    assert h.status_code == 200
    assert h.json()["status"] == "healthy"

    m = client.get("/metrics")
    assert m.status_code == 200
    assert m.json()["lora_evaluation"]["ci_cd_evaluation_gate_passed"] is True

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

    sum_resp = client.post(
        "/summarize",
        json={
            "dialogue": (
                "SRE Lead: Our FastAPI risk prediction container on AWS EC2 returned HTTP 500. "
                "ML Engineer: Joblib failed to unpickle RiskDomainFeatureGenerator from __main__. "
                "We moved it into the installable src package with pip install -e . and resolved the outage."
            ),
            "max_new_tokens": 64,
        },
    )
    assert sum_resp.status_code == 200
    assert len(sum_resp.json()["summary"]) > 10
