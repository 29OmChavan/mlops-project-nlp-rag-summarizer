# Production MLOps Project 03: NLP + LLM Fine-Tuning (LoRA Summarizer & Hybrid Qdrant RAG System)

[![CI/CD Pipeline](https://github.com/29OmChavan/mlops-project-3-nlp-rag-summarizer/actions/workflows/main.yaml/badge.svg)](https://github.com/29OmChavan/mlops-project-3-nlp-rag-summarizer/actions)
[![Hugging Face LoRA](https://img.shields.io/badge/Hugging%20Face-29Oms%2Fmlops--project--3--lora--summarizer-FFD21E)](https://huggingface.co/29Oms/mlops-project-3-lora-summarizer)
[![DagsHub MLflow](https://img.shields.io/badge/MLflow-DagsHub-purple)](https://dagshub.com/29omchavan/mlops-project-3-nlp-rag-summarizer)
[![Docker Hub](https://img.shields.io/badge/Docker-29omchavan%2Fmlops--project--3--nlp--rag--summarizer-2496ED)](https://hub.docker.com/r/29omchavan/mlops-project-3-nlp-rag-summarizer)

An end-to-end **NLP, Transformer Fine-Tuning & Production LLMOps System** combining first-principles **BPE Tokenization, Multi-Head Self-Attention & LoRA** implementations with **PEFT LoRA Fine-Tuning (`google/flan-t5-small`)**, **Hugging Face Hub Model Versioning**, **Qdrant + BM25 Hybrid RAG with Reciprocal Rank Fusion (RRF) & Inline Citations**, **ROUGE/BLEU CI/CD Evaluation Gates**, **LLM Hallucination Monitoring**, **FastAPI + Streamlit**, and **Docker**.

---

## Documentation Index (Team Role Simulation)

* **[Complete Step-by-Step Implementation Guide](docs/IMPLEMENTATION_GUIDE.md)** — Architecture diagrams, BPE/Attention/LoRA math, Hybrid RRF RAG code walkthrough, and LLMOps monitoring.
* **[Data Engineer: NLP ETL, Tokenization & RAG Chunking Spec](docs/01_DATA_AND_TOKENIZATION_SPEC.md)** — Karpathy BPE algorithm, `-100` label masking, and dataset card.
* **[ML Researcher: Self-Attention, LoRA Math & RAG Benchmark Log](docs/02_LORA_AND_RAG_EXPERIMENT_LOG.md)** — $\sqrt{d_k}$ scaling derivation, LoRA low-rank matrix initialization, and Naive vs. Hybrid RAG retrieval benchmarks.
* **[ML & DevOps Engineer: LLMOps Deployment & Monitoring Runbook](docs/03_LLMOPS_DEPLOYMENT_AND_MONITORING_RUNBOOK.md)** — CI/CD ROUGE-L evaluation gate and faithfulness hallucination alerts.
* **[Changelog](CHANGELOG.md)** — Release history (`v1.0.0`).

---

## Quickstart

```bash
# 1. Create virtual environment & install dependencies
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Run 6-stage LoRA Fine-Tuning, HF Hub Push, ROUGE/BLEU Gate & Hybrid RAG Benchmark
python main.py

# 3. Run unit & integration test suite
pytest tests/ -v

# 4. Launch FastAPI Server (http://localhost:8080) or Streamlit App
python app.py
# streamlit run streamlit_app.py
```
