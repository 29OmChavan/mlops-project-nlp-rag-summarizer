# Changelog

All notable changes to **Project 03: NLP + LLM Fine-Tuning & Hybrid RAG System** are documented in this file.

## [1.0.0] - 2026-10-19
### Added
- **First-Principles LLM Module (`first_principles_llm.py`)**: Implemented `BytePairTokenizerFromScratch` (Karpathy BPE), `ScaledDotProductMultiHeadAttention`, and `LoRALinearFromScratch` (Umar Jamil LoRA).
- **PEFT LoRA Fine-Tuning (`model_trainer.py`)**: Fine-tuned `google/flan-t5-small` (`r=8, lora_alpha=16`, `0.445%` trainable parameters) and pushed adapters to Hugging Face Hub (`29Oms/mlops-project-3-lora-summarizer`).
- **ROUGE/BLEU Evaluation & CI/CD Gate (`model_evaluation.py`)**: Evaluated Base vs. LoRA models across `ROUGE-1`, `ROUGE-2`, `ROUGE-L`, `ROUGE-Lsum`, and `BLEU`, enforcing a CI/CD quality gate and logging to DagsHub MLflow.
- **Advanced Hybrid RAG (`rag_engine.py`)**: Built Qdrant dense vector search + BM25 lexical search with Query Expansion, Metadata Filtering, Reciprocal Rank Fusion (RRF), Cross-Feature Re-Ranking, and `[DOC-ID]` inline citations.
- **LLM Hallucination & Telemetry Monitor (`llm_monitor.py`)**: Logged token usage, latency, cost, and faithfulness grounding scores.
- **FastAPI + Streamlit + Docker**: Created `app.py`, `streamlit_app.py`, `Dockerfile`, `docker-compose.yml`, `.github/workflows/main.yaml`, and `pytest` suite (`tests/test_nlp_rag_pipeline.py`).
