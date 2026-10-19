# Role Simulation: ML & DevOps Engineer — LLMOps Deployment & Monitoring Runbook

**Project**: Project 03 · NLP + LLM Fine-Tuning (Text Summarization & Domain RAG System)  
**Owner Roles**: ML Engineer & DevOps/LLMOps Engineer  
**Container Image**: `29omchavan/mlops-project-3-nlp-rag-summarizer:latest`

---

## 1. Reproduction & Deployment Runbook

```bash
# 1. Create virtual environment & install dependencies
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Run 6-stage LoRA Fine-Tuning, ROUGE/BLEU Gate & Hybrid RAG Benchmark
python main.py

# 3. Run unit & integration tests
pytest tests/ -v

# 4. Launch FastAPI REST API + Web Dashboard (http://localhost:8080)
python app.py

# 5. Launch Streamlit UI Demo
streamlit run streamlit_app.py
```

---

## 2. CI/CD Model Evaluation Gate (`.github/workflows/main.yaml`)
Unlike traditional software CI/CD that only checks unit tests, production LLMOps pipelines enforce a **statistical quality gate** before building or deploying a new container image:
- In Stage 05 (`ModelEvaluation.evaluate`), both the base `google/flan-t5-small` and the newly trained PEFT LoRA adapter generate summaries across the validation corpus.
- If `ci_cd_evaluation_gate_passed` is `False` in `artifacts/model_evaluation/evaluation_report.json`, the GitHub Actions workflow exits with code `1`, preventing regression deployments.

---

## 3. Post-Deployment LLM Hallucination & Telemetry Monitoring (`components/llm_monitor.py`)
Every call to `/summarize` and `/rag-qa` logs a JSONL event to `artifacts/monitoring/llm_telemetry_logs.jsonl` containing:
- `prompt_tokens`, `response_tokens`, `latency_ms`, `estimated_cost_usd`
- `faithfulness_score`: Content-word grounding overlap between the generated answer and the retrieved Qdrant chunks / input dialogue:
  $$\text{Faithfulness}(A, C) = \frac{|\text{Tokens}_{\text{content}}(A) \cap \text{Tokens}_{\text{content}}(C)|}{|\text{Tokens}_{\text{content}}(A)|}$$
- `hallucination_alert`: Triggered automatically when $\text{Faithfulness} < 0.45$.
