"""
Streamlit Interactive Demo Application for Project 03:
LoRA Technical Summarizer + Citation-Backed Hybrid RAG QA + Live LLM Monitoring Telemetry.
Run via:
    streamlit run streamlit_app.py
"""
from pathlib import Path
import streamlit as st
from src.nlp_rag_system.utils.common import load_json
from src.nlp_rag_system.pipeline.prediction_pipeline import PredictionPipeline

st.set_page_config(
    page_title="Project 03 · LoRA Summarizer & Hybrid RAG",
    layout="wide",
)

st.title("Production LLMOps: PEFT LoRA Summarizer & Hybrid Qdrant RAG")
st.caption("Fine-Tuned `google/flan-t5-small` (LoRA) · Qdrant + BM25 RRF · Multi-Document Citations · Hallucination Monitor")


@st.cache_resource
def load_pipeline() -> PredictionPipeline:
    return PredictionPipeline()


predictor = load_pipeline()

tab1, tab2, tab3 = st.tabs(
    ["1. Hybrid RAG QA (with Citations)", "2. LoRA Incident Summarizer", "3. ROUGE/BLEU & LLM Monitoring"]
)

with tab1:
    st.subheader("Ask a Domain MLOps / LLMOps Question")
    q = st.text_input(
        "Question",
        value="How does LoRA parameterize weight updates and why is matrix B initialized to zero?",
    )
    if st.button("Run Hybrid RAG Retrieval & Synthesis"):
        res = predictor.answer_question_rag(query=q, top_k=3)
        st.success(res["answer"])
        st.write(
            f"**Citations:** `{', '.join(res['citation_tags'])}` | "
            f"**Faithfulness Score:** `{res['faithfulness_score']}` | "
            f"**Latency:** `{res['latency_ms']} ms`"
        )
        st.json(res["citations"])

with tab2:
    st.subheader("Summarize Technical Incident Dialogue (PEFT LoRA)")
    dlg = st.text_area(
        "Dialogue Transcript",
        value=(
            "SRE Lead: Our FastAPI risk prediction container on AWS EC2 returned HTTP 500. "
            "ML Engineer: Joblib failed to unpickle RiskDomainFeatureGenerator from __main__. "
            "We moved it into the installable src package with pip install -e . and resolved the outage."
        ),
        height=140,
    )
    if st.button("Generate LoRA Summary"):
        res = predictor.summarize(dialogue=dlg)
        st.info(res["summary"])
        st.write(
            f"**Faithfulness:** `{res['faithfulness_score']}` | "
            f"**Latency:** `{res['latency_ms']} ms` | **Model:** `{res['model']}`"
        )

with tab3:
    eval_path = Path("artifacts/model_evaluation/evaluation_report.json")
    rag_path = Path("artifacts/rag_pipeline/rag_benchmark_report.json")
    mon_path = Path("artifacts/monitoring/monitoring_summary.json")
    col1, col2 = st.columns(2)
    with col1:
        if eval_path.exists():
            st.markdown("#### LoRA vs. Base Model Evaluation (ROUGE / BLEU)")
            st.json(load_json(eval_path))
    with col2:
        if rag_path.exists():
            st.markdown("#### Naive Dense vs. Hybrid RRF RAG Benchmark")
            st.json(load_json(rag_path))
        if mon_path.exists():
            st.markdown("#### LLM Hallucination & Cost Telemetry")
            st.json(load_json(mon_path))
