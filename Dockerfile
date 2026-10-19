FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt setup.py README.md ./
COPY src/ ./src/

# Step 1: Install lightweight CPU-only PyTorch (~170MB instead of 2.3GB CUDA wheels)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Step 2: Install NLP, PEFT/LoRA, Qdrant, BM25, and FastAPI serving dependencies
RUN pip install --no-cache-dir \
    transformers peft qdrant-client rank-bm25 scikit-learn numpy pandas pyyaml \
    fastapi uvicorn pydantic jinja2 -e .

COPY config/ ./config/
COPY params.yaml main.py app.py streamlit_app.py ./
COPY templates/ ./templates/
COPY artifacts/ ./artifacts/

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
