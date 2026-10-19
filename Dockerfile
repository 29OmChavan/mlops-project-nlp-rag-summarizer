FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt setup.py README.md ./
COPY src/ ./src/

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY config/ ./config/
COPY params.yaml main.py app.py streamlit_app.py ./
COPY templates/ ./templates/
COPY artifacts/ ./artifacts/

# Pre-cache google/flan-t5-small tokenizer & model weights inside image so inference starts in <2s
RUN python -c "from transformers import AutoTokenizer, AutoModelForSeq2SeqLM; AutoTokenizer.from_pretrained('google/flan-t5-small'); AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-small')"
RUN if [ ! -f artifacts/model_evaluation/evaluation_report.json ]; then DISABLE_REMOTE_MLFLOW=1 DISABLE_HF_PUSH=1 python main.py; fi

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
