FROM python:3.10-slim

WORKDIR /app
ENV TMPDIR=/var/tmp

COPY requirements.txt setup.py README.md ./
COPY src/ ./src/

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    qdrant-client rank-bm25 rouge-score nltk scikit-learn numpy pandas pyyaml \
    fastapi uvicorn pydantic jinja2 -e . && \
    rm -rf /var/tmp/*

COPY config/ ./config/
COPY params.yaml main.py app.py streamlit_app.py ./
COPY templates/ ./templates/
COPY artifacts/ ./artifacts/

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
