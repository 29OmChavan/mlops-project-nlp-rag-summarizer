"""
Project 03 Scaffolding Script (Krish Naik NLP Summarization + James Briggs RAG + Umar Jamil LoRA Pattern)
Creates the modular directory structure for the NLP LoRA Fine-Tuning & Advanced RAG system.
"""
import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="[%(asctime)s]: %(message)s")

project_name = "nlp_rag_system"

list_of_files = [
    ".github/workflows/main.yaml",
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/components/first_principles_llm.py",
    f"src/{project_name}/components/data_ingestion.py",
    f"src/{project_name}/components/data_validation.py",
    f"src/{project_name}/components/data_transformation.py",
    f"src/{project_name}/components/model_trainer.py",
    f"src/{project_name}/components/model_evaluation.py",
    f"src/{project_name}/components/rag_engine.py",
    f"src/{project_name}/components/llm_monitor.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/utils/common.py",
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/configuration.py",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/pipeline/training_pipeline.py",
    f"src/{project_name}/pipeline/prediction_pipeline.py",
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/entity/config_entity.py",
    f"src/{project_name}/constants/__init__.py",
    f"src/{project_name}/logger.py",
    f"src/{project_name}/exception.py",
    "config/config.yaml",
    "params.yaml",
    "main.py",
    "app.py",
    "streamlit_app.py",
    "Dockerfile",
    "docker-compose.yml",
    ".dockerignore",
    ".gitignore",
    "requirements.txt",
    "setup.py",
    "templates/index.html",
    "notebooks/01_attention_bpe_lora_from_scratch.py",
    "tests/__init__.py",
    "tests/test_nlp_rag_pipeline.py",
    "docs/01_DATA_AND_TOKENIZATION_SPEC.md",
    "docs/02_LORA_AND_RAG_EXPERIMENT_LOG.md",
    "docs/03_LLMOPS_DEPLOYMENT_AND_MONITORING_RUNBOOK.md",
    "docs/IMPLEMENTATION_GUIDE.md",
    "CHANGELOG.md",
    "README.md",
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for the file: {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w", encoding="utf-8") as f:
            pass
        logging.info(f"Creating empty file: {filepath}")
    else:
        logging.info(f"{filename} already exists")
