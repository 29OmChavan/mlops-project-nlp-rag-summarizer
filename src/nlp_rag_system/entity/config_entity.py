"""
Immutable Configuration Dataclasses for Project 03 NLP LoRA & Hybrid RAG Pipeline.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    summarization_train_path: Path
    summarization_val_path: Path
    knowledge_base_path: Path
    rag_benchmark_path: Path
    chunk_size: int
    chunk_overlap: int


@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    summarization_train_path: Path
    summarization_val_path: Path
    knowledge_base_path: Path
    status_file: Path
    dataset_card_file: Path


@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    summarization_train_path: Path
    summarization_val_path: Path
    model_ckpt: str
    tokenized_stats_path: Path
    max_input_length: int
    max_target_length: int


@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    summarization_train_path: Path
    summarization_val_path: Path
    model_ckpt: str
    lora_adapter_dir: Path
    training_metrics_path: Path
    hf_repo_id: str
    max_input_length: int
    max_target_length: int
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    target_modules: List[str]
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float


@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    summarization_val_path: Path
    model_ckpt: str
    lora_adapter_dir: Path
    evaluation_report_path: Path
    mlflow_uri: str
    experiment_name: str
    min_rouge_l_improvement_pct: float
    max_input_length: int
    max_target_length: int


@dataclass(frozen=True)
class RAGPipelineConfig:
    root_dir: Path
    knowledge_base_path: Path
    rag_benchmark_path: Path
    collection_name: str
    embedding_dim: int
    top_k: int
    rrf_k: int
    rerank_alpha: float
    rag_benchmark_report_path: Path
