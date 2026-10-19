"""
ConfigurationManager for Project 03 NLP LoRA Fine-Tuning & Hybrid RAG System.
"""
from pathlib import Path
from src.nlp_rag_system.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from src.nlp_rag_system.utils.common import read_yaml, create_directories
from src.nlp_rag_system.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig,
    RAGPipelineConfig,
)


class ConfigurationManager:
    def __init__(
        self,
        config_filepath: Path = CONFIG_FILE_PATH,
        params_filepath: Path = PARAMS_FILE_PATH,
    ):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        create_directories([self.config["artifacts_root"]])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        cfg = self.config["data_ingestion"]
        rag_p = self.params["RAG"]
        create_directories([cfg["root_dir"]])
        return DataIngestionConfig(
            root_dir=Path(cfg["root_dir"]),
            summarization_train_path=Path(cfg["summarization_train_path"]),
            summarization_val_path=Path(cfg["summarization_val_path"]),
            knowledge_base_path=Path(cfg["knowledge_base_path"]),
            rag_benchmark_path=Path(cfg["rag_benchmark_path"]),
            chunk_size=int(rag_p["chunk_size"]),
            chunk_overlap=int(rag_p["chunk_overlap"]),
        )

    def get_data_validation_config(self) -> DataValidationConfig:
        cfg = self.config["data_validation"]
        ing = self.config["data_ingestion"]
        create_directories([cfg["root_dir"]])
        return DataValidationConfig(
            root_dir=Path(cfg["root_dir"]),
            summarization_train_path=Path(ing["summarization_train_path"]),
            summarization_val_path=Path(ing["summarization_val_path"]),
            knowledge_base_path=Path(ing["knowledge_base_path"]),
            status_file=Path(cfg["status_file"]),
            dataset_card_file=Path(cfg["dataset_card_file"]),
        )

    def get_data_transformation_config(self) -> DataTransformationConfig:
        cfg = self.config["data_transformation"]
        ing = self.config["data_ingestion"]
        tok_p = self.params["Tokenization"]
        create_directories([cfg["root_dir"]])
        return DataTransformationConfig(
            root_dir=Path(cfg["root_dir"]),
            summarization_train_path=Path(ing["summarization_train_path"]),
            summarization_val_path=Path(ing["summarization_val_path"]),
            model_ckpt=str(cfg["model_ckpt"]),
            tokenized_stats_path=Path(cfg["tokenized_stats_path"]),
            max_input_length=int(tok_p["max_input_length"]),
            max_target_length=int(tok_p["max_target_length"]),
        )

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        cfg = self.config["model_trainer"]
        ing = self.config["data_ingestion"]
        tok_p = self.params["Tokenization"]
        lora_p = self.params["LoRA"]
        train_p = self.params["TrainingArguments"]
        create_directories([cfg["root_dir"]])
        return ModelTrainerConfig(
            root_dir=Path(cfg["root_dir"]),
            summarization_train_path=Path(ing["summarization_train_path"]),
            summarization_val_path=Path(ing["summarization_val_path"]),
            model_ckpt=str(cfg["model_ckpt"]),
            lora_adapter_dir=Path(cfg["lora_adapter_dir"]),
            training_metrics_path=Path(cfg["training_metrics_path"]),
            hf_repo_id=str(cfg["hf_repo_id"]),
            max_input_length=int(tok_p["max_input_length"]),
            max_target_length=int(tok_p["max_target_length"]),
            lora_r=int(lora_p["r"]),
            lora_alpha=int(lora_p["lora_alpha"]),
            lora_dropout=float(lora_p["lora_dropout"]),
            target_modules=list(lora_p["target_modules"]),
            epochs=int(train_p["epochs"]),
            batch_size=int(train_p["batch_size"]),
            learning_rate=float(train_p["learning_rate"]),
            weight_decay=float(train_p["weight_decay"]),
        )

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        cfg = self.config["model_evaluation"]
        ing = self.config["data_ingestion"]
        tr = self.config["model_trainer"]
        tok_p = self.params["Tokenization"]
        create_directories([cfg["root_dir"]])
        return ModelEvaluationConfig(
            root_dir=Path(cfg["root_dir"]),
            summarization_val_path=Path(ing["summarization_val_path"]),
            model_ckpt=str(tr["model_ckpt"]),
            lora_adapter_dir=Path(tr["lora_adapter_dir"]),
            evaluation_report_path=Path(cfg["evaluation_report_path"]),
            mlflow_uri=str(cfg["mlflow_uri"]),
            experiment_name=str(cfg["experiment_name"]),
            min_rouge_l_improvement_pct=float(cfg["min_rouge_l_improvement_pct"]),
            max_input_length=int(tok_p["max_input_length"]),
            max_target_length=int(tok_p["max_target_length"]),
        )

    def get_rag_pipeline_config(self) -> RAGPipelineConfig:
        cfg = self.config["rag_pipeline"]
        ing = self.config["data_ingestion"]
        rag_p = self.params["RAG"]
        create_directories([cfg["root_dir"]])
        return RAGPipelineConfig(
            root_dir=Path(cfg["root_dir"]),
            knowledge_base_path=Path(ing["knowledge_base_path"]),
            rag_benchmark_path=Path(ing["rag_benchmark_path"]),
            collection_name=str(cfg["collection_name"]),
            embedding_dim=int(cfg["embedding_dim"]),
            top_k=int(cfg["top_k"]),
            rrf_k=int(rag_p["rrf_k"]),
            rerank_alpha=float(rag_p["rerank_alpha"]),
            rag_benchmark_report_path=Path(cfg["rag_benchmark_report_path"]),
        )
