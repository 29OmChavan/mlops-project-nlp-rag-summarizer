"""
End-to-End NLP LoRA Fine-Tuning & Hybrid RAG Training Pipeline Orchestrator.
"""
import sys
from typing import Dict, Any
from src.nlp_rag_system.logger import logger
from src.nlp_rag_system.exception import CustomException
from src.nlp_rag_system.config.configuration import ConfigurationManager
from src.nlp_rag_system.components.data_ingestion import DataIngestion
from src.nlp_rag_system.components.data_validation import DataValidation
from src.nlp_rag_system.components.data_transformation import DataTransformation
from src.nlp_rag_system.components.model_trainer import ModelTrainer
from src.nlp_rag_system.components.model_evaluation import ModelEvaluation
from src.nlp_rag_system.components.rag_engine import HybridRAGEngine


class TrainingPipeline:
    """Runs the 6-stage NLP LoRA & Hybrid RAG MLOps pipeline."""

    def __init__(self):
        self.cm = ConfigurationManager()

    def run_pipeline(self) -> Dict[str, Any]:
        try:
            logger.info("=" * 72)
            logger.info(">>>>> STAGE 01: DATA INGESTION & CHUNKING STARTED <<<<<")
            ing_res = DataIngestion(self.cm.get_data_ingestion_config()).initiate_data_ingestion()

            logger.info(">>>>> STAGE 02: DATA VALIDATION & DATASET CARD STARTED <<<<<")
            val_res = DataValidation(self.cm.get_data_validation_config()).initiate_data_validation()
            if not val_res["validation_passed"]:
                raise ValueError(f"Data Validation failed: {val_res['anomalies']}")

            logger.info(">>>>> STAGE 03: TOKENIZATION & LENGTH VALIDATION STARTED <<<<<")
            tok_res = DataTransformation(
                self.cm.get_data_transformation_config()
            ).initiate_data_transformation()

            logger.info(">>>>> STAGE 04: PEFT LORA FINE-TUNING & HF HUB PUSH STARTED <<<<<")
            train_res = ModelTrainer(self.cm.get_model_trainer_config()).train()

            logger.info(">>>>> STAGE 05: ROUGE/BLEU EVALUATION & CI/CD GATE STARTED <<<<<")
            eval_res = ModelEvaluation(self.cm.get_model_evaluation_config()).evaluate()

            logger.info(">>>>> STAGE 06: HYBRID RAG BENCHMARK (QDRANT + BM25 RRF) STARTED <<<<<")
            rag_engine = HybridRAGEngine(self.cm.get_rag_pipeline_config())
            rag_res = rag_engine.run_retrieval_benchmark()
            logger.info("=" * 72)

            return {
                "ingestion": ing_res,
                "validation": val_res,
                "tokenization": tok_res,
                "lora_training": train_res,
                "evaluation": eval_res,
                "rag_benchmark": rag_res,
            }
        except Exception as e:
            raise CustomException(e, sys)
