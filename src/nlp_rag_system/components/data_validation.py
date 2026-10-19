"""
Stage 02: Data Validation & Dataset Card Component
Validates sequence non-emptiness, schema keys, and writes a Hugging Face-compatible dataset_card.json.
"""
import sys
from typing import Dict, Any
from src.nlp_rag_system.logger import logger
from src.nlp_rag_system.exception import CustomException
from src.nlp_rag_system.utils.common import load_json, save_json
from src.nlp_rag_system.entity.config_entity import DataValidationConfig


class DataValidation:
    """Validates summarization and RAG corpora and generates a Dataset Card."""

    def __init__(self, config: DataValidationConfig):
        self.config = config

    def initiate_data_validation(self) -> Dict[str, Any]:
        logger.info("Entered Project 03 DataValidation.initiate_data_validation()")
        try:
            train_data = load_json(self.config.summarization_train_path).get("data", [])
            val_data = load_json(self.config.summarization_val_path).get("data", [])
            kb_chunks = load_json(self.config.knowledge_base_path).get("chunks", [])

            required_sum_keys = {"id", "domain", "dialogue", "summary"}
            required_kb_keys = {"doc_id", "title", "category", "source", "chunk_text"}

            validation_status = True
            anomalies = []

            if not train_data or not val_data or not kb_chunks:
                validation_status = False
                anomalies.append("One or more ingested datasets are empty.")

            for idx, row in enumerate(val_data):
                missing = required_sum_keys - set(row.keys())
                if missing or len(row.get("dialogue", "").strip()) < 20:
                    validation_status = False
                    anomalies.append(f"Invalid summarization record at index {idx}")

            for idx, chunk in enumerate(kb_chunks):
                missing = required_kb_keys - set(chunk.keys())
                if missing or len(chunk.get("chunk_text", "").strip()) < 30:
                    validation_status = False
                    anomalies.append(f"Invalid knowledge base chunk at index {idx}")

            avg_dialogue_words = sum(len(r["dialogue"].split()) for r in val_data) / max(
                len(val_data), 1
            )
            avg_summary_words = sum(len(r["summary"].split()) for r in val_data) / max(
                len(val_data), 1
            )

            dataset_card = {
                "dataset_name": "MLOps Technical Incident & Architecture Summarization + RAG Corpus",
                "validation_passed": validation_status,
                "anomalies": anomalies,
                "summarization_stats": {
                    "train_count": len(train_data),
                    "val_count": len(val_data),
                    "avg_dialogue_words": round(avg_dialogue_words, 1),
                    "avg_summary_words": round(avg_summary_words, 1),
                    "compression_ratio": round(avg_summary_words / max(avg_dialogue_words, 1), 3),
                },
                "rag_knowledge_base_stats": {
                    "total_chunks": len(kb_chunks),
                    "categories": sorted(list({c["category"] for c in kb_chunks})),
                },
            }

            save_json(self.config.dataset_card_file, dataset_card)
            with open(self.config.status_file, "w", encoding="utf-8") as f:
                f.write(f"Validation status: {validation_status}\n")

            logger.info(f"Data Validation complete. Status={validation_status}")
            return dataset_card
        except Exception as e:
            raise CustomException(e, sys)
