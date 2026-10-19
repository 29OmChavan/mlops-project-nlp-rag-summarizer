"""
Stage 03: Tokenization & Sequence Length Validation Component
Tokenizes dialogues and target summaries with Hugging Face AutoTokenizer (google/flan-t5-small)
and logs subword length statistics.
"""
import sys
from typing import Dict, Any
from transformers import AutoTokenizer
from src.nlp_rag_system.logger import logger
from src.nlp_rag_system.exception import CustomException
from src.nlp_rag_system.utils.common import load_json, save_json
from src.nlp_rag_system.entity.config_entity import DataTransformationConfig


class DataTransformation:
    """Validates subword tokenization and sequence truncation bounds using AutoTokenizer."""

    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def initiate_data_transformation(self) -> Dict[str, Any]:
        logger.info("Entered Project 03 DataTransformation.initiate_data_transformation()")
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.config.model_ckpt)
            val_records = load_json(self.config.summarization_val_path).get("data", [])

            input_lengths = []
            target_lengths = []

            for rec in val_records:
                prompt = "summarize technical incident: " + rec["dialogue"]
                in_ids = tokenizer.encode(
                    prompt, truncation=True, max_length=self.config.max_input_length
                )
                tgt_ids = tokenizer.encode(
                    rec["summary"], truncation=True, max_length=self.config.max_target_length
                )
                input_lengths.append(len(in_ids))
                target_lengths.append(len(tgt_ids))

            stats = {
                "tokenizer_class": type(tokenizer).__name__,
                "model_ckpt": self.config.model_ckpt,
                "vocab_size": int(tokenizer.vocab_size),
                "max_input_tokens_observed": int(max(input_lengths)),
                "avg_input_tokens": round(sum(input_lengths) / max(len(input_lengths), 1), 1),
                "max_target_tokens_observed": int(max(target_lengths)),
                "avg_target_tokens": round(sum(target_lengths) / max(len(target_lengths), 1), 1),
                "configured_max_input_length": self.config.max_input_length,
                "configured_max_target_length": self.config.max_target_length,
            }
            save_json(self.config.tokenized_stats_path, stats)
            logger.info(f"Tokenization statistics saved: {stats}")
            return stats
        except Exception as e:
            raise CustomException(e, sys)
