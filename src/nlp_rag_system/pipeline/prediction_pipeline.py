"""
Production Inference Pipeline for LoRA Summarization & Citation-Grounded Hybrid RAG QA.
"""
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

from src.nlp_rag_system.logger import logger
from src.nlp_rag_system.exception import CustomException
from src.nlp_rag_system.config.configuration import ConfigurationManager
from src.nlp_rag_system.components.rag_engine import HybridRAGEngine
from src.nlp_rag_system.components.llm_monitor import LLMMonitor


class PredictionPipeline:
    """Provides low-latency LoRA summarization and Citation-Backed Hybrid RAG QA with LLM Monitoring."""

    def __init__(self):
        self.cm = ConfigurationManager()
        self.eval_cfg = self.cm.get_model_evaluation_config()
        self.rag_cfg = self.cm.get_rag_pipeline_config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = None
        self.model = None
        self.rag_engine = HybridRAGEngine(self.rag_cfg)
        self.monitor = LLMMonitor()

    def _ensure_model_loaded(self) -> None:
        if self.model is not None and self.tokenizer is not None:
            return
        self.tokenizer = AutoTokenizer.from_pretrained(self.eval_cfg.model_ckpt)
        base_model = AutoModelForSeq2SeqLM.from_pretrained(self.eval_cfg.model_ckpt)
        if Path(self.eval_cfg.lora_adapter_dir).exists():
            self.model = PeftModel.from_pretrained(
                base_model, str(self.eval_cfg.lora_adapter_dir)
            ).to(self.device)
            logger.info(f"Loaded fine-tuned LoRA adapter from {self.eval_cfg.lora_adapter_dir}")
        else:
            self.model = base_model.to(self.device)
            logger.warning("LoRA adapter not found; using base model.")
        self.model.eval()

    def summarize(self, dialogue: str, max_new_tokens: int = 80) -> Dict[str, Any]:
        try:
            t0 = time.perf_counter()
            self._ensure_model_loaded()
            prompt = "summarize technical incident: " + dialogue.strip()
            inputs = self.tokenizer(
                prompt,
                max_length=self.eval_cfg.max_input_length,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                out_ids = self.model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=max_new_tokens,
                    num_beams=2,
                    early_stopping=True,
                )
            summary = self.tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()
            latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

            telemetry = self.monitor.record_interaction(
                endpoint="/summarize",
                prompt=prompt,
                response=summary,
                source_contexts=[dialogue],
                latency_ms=latency_ms,
            )

            return {
                "summary": summary,
                "latency_ms": latency_ms,
                "faithfulness_score": telemetry["faithfulness_score"],
                "hallucination_alert": telemetry["hallucination_alert"],
                "model": f"{self.eval_cfg.model_ckpt} + PEFT-LoRA",
            }
        except Exception as e:
            raise CustomException(e, sys)

    def answer_question_rag(
        self,
        query: str,
        top_k: int = 3,
        category_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        try:
            t0 = time.perf_counter()
            rag_result = self.rag_engine.answer_with_citations(
                query=query, top_k=top_k, category_filter=category_filter
            )
            latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

            context_texts = [c["chunk_text"] for c in rag_result["retrieved_chunks"]]
            telemetry = self.monitor.record_interaction(
                endpoint="/rag-qa",
                prompt=query,
                response=rag_result["answer"],
                source_contexts=context_texts,
                latency_ms=latency_ms,
            )

            return {
                "query": rag_result["query"],
                "expanded_query": rag_result["expanded_query"],
                "answer": rag_result["answer"],
                "citations": rag_result["citations"],
                "citation_tags": rag_result["citation_tags"],
                "latency_ms": latency_ms,
                "faithfulness_score": telemetry["faithfulness_score"],
                "hallucination_alert": telemetry["hallucination_alert"],
            }
        except Exception as e:
            raise CustomException(e, sys)
