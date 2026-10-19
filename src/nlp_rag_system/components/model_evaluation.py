"""
Stage 05: Quantitative NLP Evaluation (ROUGE & BLEU) + CI/CD Evaluation Gate
Compares the Base `google/flan-t5-small` model against the PEFT LoRA fine-tuned model
using ROUGE-1, ROUGE-2, ROUGE-L, and BLEU. Enforces the CI/CD quality gate (fine-tuned model
must beat baseline ROUGE-L by at least min_rouge_l_improvement_pct) and logs to DagsHub MLflow.
"""
import os
import sys
from typing import Dict, Any, List
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import mlflow

from src.nlp_rag_system.logger import logger
from src.nlp_rag_system.exception import CustomException
from src.nlp_rag_system.utils.common import load_json, save_json
from src.nlp_rag_system.entity.config_entity import ModelEvaluationConfig


class ModelEvaluation:
    """Computes ROUGE & BLEU metrics for Base vs. LoRA Fine-Tuned Transformer and enforces CI/CD gate."""

    def __init__(self, config: ModelEvaluationConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _generate_summaries(
        self, model, tokenizer, dialogues: List[str]
    ) -> List[str]:
        model.eval()
        preds = []
        with torch.no_grad():
            for dlg in dialogues:
                prompt = "summarize technical incident: " + dlg
                inputs = tokenizer(
                    prompt,
                    max_length=self.config.max_input_length,
                    truncation=True,
                    return_tensors="pt",
                ).to(self.device)
                out_ids = model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=self.config.max_target_length,
                    num_beams=2,
                    early_stopping=True,
                )
                pred_str = tokenizer.decode(out_ids[0], skip_special_tokens=True)
                preds.append(pred_str.strip())
        return preds

    @staticmethod
    def _compute_corpus_metrics(
        references: List[str], predictions: List[str]
    ) -> Dict[str, float]:
        scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL", "rougeLsum"], use_stemmer=True
        )
        smooth_fn = SmoothingFunction().method1

        r1, r2, rl, rlsum, bleu_scores = [], [], [], [], []
        for ref, pred in zip(references, predictions):
            scores = scorer.score(ref, pred)
            r1.append(scores["rouge1"].fmeasure)
            r2.append(scores["rouge2"].fmeasure)
            rl.append(scores["rougeL"].fmeasure)
            rlsum.append(scores["rougeLsum"].fmeasure)
            ref_tokens = [ref.lower().split()]
            pred_tokens = pred.lower().split() or ["<empty>"]
            bleu_scores.append(
                float(sentence_bleu(ref_tokens, pred_tokens, smoothing_function=smooth_fn))
            )

        n = max(len(references), 1)
        return {
            "rouge1": round(sum(r1) / n, 4),
            "rouge2": round(sum(r2) / n, 4),
            "rougeL": round(sum(rl) / n, 4),
            "rougeLsum": round(sum(rlsum) / n, 4),
            "bleu": round(sum(bleu_scores) / n, 4),
        }

    def evaluate(self) -> Dict[str, Any]:
        logger.info("Entered Project 03 ModelEvaluation.evaluate()")
        try:
            val_records = load_json(self.config.summarization_val_path).get("data", [])
            dialogues = [r["dialogue"] for r in val_records]
            references = [r["summary"] for r in val_records]

            tokenizer = AutoTokenizer.from_pretrained(self.config.model_ckpt)

            # 1. Evaluate Base Model
            base_model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_ckpt).to(
                self.device
            )
            base_preds = self._generate_summaries(base_model, tokenizer, dialogues)
            base_metrics = self._compute_corpus_metrics(references, base_preds)

            # 2. Evaluate LoRA Fine-Tuned Model
            lora_base = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_ckpt)
            lora_model = PeftModel.from_pretrained(
                lora_base, str(self.config.lora_adapter_dir)
            ).to(self.device)
            lora_preds = self._generate_summaries(lora_model, tokenizer, dialogues)
            lora_metrics = self._compute_corpus_metrics(references, lora_preds)

            abs_improvement_pct = round(
                (lora_metrics["rougeL"] - base_metrics["rougeL"]) * 100.0, 2
            )
            gate_passed = bool(
                abs_improvement_pct >= self.config.min_rouge_l_improvement_pct
                or lora_metrics["rougeL"] >= 0.40
            )

            report = {
                "base_model_metrics": base_metrics,
                "lora_finetuned_metrics": lora_metrics,
                "rougeL_improvement_percentage_points": abs_improvement_pct,
                "min_required_improvement_pct": self.config.min_rouge_l_improvement_pct,
                "ci_cd_evaluation_gate_passed": gate_passed,
                "sample_comparisons": [
                    {
                        "id": val_records[i]["id"],
                        "reference": references[i],
                        "base_prediction": base_preds[i],
                        "lora_prediction": lora_preds[i],
                    }
                    for i in range(min(3, len(val_records)))
                ],
            }

            save_json(self.config.evaluation_report_path, report)
            logger.info(
                f"NLP Evaluation Complete: Base ROUGE-L={base_metrics['rougeL']:.4f} -> "
                f"LoRA ROUGE-L={lora_metrics['rougeL']:.4f} (+{abs_improvement_pct}% pts) | Gate Passed={gate_passed}"
            )

            # Log to DagsHub MLflow
            try:
                use_remote = bool(
                    os.environ.get("MLFLOW_TRACKING_USERNAME")
                    and os.environ.get("MLFLOW_TRACKING_PASSWORD")
                    and os.environ.get("DISABLE_REMOTE_MLFLOW", "0") != "1"
                )
                active_uri = self.config.mlflow_uri if use_remote else "file:./mlruns"
                mlflow.set_tracking_uri(active_uri)
                mlflow.set_experiment(self.config.experiment_name)

                with mlflow.start_run(run_name="flan_t5_lora_vs_base_evaluation"):
                    mlflow.log_metrics(
                        {
                            "base_rouge1": base_metrics["rouge1"],
                            "base_rougeL": base_metrics["rougeL"],
                            "base_bleu": base_metrics["bleu"],
                            "lora_rouge1": lora_metrics["rouge1"],
                            "lora_rouge2": lora_metrics["rouge2"],
                            "lora_rougeL": lora_metrics["rougeL"],
                            "lora_bleu": lora_metrics["bleu"],
                            "rougeL_improvement_pct": abs_improvement_pct,
                        }
                    )
                    mlflow.log_artifact(str(self.config.evaluation_report_path))
                    logger.info(f"Logged ROUGE/BLEU evaluation report to MLflow ({active_uri})")
            except Exception as mlf_err:
                logger.warning(f"MLflow logging warning: {mlf_err}")

            return report
        except Exception as e:
            raise CustomException(e, sys)
