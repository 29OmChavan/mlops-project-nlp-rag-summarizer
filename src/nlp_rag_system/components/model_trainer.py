"""
Stage 04: Transformer LoRA Fine-Tuning Component (PEFT + Hugging Face Hub)
Fine-tunes `google/flan-t5-small` on technical dialogue summarization using Low-Rank Adaptation (LoRA),
tracks parameter reduction (~0.45% trainable), saves adapter weights locally, and pushes to Hugging Face Hub.
"""
import os
import sys
import time
from typing import Dict, Any, List
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import LoraConfig, get_peft_model, TaskType

from src.nlp_rag_system.logger import logger
from src.nlp_rag_system.exception import CustomException
from src.nlp_rag_system.utils.common import load_json, save_json
from src.nlp_rag_system.entity.config_entity import ModelTrainerConfig


class ModelTrainer:
    """Fine-tunes Seq2Seq Transformer with PEFT LoRA adapters and pushes to Hugging Face Hub."""

    def __init__(self, config: ModelTrainerConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(self) -> Dict[str, Any]:
        logger.info("Entered Project 03 ModelTrainer.train() [PEFT LoRA Fine-Tuning]")
        try:
            torch.manual_seed(42)
            tokenizer = AutoTokenizer.from_pretrained(self.config.model_ckpt)
            base_model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_ckpt)

            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                target_modules=self.config.target_modules,
                lora_dropout=self.config.lora_dropout,
                bias="none",
                task_type=TaskType.SEQ_2_SEQ_LM,
            )

            peft_model = get_peft_model(base_model, lora_config)
            peft_model.to(self.device)

            total_params = sum(p.numel() for p in peft_model.parameters())
            trainable_params = sum(
                p.numel() for p in peft_model.parameters() if p.requires_grad
            )
            trainable_pct = round(100.0 * trainable_params / max(total_params, 1), 4)
            logger.info(
                f"LoRA Model Initialized: Total Params={total_params:,} | "
                f"Trainable LoRA Params={trainable_params:,} ({trainable_pct}%)"
            )

            train_records = load_json(self.config.summarization_train_path).get("data", [])
            val_records = load_json(self.config.summarization_val_path).get("data", [])
            all_train_samples = train_records + val_records

            optimizer = torch.optim.AdamW(
                [p for p in peft_model.parameters() if p.requires_grad],
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )

            epoch_losses: List[float] = []
            t0 = time.time()
            peft_model.train()

            for epoch in range(1, self.config.epochs + 1):
                running_loss = 0.0
                steps = 0
                for i in range(0, len(all_train_samples), self.config.batch_size):
                    batch = all_train_samples[i:i + self.config.batch_size]
                    inputs = [
                        "summarize technical incident: " + item["dialogue"]
                        for item in batch
                    ]
                    targets = [item["summary"] for item in batch]

                    model_inputs = tokenizer(
                        inputs,
                        max_length=self.config.max_input_length,
                        padding=True,
                        truncation=True,
                        return_tensors="pt",
                    ).to(self.device)

                    labels = tokenizer(
                        text_target=targets,
                        max_length=self.config.max_target_length,
                        padding=True,
                        truncation=True,
                        return_tensors="pt",
                    ).input_ids.to(self.device)

                    # Replace pad_token_id with -100 so CrossEntropyLoss ignores padding tokens
                    labels[labels == tokenizer.pad_token_id] = -100

                    optimizer.zero_grad()
                    outputs = peft_model(
                        input_ids=model_inputs.input_ids,
                        attention_mask=model_inputs.attention_mask,
                        labels=labels,
                    )
                    loss = outputs.loss
                    loss.backward()
                    optimizer.step()

                    running_loss += float(loss.item())
                    steps += 1

                avg_loss = round(running_loss / max(steps, 1), 4)
                epoch_losses.append(avg_loss)
                logger.info(f"LoRA Fine-Tuning Epoch [{epoch}/{self.config.epochs}] Loss: {avg_loss:.4f}")

            duration_sec = round(time.time() - t0, 2)

            # Save LoRA adapter & tokenizer locally
            os.makedirs(self.config.lora_adapter_dir, exist_ok=True)
            peft_model.save_pretrained(str(self.config.lora_adapter_dir))
            tokenizer.save_pretrained(str(self.config.lora_adapter_dir))
            logger.info(f"Saved LoRA adapter weights to {self.config.lora_adapter_dir}")

            # Optional push to Hugging Face Hub if HF_TOKEN is provided
            hf_token = os.environ.get("HF_TOKEN")
            pushed_to_hf = False
            if hf_token and os.environ.get("DISABLE_HF_PUSH", "0") != "1":
                try:
                    peft_model.push_to_hub(
                        self.config.hf_repo_id, token=hf_token, private=False
                    )
                    tokenizer.push_to_hub(
                        self.config.hf_repo_id, token=hf_token, private=False
                    )
                    pushed_to_hf = True
                    logger.info(
                        f"Successfully pushed LoRA adapter to Hugging Face Hub: https://huggingface.co/{self.config.hf_repo_id}"
                    )
                except Exception as hf_err:
                    logger.warning(f"Hugging Face Hub push skipped: {hf_err}")

            metrics = {
                "base_model": self.config.model_ckpt,
                "lora_r": self.config.lora_r,
                "lora_alpha": self.config.lora_alpha,
                "target_modules": self.config.target_modules,
                "total_parameters": total_params,
                "trainable_lora_parameters": trainable_params,
                "trainable_percentage": trainable_pct,
                "epochs": self.config.epochs,
                "epoch_losses": epoch_losses,
                "final_train_loss": epoch_losses[-1] if epoch_losses else 0.0,
                "training_duration_sec": duration_sec,
                "pushed_to_huggingface_hub": pushed_to_hf,
                "hf_repo_id": self.config.hf_repo_id,
            }
            save_json(self.config.training_metrics_path, metrics)
            return metrics

        except Exception as e:
            raise CustomException(e, sys)
