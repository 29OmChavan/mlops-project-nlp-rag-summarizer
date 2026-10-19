"""
CLI Entry Point for Project 03: NLP LoRA Fine-Tuning & Advanced Hybrid RAG Pipeline.
"""
from src.nlp_rag_system.logger import logger
from src.nlp_rag_system.pipeline.training_pipeline import TrainingPipeline


def main():
    logger.info("Starting Project 03 End-to-End NLP LoRA & Hybrid RAG Pipeline...")
    pipeline = TrainingPipeline()
    results = pipeline.run_pipeline()
    eval_rep = results["evaluation"]
    rag_rep = results["rag_benchmark"]
    logger.info(
        f"Pipeline Finished! LoRA ROUGE-L={eval_rep['lora_finetuned_metrics']['rougeL']} "
        f"(+{eval_rep['rougeL_improvement_percentage_points']}% pts vs Base) | "
        f"Hybrid RAG MRR={rag_rep['advanced_hybrid_rrf_rag']['mrr']}"
    )
    return results


if __name__ == "__main__":
    main()
