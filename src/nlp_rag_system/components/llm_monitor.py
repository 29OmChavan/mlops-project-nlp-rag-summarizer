"""
Stage 07: Production LLM Monitoring & Hallucination Detection Component
Logs every prompt, response, token count, latency, estimated cost, and Faithfulness Grounding Score
(measuring content-word & bigram overlap against source context). Raises quality alerts on hallucination risk.
"""
import os
import re
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List

from src.nlp_rag_system.logger import logger
from src.nlp_rag_system.utils.common import save_json

STOPWORDS = {
    "the", "a", "an", "and", "or", "in", "on", "of", "to", "for", "with", "by", "is",
    "are", "was", "were", "be", "been", "this", "that", "it", "as", "at", "from", "based",
    "related", "reference",
}


class LLMMonitor:
    """Monitors LLM latency, token cost, and grounded faithfulness / hallucination rate."""

    def __init__(
        self,
        telemetry_log_path: Path = Path("artifacts/monitoring/llm_telemetry_logs.jsonl"),
        summary_report_path: Path = Path("artifacts/monitoring/monitoring_summary.json"),
    ):
        self.telemetry_log_path = telemetry_log_path
        self.summary_report_path = summary_report_path
        os.makedirs(os.path.dirname(self.telemetry_log_path), exist_ok=True)

    @staticmethod
    def compute_faithfulness_score(answer: str, context_texts: List[str]) -> float:
        """
        Computes lexical content-token grounding overlap between the generated response
        and the source context (retrieved RAG chunks or input dialogue):
            Faithfulness = |Tokens_content(Answer) intersect Tokens_content(Context)| / |Tokens_content(Answer)|
        """
        ans_tokens = [
            t
            for t in re.findall(r"[a-z0-9_]+", answer.lower())
            if t not in STOPWORDS and len(t) > 2
        ]
        if not ans_tokens:
            return 1.0
        ctx_combined = " ".join(context_texts).lower()
        ctx_tokens = set(re.findall(r"[a-z0-9_]+", ctx_combined))
        grounded = sum(1 for t in ans_tokens if t in ctx_tokens)
        return round(float(grounded) / len(ans_tokens), 4)

    def record_interaction(
        self,
        endpoint: str,
        prompt: str,
        response: str,
        source_contexts: List[str],
        latency_ms: float,
    ) -> Dict[str, Any]:
        prompt_tokens = len(prompt.split())
        response_tokens = len(response.split())
        total_tokens = prompt_tokens + response_tokens
        # Standard enterprise self-hosted/API cost estimation ($0.0005 / 1K tokens)
        cost_usd = round(total_tokens * 0.0000005, 8)

        faithfulness = self.compute_faithfulness_score(response, source_contexts)
        hallucination_alert = bool(faithfulness < 0.45)

        if hallucination_alert:
            logger.warning(
                f"[LLM MONITOR ALERT] Potential hallucination detected on {endpoint}: "
                f"faithfulness={faithfulness:.4f} (< 0.45 threshold)"
            )

        event = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "endpoint": endpoint,
            "prompt_tokens": prompt_tokens,
            "response_tokens": response_tokens,
            "total_tokens": total_tokens,
            "latency_ms": round(latency_ms, 2),
            "estimated_cost_usd": cost_usd,
            "faithfulness_score": faithfulness,
            "hallucination_alert": hallucination_alert,
        }

        with open(self.telemetry_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")

        self._update_summary()
        return event

    def _update_summary(self) -> Dict[str, Any]:
        if not self.telemetry_log_path.exists():
            return {}
        events = []
        with open(self.telemetry_log_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    events.append(json.loads(line))
        if not events:
            return {}

        n = len(events)
        summary = {
            "total_requests_logged": n,
            "avg_latency_ms": round(sum(e["latency_ms"] for e in events) / n, 2),
            "avg_faithfulness_score": round(
                sum(e["faithfulness_score"] for e in events) / n, 4
            ),
            "hallucination_alert_count": sum(
                1 for e in events if e["hallucination_alert"]
            ),
            "total_tokens_processed": sum(e["total_tokens"] for e in events),
            "total_estimated_cost_usd": round(
                sum(e["estimated_cost_usd"] for e in events), 7
            ),
        }
        save_json(self.summary_report_path, summary)
        return summary
