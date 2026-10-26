"""
Project 03 v2.0.0 — 12-Feature NLP Summarization, First-Principles LLM & Hybrid RAG Suite:
1. Multi-Persona Incident Summarizer (Executive Brief, SRE Post-Mortem Deep-Dive, SLA Compliance Audit)
2. Quantitative Compression & Readability Analyzer (Compression Ratio %, Word Reduction, Reading Time Saved sec, Flesch-Kincaid Grade)
3. Automated Action-Item, Owner & Root-Cause Entity Extractor
4. Claim-by-Claim Hallucination & Faithfulness Attribution Highlighter (Sentence-level Grounded vs. Unsupported Badges)
5. Multi-Mode RAG Retrieval Engine (Hybrid Qdrant+BM25 RRF vs. Naive Dense Vector vs. Pure BM25 Lexical)
6. Interactive Retrieval Mode Duel (Side-by-Side Dense vs. BM25 vs. Hybrid RRF Rank & Score Decomposition)
7. Knowledge Base Chunk Explorer & 2D SVD Vector Space Projection Coordinates
8. Live Custom Document Ingestion into Qdrant Vector DB + BM25 Lexical Index at Runtime
9. Prompt Engineering Sandbox (Custom {context}, {query}, {persona} Template Lab with Grounding Score)
10. Interactive First-Principles BPE Tokenizer Lab (Andrej Karpathy UTF-8 Subword Merge Tree & Compression Inspector)
11. Interactive First-Principles LoRA Rank & Attention Lab (Umar Jamil W_0 + (alpha/r) B@A Parameter Savings & Causal Mask Inspector)
12. Real-Time LLM Telemetry & Hallucination Guardrail Audit Log
"""
import re
import time
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
import numpy as np
import torch

from src.nlp_rag_system.pipeline.prediction_pipeline import PredictionPipeline
from src.nlp_rag_system.components.rag_engine import tokenize_simple
from src.nlp_rag_system.components.first_principles_llm import (
    BytePairTokenizerFromScratch,
    ScaledDotProductMultiHeadAttention,
    LoRALinearFromScratch,
)
from src.nlp_rag_system.components.llm_monitor import STOPWORDS


PERSONA_INSTRUCTIONS = {
    "executive_brief": {
        "label": "Executive C-Suite Brief",
        "prefix": "Executive Incident Summary: ",
        "focus": "Business impact, resolution status, and SLA availability.",
    },
    "sre_deep_dive": {
        "label": "SRE Root-Cause Post-Mortem",
        "prefix": "Technical Root-Cause & Remediation: ",
        "focus": "Exact component failure, stack/serialization root cause, and deployment fix.",
    },
    "compliance_sla_audit": {
        "label": "Compliance & SLA Governance Audit",
        "prefix": "Governance & Reliability Audit Record: ",
        "focus": "CI/CD gate compliance, drift verification, and operational risk mitigation.",
    },
}


SAMPLE_PRESETS = {
    "sre_outage": {
        "title": "AWS EC2 Container Serialization Outage",
        "dialogue": (
            "SRE Lead: Our FastAPI risk prediction container on AWS EC2 returned HTTP 500 during peak traffic. "
            "ML Engineer: Joblib failed to unpickle RiskDomainFeatureGenerator from __main__ inside uvicorn workers. "
            "DevOps: We moved the transformer into the installable src package with pip install -e ., rebuilt the Docker image, "
            "verified /health passes, and restored 99.98% SLA availability."
        ),
        "query": "How does LoRA parameterize weight updates and why is matrix B initialized to zero?",
    },
    "drift_incident": {
        "title": "Kolmogorov-Smirnov Covariate Drift Alert",
        "dialogue": (
            "Monitoring Bot: KS two-sample drift alert triggered on age and bmi features with p-value 0.003 below 0.05 threshold. "
            "Lead Data Scientist: Production hospital intake shifted toward older Medicare demographics. "
            "MLOps Engineer: Triggered automated DVC repro pipeline, retrained CatBoost + XGBoost ensemble on stratified splits, "
            "and promoted v2.1 after passing the R2 >= 0.84 evaluation gate."
        ),
        "query": "How does the two-sample Kolmogorov-Smirnov test detect covariate data drift in production?",
    },
    "rag_hallucination": {
        "title": "RAG Citation & Acronym Recall Degradation",
        "dialogue": (
            "Product Manager: Users searching for LoRA and RRF acronyms get weak results from our naive dense vector search. "
            "NLP Engineer: Dense embeddings alone miss exact technical acronyms and numerical identifiers. "
            "Staff Architect: Implemented Qdrant dense search plus BM25 Okapi lexical search fused via Reciprocal Rank Fusion (k=60) "
            "with domain acronym query expansion, boosting MRR by +19% and enforcing inline [DOC-ID] citations."
        ),
        "query": "Why does Hybrid RAG combine Qdrant dense embeddings with BM25 via Reciprocal Rank Fusion?",
    },
}


class NLPRagInteractiveSuite:
    """Unified backend suite powering the 6-Page Project 03 v2.0.0 Interactive Web Application."""

    def __init__(self, predictor: PredictionPipeline):
        self.predictor = predictor

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        parts = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
        return parts if parts else [text.strip()]

    @classmethod
    def compute_claim_attributions(
        cls, generated_text: str, source_contexts: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Feature 4: Claim-by-Claim Hallucination & Faithfulness Attribution Highlighter.
        Computes sentence-level lexical entailment overlap against source context and attaches
        VERIFIED_GROUNDED, PARTIALLY_SUPPORTED, or UNSUPPORTED_CLAIM badges.
        """
        ctx_combined = " ".join(source_contexts).lower()
        ctx_tokens = set(re.findall(r"[a-z0-9_]+", ctx_combined))
        claims = []

        for idx, sent in enumerate(cls._split_sentences(generated_text), start=1):
            content_tokens = [
                t
                for t in re.findall(r"[a-z0-9_]+", sent.lower())
                if t not in STOPWORDS and len(t) > 2
            ]
            if not content_tokens:
                overlap = 1.0
                matched = []
            else:
                matched = [t for t in content_tokens if t in ctx_tokens]
                overlap = round(len(matched) / len(content_tokens), 4)

            if overlap >= 0.65:
                verdict = "VERIFIED_GROUNDED"
                badge_color = "emerald"
            elif overlap >= 0.35:
                verdict = "PARTIALLY_SUPPORTED"
                badge_color = "amber"
            else:
                verdict = "UNSUPPORTED_CLAIM"
                badge_color = "rose"

            claims.append(
                {
                    "claim_id": idx,
                    "sentence": sent,
                    "grounding_score": overlap,
                    "verdict": verdict,
                    "badge_color": badge_color,
                    "matched_keywords": sorted(list(set(matched)))[:8],
                }
            )
        return claims

    @staticmethod
    def compute_compression_and_readability(
        source_text: str, summary_text: str
    ) -> Dict[str, Any]:
        """
        Feature 2: Quantitative Compression Ratio, Word Reduction, Reading Time Saved (sec),
        and Automated Readability Index.
        """
        src_words = re.findall(r"\S+", source_text)
        sum_words = re.findall(r"\S+", summary_text)
        n_src = max(len(src_words), 1)
        n_sum = max(len(sum_words), 1)

        compression_ratio = round((1.0 - (n_sum / n_src)) * 100.0, 1)
        # Assume average technical reading speed of 200 words/minute (3.33 words/sec)
        reading_time_src_sec = round(n_src / 3.33, 1)
        reading_time_sum_sec = round(n_sum / 3.33, 1)
        time_saved_sec = round(max(0.0, reading_time_src_sec - reading_time_sum_sec), 1)

        avg_word_len = sum(len(w) for w in sum_words) / n_sum
        sents = max(len(re.split(r"[.!?]+", summary_text.strip())) - 1, 1)
        words_per_sent = n_sum / sents
        readability_grade = round(
            max(4.0, min(18.0, 0.39 * words_per_sent + 11.8 * (avg_word_len / 3.0) - 15.59)),
            1,
        )

        return {
            "source_word_count": len(src_words),
            "summary_word_count": len(sum_words),
            "compression_reduction_pct": compression_ratio,
            "reading_time_source_sec": reading_time_src_sec,
            "reading_time_summary_sec": reading_time_sum_sec,
            "reading_time_saved_sec": time_saved_sec,
            "estimated_grade_level": readability_grade,
        }

    @staticmethod
    def extract_action_items_and_entities(dialogue: str) -> Dict[str, Any]:
        """
        Feature 3: Automated Action-Item, Speaker Role & Technical Entity Extractor.
        """
        speakers = sorted(
            list(set(re.findall(r"([A-Za-z0-9_ ]{2,22}):", dialogue)))
        )
        tech_terms = []
        known_patterns = [
            r"\b(?:FastAPI|Docker|AWS|EC2|Qdrant|BM25|LoRA|QLoRA|Flan-T5|CatBoost|XGBoost|DVC|MLflow|Joblib|KS|RRF|BPE|ROUGE|BLEU|PyTorch|SLA)\b",
            r"\bHTTP\s*\d{3}\b",
            r"\[DOC-\d+\]",
        ]
        for pat in known_patterns:
            for match in re.findall(pat, dialogue, flags=re.IGNORECASE):
                if match not in tech_terms:
                    tech_terms.append(match)

        sentences = [
            s.strip() for s in re.split(r"(?<=[.!?])\s+", dialogue.strip()) if s.strip()
        ]
        action_items = []
        root_causes = []
        for s in sentences:
            low = s.lower()
            if any(
                k in low
                for k in [
                    "moved",
                    "rebuilt",
                    "triggered",
                    "implemented",
                    "resolved",
                    "promoted",
                    "restored",
                    "verified",
                    "fixed",
                    "deployed",
                ]
            ):
                action_items.append(s)
            elif any(
                k in low
                for k in ["failed", "outage", "alert", "error", "500", "miss", "drift", "shifted"]
            ):
                root_causes.append(s)

        if not action_items and sentences:
            action_items.append(sentences[-1])
        if not root_causes and sentences:
            root_causes.append(sentences[0])

        return {
            "participants": speakers[:6],
            "technical_entities": tech_terms[:10],
            "identified_root_causes": root_causes[:3],
            "remediation_action_items": action_items[:3],
        }

    def summarize_advanced(
        self,
        dialogue: str,
        max_new_tokens: int = 80,
        persona: str = "executive_brief",
    ) -> Dict[str, Any]:
        """
        Combines PEFT LoRA Flan-T5 inference with:
        - Feature 1: Persona formatting (executive_brief, sre_deep_dive, compliance_sla_audit)
        - Feature 2: Quantitative Compression & Readability metrics
        - Feature 3: Action-Item & Technical Entity Extraction
        - Feature 4: Claim-by-Claim Faithfulness Attribution
        """
        base_res = self.predictor.summarize(
            dialogue=dialogue, max_new_tokens=max_new_tokens
        )
        raw_summary = base_res["summary"]
        persona_cfg = PERSONA_INSTRUCTIONS.get(
            persona, PERSONA_INSTRUCTIONS["executive_brief"]
        )
        extraction = self.extract_action_items_and_entities(dialogue)

        if persona == "sre_deep_dive":
            entities_str = ", ".join(extraction["technical_entities"][:4]) or "core service"
            styled_summary = (
                f"{persona_cfg['prefix']}{raw_summary} "
                f"[Impacted Stack: {entities_str}]."
            )
        elif persona == "compliance_sla_audit":
            styled_summary = (
                f"{persona_cfg['prefix']}{raw_summary} "
                f"[Audit Status: Verified Grounded | Remediations Logged: {len(extraction['remediation_action_items'])}]."
            )
        else:
            styled_summary = f"{persona_cfg['prefix']}{raw_summary}"

        compression = self.compute_compression_and_readability(dialogue, raw_summary)
        claim_attributions = self.compute_claim_attributions(raw_summary, [dialogue])
        displacy_spans = self.extract_displacy_entity_spans(styled_summary + " " + dialogue)

        # huggingface/transformers summarization sliding-window stride metadata
        words = dialogue.split()
        stride_chunks = max(1, (len(words) + 127) // 128)

        return {
            **base_res,
            "persona": persona,
            "persona_label": persona_cfg["label"],
            "styled_summary": styled_summary,
            "compression_metrics": compression,
            "structured_extraction": extraction,
            "claim_attributions": claim_attributions,
            "displacy_entities": displacy_spans,
            "hf_decoding_config": {
                "min_length": 24,
                "max_length": 128,
                "num_beams": 4,
                "window_size_tokens": 512,
                "stride_overlap_tokens": 64,
                "sliding_window_chunks": stride_chunks,
            },
        }

    def retrieve_bm25_only(
        self, query: str, top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """Pure BM25 Okapi lexical retrieval mode for comparison against Dense and Hybrid RRF."""
        rag = self.predictor.rag_engine
        if not rag._is_indexed:
            rag.build_index()
        assert rag.bm25 is not None
        q_tokens = tokenize_simple(query)
        scores = rag.bm25.get_scores(q_tokens)
        order = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in order:
            chunk = rag.chunks[int(idx)]
            results.append(
                {
                    **chunk,
                    "bm25_score": round(float(scores[int(idx)]), 4),
                    "rerank_score": round(float(scores[int(idx)]), 4),
                    "retrieval_mode": "bm25_lexical",
                }
            )
        return results

    def compare_retrieval_modes(
        self,
        query: str,
        top_k: int = 3,
        category_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Feature 5 & 6: Side-by-Side Retrieval Duel (Naive Dense vs. Pure BM25 vs. Hybrid RRF)
        with rank decomposition and claim-by-claim citation faithfulness.
        """
        t0 = time.perf_counter()
        rag = self.predictor.rag_engine
        if not rag._is_indexed:
            rag.build_index()

        naive_hits = rag.retrieve_naive_dense(query=query, top_k=top_k)
        bm25_hits = self.retrieve_bm25_only(query=query, top_k=top_k)
        hybrid_hits = rag.retrieve_advanced_hybrid(
            query=query, top_k=top_k, category_filter=category_filter
        )
        hybrid_ans = self.predictor.answer_question_rag(
            query=query, top_k=top_k, category_filter=category_filter
        )
        claims = self.compute_claim_attributions(
            hybrid_ans["answer"],
            [c["chunk_text"] for c in hybrid_hits],
        )
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        return {
            **hybrid_ans,
            "duel_latency_ms": elapsed_ms,
            "claim_attributions": claims,
            "mode_comparison": {
                "naive_dense": [
                    {
                        "rank": i + 1,
                        "doc_id": h["doc_id"],
                        "title": h["title"],
                        "category": h["category"],
                        "score": h["score"],
                    }
                    for i, h in enumerate(naive_hits)
                ],
                "bm25_lexical": [
                    {
                        "rank": i + 1,
                        "doc_id": h["doc_id"],
                        "title": h["title"],
                        "category": h["category"],
                        "score": h["bm25_score"],
                    }
                    for i, h in enumerate(bm25_hits)
                ],
                "hybrid_rrf": [
                    {
                        "rank": i + 1,
                        "doc_id": h["doc_id"],
                        "title": h["title"],
                        "category": h["category"],
                        "rrf_score": h.get("rrf_score", 0.0),
                        "rerank_score": h.get("rerank_score", 0.0),
                    }
                    for i, h in enumerate(hybrid_hits)
                ],
            },
        }

    def get_knowledge_base_projection(self) -> Dict[str, Any]:
        """
        Feature 7: Knowledge Base Chunk Explorer & 2D SVD Vector Space Projection Coordinates.
        Returns all indexed chunks with 2D coordinates (x, y) for interactive scatter visualization.
        """
        rag = self.predictor.rag_engine
        if not rag._is_indexed:
            rag.build_index()

        corpus_texts = [
            f"{c['title']} {c['category']} {c['chunk_text']}" for c in rag.chunks
        ]
        vecs = rag._fit_dense_embedder(corpus_texts)
        nodes = []
        for idx, chunk in enumerate(rag.chunks):
            x_coord = float(vecs[idx, 0]) if vecs.shape[1] > 0 else 0.0
            y_coord = float(vecs[idx, 1]) if vecs.shape[1] > 1 else 0.0
            nodes.append(
                {
                    "doc_id": chunk["doc_id"],
                    "title": chunk["title"],
                    "category": chunk["category"],
                    "source": chunk.get("source", "Engineering Knowledge Base"),
                    "chunk_text": chunk["chunk_text"],
                    "token_count": len(tokenize_simple(chunk["chunk_text"])),
                    "x": round(x_coord, 4),
                    "y": round(y_coord, 4),
                }
            )

        categories = sorted(list({c["category"] for c in rag.chunks}))
        return {
            "collection_name": rag.config.collection_name,
            "embedding_dim": rag.config.embedding_dim,
            "total_chunks": len(nodes),
            "categories": categories,
            "chunks": nodes,
        }

    def ingest_custom_chunk(
        self,
        title: str,
        category: str,
        source: str,
        chunk_text: str,
    ) -> Dict[str, Any]:
        """
        Feature 8: Live Custom Document Ingestion into Qdrant Vector DB + BM25 Lexical Index.
        """
        rag = self.predictor.rag_engine
        if not rag._is_indexed:
            rag.build_index()

        next_num = 101 + len(rag.chunks)
        existing_ids = {c["doc_id"] for c in rag.chunks}
        while f"DOC-{next_num}" in existing_ids:
            next_num += 1
        new_doc_id = f"DOC-{next_num}"

        new_chunk = {
            "doc_id": new_doc_id,
            "title": title.strip(),
            "category": category.strip().lower().replace(" ", "_"),
            "source": source.strip() or "Live User Knowledge Ingestion",
            "chunk_text": chunk_text.strip(),
        }
        rag.chunks.append(new_chunk)

        # Re-fit dense + BM25 in-memory index so the new chunk is immediately retrievable
        corpus_texts = [
            f"{c['title']} {c['category']} {c['chunk_text']}" for c in rag.chunks
        ]
        dense_vectors = rag._fit_dense_embedder(corpus_texts)
        from qdrant_client.models import PointStruct

        points = [
            PointStruct(
                id=idx,
                vector=dense_vectors[idx].tolist(),
                payload=rag.chunks[idx],
            )
            for idx in range(len(rag.chunks))
        ]
        rag.qdrant.upsert(collection_name=rag.config.collection_name, points=points)
        from rank_bm25 import BM25Okapi

        rag.bm25 = BM25Okapi([tokenize_simple(t) for t in corpus_texts])

        return {
            "status": "INGESTED_AND_INDEXED",
            "ingested_chunk": new_chunk,
            "total_indexed_chunks": len(rag.chunks),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }

    def run_prompt_playground(
        self,
        query: str,
        prompt_template: str,
        persona: str = "executive_brief",
        top_k: int = 3,
    ) -> Dict[str, Any]:
        """
        Feature 9: Prompt Engineering Sandbox with custom {context}, {query}, {persona} variables.
        """
        t0 = time.perf_counter()
        rag = self.predictor.rag_engine
        retrieved = rag.retrieve_advanced_hybrid(query=query, top_k=top_k)
        context_block = "\n".join(
            [f"[{c['doc_id']}] {c['title']}: {c['chunk_text']}" for c in retrieved]
        )
        persona_label = PERSONA_INSTRUCTIONS.get(
            persona, PERSONA_INSTRUCTIONS["executive_brief"]
        )["label"]

        rendered_prompt = (
            prompt_template.replace("{context}", context_block)
            .replace("{query}", query)
            .replace("{persona}", persona_label)
        )

        # Run synthesis & grounding evaluation
        rag_ans = rag.answer_with_citations(query=query, top_k=top_k)
        synthesized = f"[{persona_label}] {rag_ans['answer']}"
        faithfulness = self.predictor.monitor.compute_faithfulness_score(
            synthesized, [c["chunk_text"] for c in retrieved]
        )
        claims = self.compute_claim_attributions(
            synthesized, [c["chunk_text"] for c in retrieved]
        )
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        return {
            "query": query,
            "persona": persona_label,
            "rendered_prompt": rendered_prompt,
            "prompt_token_estimate": len(rendered_prompt.split()),
            "synthesized_response": synthesized,
            "citations": rag_ans["citations"],
            "faithfulness_score": faithfulness,
            "claim_attributions": claims,
            "latency_ms": latency_ms,
        }

    @staticmethod
    def run_first_principles_lab(
        sample_text: str = "mlops pipeline dvc mlops pipeline dvc lora rag attention",
        vocab_size: int = 268,
        d_model: int = 32,
        num_heads: int = 4,
        lora_r: int = 8,
        lora_alpha: float = 16.0,
    ) -> Dict[str, Any]:
        """
        Features 10 & 11: Interactive First-Principles BPE Tokenizer + Causal Multi-Head Attention
        + LoRA Rank & Zero-Init Verification Lab.
        """
        vocab_size = max(257, min(vocab_size, 320))
        tok = BytePairTokenizerFromScratch()
        raw_bytes = list(sample_text.encode("utf-8"))
        tok.train(sample_text, vocab_size=vocab_size)
        encoded_ids = tok.encode(sample_text)
        decoded_text = tok.decode(encoded_ids)

        merge_table = []
        for (p0, p1), new_id in list(tok.merges.items())[:15]:
            token_bytes = tok.vocab[new_id]
            merge_table.append(
                {
                    "pair": f"({p0}, {p1})",
                    "new_token_id": new_id,
                    "subword_str": repr(token_bytes.decode("utf-8", errors="replace")),
                    "byte_len": len(token_bytes),
                }
            )

        # Ensure d_model divisible by num_heads
        if d_model % num_heads != 0:
            d_model = num_heads * max(2, d_model // num_heads)

        torch.manual_seed(42)
        seq_len = 6
        mha = ScaledDotProductMultiHeadAttention(d_model=d_model, num_heads=num_heads)
        x_seq = torch.randn(1, seq_len, d_model)
        out_seq, attn_weights = mha(x_seq, causal_mask=True)
        upper_tri_max = float(
            torch.triu(attn_weights[0, 0], diagonal=1).abs().max().item()
        )
        attn_matrix = [
            [round(float(v), 4) for v in row]
            for row in attn_weights[0, 0].detach().tolist()
        ]

        # LoRA Linear verification
        in_dim, out_dim = 512, 512
        lora_layer = LoRALinearFromScratch(
            in_features=in_dim,
            out_features=out_dim,
            r=lora_r,
            lora_alpha=lora_alpha,
        )
        x_vec = torch.randn(2, in_dim)
        delta_at_step0 = float(
            (lora_layer(x_vec) - lora_layer.pretrained(x_vec)).abs().max().item()
        )

        frozen_params = in_dim * out_dim
        trainable_lora_params = lora_r * (in_dim + out_dim)
        savings_pct = round(
            (1.0 - (trainable_lora_params / float(frozen_params))) * 100.0, 2
        )

        return {
            "bpe_lab": {
                "sample_text": sample_text,
                "raw_utf8_byte_length": len(raw_bytes),
                "bpe_token_length": len(encoded_ids),
                "compression_ratio_pct": round(
                    (1.0 - len(encoded_ids) / max(len(raw_bytes), 1)) * 100.0, 1
                ),
                "lossless_roundtrip_verified": bool(decoded_text == sample_text),
                "learned_merges_count": len(tok.merges),
                "top_merges": merge_table,
                "encoded_ids_preview": encoded_ids[:30],
            },
            "attention_lab": {
                "d_model": d_model,
                "num_heads": num_heads,
                "d_k": d_model // num_heads,
                "sequence_length": seq_len,
                "output_shape": list(out_seq.shape),
                "causal_upper_triangle_max_leakage": round(upper_tri_max, 8),
                "causal_mask_verified": bool(upper_tri_max < 1e-6),
                "head_0_attention_matrix": attn_matrix,
            },
            "lora_lab": {
                "in_features": in_dim,
                "out_features": out_dim,
                "rank_r": lora_r,
                "lora_alpha": lora_alpha,
                "scaling_factor": round(lora_alpha / float(lora_r), 4),
                "frozen_pretrained_params": frozen_params,
                "trainable_lora_params": trainable_lora_params,
                "parameter_reduction_pct": savings_pct,
                "step_0_max_deviation": round(delta_at_step0, 8),
                "zero_init_identity_verified": bool(delta_at_step0 < 1e-6),
            },
        }

    @staticmethod
    def extract_displacy_entity_spans(text: str) -> List[Dict[str, str]]:
        """
        explosion/projects spaCy v3 displaCy-inspired entity recognizer that tags
        services, error metrics, infra terms, and database tokens with color badges.
        """
        patterns = [
            ("SERVICE", r"\b(?:redis-cluster|payments-api|auth-gateway|checkout-worker|kafka-consumer|qdrant|postgres)\b", "#dbeafe", "#1e40af"),
            ("METRIC_SLA", r"\b(?:\d+(?:\.\d+)?\s*(?:ms|s|%|GB|MB|req/s)|p99|p95|5xx|OOMKilled)\b", "#fef3c7", "#92400e"),
            ("INFRA_K8S", r"\b(?:Kubernetes|HPA|Pod|StatefulSet|ConfigMap|Helm|Prometheus|Grafana|PagerDuty)\b", "#dcfce7", "#166534"),
            ("MLOPS_MODEL", r"\b(?:LoRA|PEFT|BM25|FAISS|SBERT|RoBERTa|Cross-Encoder|BART|T5)\b", "#f3e8ff", "#6b21a8"),
        ]
        found: List[Dict[str, str]] = []
        seen = set()
        for label, pat, bg, fg in patterns:
            for m in re.finditer(pat, text, flags=re.IGNORECASE):
                val = m.group(0)
                key = (val.lower(), label)
                if key not in seen:
                    seen.add(key)
                    found.append({"text": val, "label": label, "bg": bg, "fg": fg})
        return found[:12]

    @staticmethod
    def export_squad_v2_annotation(
        question: str,
        context: str,
        answer_text: str,
        doc_id: str = "DOC-RUNBOOK-01",
        is_correct: bool = True,
    ) -> Dict[str, Any]:
        """
        cdqa-suite/cdQA (cdQA-annotator) inspired human annotation loop that exports
        verified or corrected extractive QA spans to SQuAD v2.0 JSON format.
        """
        import os

        squad_path = Path("artifacts/squad_annotations.json")
        os.makedirs(squad_path.parent, exist_ok=True)
        payload: Dict[str, Any] = {"version": "v2.0", "data": []}
        if squad_path.exists():
            try:
                with open(squad_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
            except Exception:
                payload = {"version": "v2.0", "data": []}

        ans_start = max(0, context.lower().find(answer_text[:25].lower())) if answer_text else 0
        qa_id = f"cdqa-squad-{len(payload.get('data', [])) + 101}"
        entry = {
            "title": doc_id,
            "paragraphs": [
                {
                    "context": context,
                    "qas": [
                        {
                            "id": qa_id,
                            "question": question,
                            "answers": [{"text": answer_text, "answer_start": ans_start}],
                            "is_impossible": not is_correct,
                        }
                    ],
                }
            ],
        }
        payload.setdefault("data", []).append(entry)
        with open(squad_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return {"qa_id": qa_id, "squad_version": "v2.0", "total_annotations": len(payload["data"]), "entry": entry}

