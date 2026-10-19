"""
Stage 06: Production Hybrid RAG Pipeline with Qdrant Vector DB, BM25, Reciprocal Rank Fusion (RRF),
Query Expansion, Metadata Filtering, Cross-Feature Re-Ranking, and Multi-Document Citation Tracking
(James Briggs Advanced RAG Architecture).
"""
import re
import sys
from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)

from src.nlp_rag_system.logger import logger
from src.nlp_rag_system.exception import CustomException
from src.nlp_rag_system.utils.common import load_json, save_json
from src.nlp_rag_system.entity.config_entity import RAGPipelineConfig


DOMAIN_ACRONYM_EXPANSIONS = {
    "lora": "low-rank adaptation rank decomposition matrices A and B",
    "qlora": "4-bit normalfloat quantized low-rank adaptation",
    "rrf": "reciprocal rank fusion hybrid dense bm25 search",
    "ks": "kolmogorov-smirnov ks_2samp covariate drift p-value",
    "dvc": "data version control dag dvc.yaml dvc.lock repro",
    "bpe": "byte-pair encoding utf-8 subword tokenizer vocabulary",
    "rouge": "recall-oriented understudy for gisting evaluation",
}


def tokenize_simple(text: str) -> List[str]:
    return re.findall(r"[a-z0-9_]+", text.lower())


class HybridRAGEngine:
    """
    Production Hybrid RAG Engine backed by Qdrant Vector Store + BM25 + Reciprocal Rank Fusion
    + Cross-Feature Re-Ranking + Multi-Document [DOC-ID] Citation Tracking.
    """

    def __init__(self, config: RAGPipelineConfig):
        self.config = config
        self.qdrant = QdrantClient(":memory:")
        self.chunks: List[Dict[str, Any]] = []
        self.bm25: Optional[BM25Okapi] = None
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.svd: Optional[TruncatedSVD] = None
        self._is_indexed = False

    def expand_query(self, query: str) -> str:
        """Expands domain acronyms in user queries to boost semantic and lexical recall."""
        tokens = tokenize_simple(query)
        expansions = [
            DOMAIN_ACRONYM_EXPANSIONS[t]
            for t in tokens
            if t in DOMAIN_ACRONYM_EXPANSIONS
        ]
        if expansions:
            return query + " " + " ".join(expansions)
        return query

    def _fit_dense_embedder(self, texts: List[str]) -> np.ndarray:
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        tfidf_mat = self.vectorizer.fit_transform(texts)
        n_comp = min(self.config.embedding_dim, tfidf_mat.shape[0] - 1, tfidf_mat.shape[1] - 1)
        if n_comp >= 2:
            self.svd = TruncatedSVD(n_components=n_comp, random_state=42)
            reduced = self.svd.fit_transform(tfidf_mat)
        else:
            reduced = tfidf_mat.toarray()

        # Pad to configured embedding_dim for Qdrant vector schema consistency
        padded = np.zeros((len(texts), self.config.embedding_dim), dtype=np.float32)
        padded[:, : reduced.shape[1]] = reduced
        return normalize(padded, norm="l2")

    def _embed_query(self, query: str) -> List[float]:
        assert self.vectorizer is not None
        tfidf_vec = self.vectorizer.transform([query])
        if self.svd is not None:
            reduced = self.svd.transform(tfidf_vec)
        else:
            reduced = tfidf_vec.toarray()
        padded = np.zeros((1, self.config.embedding_dim), dtype=np.float32)
        padded[:, : reduced.shape[1]] = reduced
        normed = normalize(padded, norm="l2")[0]
        return normed.tolist()

    def build_index(self) -> None:
        """Indexes knowledge base chunks into Qdrant vector store and BM25 lexical index."""
        try:
            kb_data = load_json(self.config.knowledge_base_path)
            self.chunks = kb_data.get("chunks", [])
            if not self.chunks:
                raise ValueError("Knowledge base chunks list is empty!")

            corpus_texts = [
                f"{c['title']} {c['category']} {c['chunk_text']}" for c in self.chunks
            ]
            dense_vectors = self._fit_dense_embedder(corpus_texts)

            # Initialize Qdrant Collection
            if self.qdrant.collection_exists(self.config.collection_name):
                self.qdrant.delete_collection(self.config.collection_name)

            self.qdrant.create_collection(
                collection_name=self.config.collection_name,
                vectors_config=VectorParams(
                    size=self.config.embedding_dim, distance=Distance.COSINE
                ),
            )

            points = [
                PointStruct(
                    id=idx,
                    vector=dense_vectors[idx].tolist(),
                    payload=self.chunks[idx],
                )
                for idx in range(len(self.chunks))
            ]
            self.qdrant.upsert(collection_name=self.config.collection_name, points=points)

            # Initialize BM25 Lexical Index
            tokenized_corpus = [tokenize_simple(t) for t in corpus_texts]
            self.bm25 = BM25Okapi(tokenized_corpus)
            self._is_indexed = True
            logger.info(
                f"Indexed {len(self.chunks)} chunks into Qdrant collection '{self.config.collection_name}' + BM25"
            )
        except Exception as e:
            raise CustomException(e, sys)

    def retrieve_naive_dense(
        self, query: str, top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """Naive RAG baseline: Dense Qdrant vector search only (no query expansion, no BM25, no re-ranking)."""
        if not self._is_indexed:
            self.build_index()
        q_vec = self._embed_query(query)
        hits = self.qdrant.query_points(
            collection_name=self.config.collection_name,
            query=q_vec,
            limit=top_k,
        ).points
        return [
            {
                **hit.payload,
                "score": round(float(hit.score), 4),
                "retrieval_mode": "naive_dense",
            }
            for hit in hits
        ]

    def retrieve_advanced_hybrid(
        self,
        query: str,
        top_k: int = 3,
        category_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Advanced RAG Pipeline:
        1. Query Expansion (domain acronym injection)
        2. Qdrant Dense Vector Search (with optional metadata filtering)
        3. BM25 Sparse Lexical Search
        4. Reciprocal Rank Fusion (RRF): score(d) = 1/(k + rank_dense) + 1/(k + rank_bm25)
        5. Cross-Feature Bigram Overlap Re-Ranking
        """
        if not self._is_indexed:
            self.build_index()
        assert self.bm25 is not None

        expanded_q = self.expand_query(query)
        q_vec = self._embed_query(expanded_q)

        qdrant_filter = None
        if category_filter:
            qdrant_filter = Filter(
                must=[
                    FieldCondition(
                        key="category", match=MatchValue(value=category_filter)
                    )
                ]
            )

        dense_hits = self.qdrant.query_points(
            collection_name=self.config.collection_name,
            query=q_vec,
            query_filter=qdrant_filter,
            limit=len(self.chunks),
        ).points
        dense_ranks = {hit.payload["doc_id"]: rank + 1 for rank, hit in enumerate(dense_hits)}
        dense_scores = {hit.payload["doc_id"]: float(hit.score) for hit in dense_hits}

        # BM25 lexical ranking
        q_tokens = tokenize_simple(expanded_q)
        bm25_raw = self.bm25.get_scores(q_tokens)
        bm25_order = np.argsort(bm25_raw)[::-1]
        bm25_ranks = {
            self.chunks[idx]["doc_id"]: rank + 1
            for rank, idx in enumerate(bm25_order)
        }

        # Reciprocal Rank Fusion + Cross-Feature Lexical Overlap Re-Ranking
        rrf_k = self.config.rrf_k
        q_token_set = set(tokenize_simple(query))
        candidates = []

        for chunk in self.chunks:
            if category_filter and chunk["category"] != category_filter:
                continue
            doc_id = chunk["doc_id"]
            r_dense = dense_ranks.get(doc_id, len(self.chunks) + 10)
            r_bm25 = bm25_ranks.get(doc_id, len(self.chunks) + 10)
            rrf_score = (1.0 / (rrf_k + r_dense)) + (1.0 / (rrf_k + r_bm25))

            doc_tokens = set(
                tokenize_simple(f"{chunk['title']} {chunk['chunk_text']}")
            )
            overlap = len(q_token_set & doc_tokens) / max(len(q_token_set), 1)
            rerank_score = (
                self.config.rerank_alpha * (rrf_score * 30.0 + dense_scores.get(doc_id, 0.0))
                + (1.0 - self.config.rerank_alpha) * overlap
            )

            candidates.append(
                {
                    **chunk,
                    "rrf_score": round(float(rrf_score), 5),
                    "rerank_score": round(float(rerank_score), 4),
                    "retrieval_mode": "advanced_hybrid_rrf_reranked",
                }
            )

        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        return candidates[:top_k]

    def answer_with_citations(
        self,
        query: str,
        top_k: int = 3,
        category_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Retrieves top-K chunks via Advanced Hybrid RAG and generates a grounded answer
        with inline [DOC-ID] citations and structured source attribution.
        """
        retrieved = self.retrieve_advanced_hybrid(
            query=query, top_k=top_k, category_filter=category_filter
        )
        if not retrieved:
            return {
                "query": query,
                "answer": "Insufficient grounded context in knowledge base to answer this query.",
                "citations": [],
                "retrieved_chunks": [],
            }

        top_chunk = retrieved[0]
        supporting = retrieved[1:2]

        citation_tags = [f"[{c['doc_id']}]" for c in retrieved]
        synthesized_answer = (
            f"Based on {top_chunk['title']} [{top_chunk['doc_id']}], "
            f"{top_chunk['chunk_text']}"
        )
        if supporting:
            sup = supporting[0]
            synthesized_answer += (
                f" Related reference: {sup['title']} [{sup['doc_id']}] ({sup['source']})."
            )

        return {
            "query": query,
            "expanded_query": self.expand_query(query),
            "answer": synthesized_answer,
            "citations": [
                {
                    "doc_id": c["doc_id"],
                    "title": c["title"],
                    "source": c["source"],
                    "category": c["category"],
                    "rerank_score": c["rerank_score"],
                }
                for c in retrieved
            ],
            "citation_tags": citation_tags,
            "retrieved_chunks": retrieved,
        }

    def run_retrieval_benchmark(self) -> Dict[str, Any]:
        """
        Benchmarks Naive Dense RAG vs. Advanced Hybrid Re-Ranked RAG on ground-truth queries:
        Computes Precision@K, Recall@K, HitRate@1, and Mean Reciprocal Rank (MRR).
        """
        logger.info("Running RAG Retrieval Benchmark (Naive Dense vs. Advanced Hybrid RRF)...")
        try:
            if not self._is_indexed:
                self.build_index()
            queries = load_json(self.config.rag_benchmark_path).get("queries", [])
            k = self.config.top_k

            def _score_method(mode: str) -> Dict[str, float]:
                recalls, precisions, mrrs, hit1s = [], [], [], []
                for q in queries:
                    expected = set(q["expected_doc_ids"])
                    if mode == "naive":
                        hits = self.retrieve_naive_dense(q["query"], top_k=k)
                    else:
                        hits = self.retrieve_advanced_hybrid(
                            q["query"],
                            top_k=k,
                            category_filter=q.get("category_filter"),
                        )
                    got_ids = [h["doc_id"] for h in hits]
                    relevant_retrieved = len(expected & set(got_ids))
                    recalls.append(relevant_retrieved / max(len(expected), 1))
                    precisions.append(relevant_retrieved / max(k, 1))
                    hit1s.append(1.0 if (got_ids and got_ids[0] in expected) else 0.0)

                    rr = 0.0
                    for idx, did in enumerate(got_ids):
                        if did in expected:
                            rr = 1.0 / (idx + 1)
                            break
                    mrrs.append(rr)

                n = max(len(queries), 1)
                return {
                    "hit_rate_at_1": round(sum(hit1s) / n, 4),
                    f"recall_at_{k}": round(sum(recalls) / n, 4),
                    f"precision_at_{k}": round(sum(precisions) / n, 4),
                    "mrr": round(sum(mrrs) / n, 4),
                }

            naive_metrics = _score_method("naive")
            hybrid_metrics = _score_method("hybrid")

            report = {
                "top_k": k,
                "num_benchmark_queries": len(queries),
                "naive_dense_rag": naive_metrics,
                "advanced_hybrid_rrf_rag": hybrid_metrics,
                "mrr_improvement": round(
                    hybrid_metrics["mrr"] - naive_metrics["mrr"], 4
                ),
            }
            save_json(self.config.rag_benchmark_report_path, report)
            logger.info(f"RAG Benchmark Complete: {report}")
            return report
        except Exception as e:
            raise CustomException(e, sys)
