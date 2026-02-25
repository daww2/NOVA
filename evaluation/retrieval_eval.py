"""
Retrieval Evaluation Engine.

Measures hybrid search quality by checking if ground truth answers
appear in retrieved chunk content.

Metrics: Recall@K, Precision@K, MRR, NDCG@10.
"""

import asyncio
import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

from src.config import settings
from src.core.embedding.generator import EmbeddingGenerator
from src.core.retrieval.bm25_search import BM25Search
from src.core.retrieval.hybrid_search import HybridSearch, HybridResult
from src.core.retrieval.vector_search import VectorSearch
from src.services.vector_store.qdrant import QdrantStore

logger = logging.getLogger(__name__)

QUERIES_PATH = Path(__file__).parent / "test_set" / "queries.json"


@dataclass
class QueryResult:
    """Result for a single query evaluation."""
    query_id: int
    query: str
    ground_truth: str
    relevant_ranks: list[int] = field(default_factory=list)
    num_retrieved: int = 0

    @property
    def is_hit(self) -> bool:
        return len(self.relevant_ranks) > 0


@dataclass
class EvalResult:
    """Aggregated evaluation results."""
    recall_at_k: dict[int, float] = field(default_factory=dict)
    precision_at_k: dict[int, float] = field(default_factory=dict)
    hit_rate: float = 0.0
    mrr: float = 0.0
    ndcg_at_10: float = 0.0
    total_queries: int = 0
    hit_count: int = 0
    query_results: list[QueryResult] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "",
            "=" * 55,
            "  Retrieval Evaluation Results",
            "=" * 55,
            f"  Total queries:    {self.total_queries}",
            f"  Hits (any rank):  {self.hit_count}/{self.total_queries}",
            f"  Hit Rate:         {self.hit_rate:.4f}",
            "-" * 55,
        ]
        for k in sorted(self.recall_at_k):
            lines.append(f"  Recall@{k:<5}      {self.recall_at_k[k]:.4f}")
        lines.append("-" * 55)
        for k in sorted(self.precision_at_k):
            lines.append(f"  Precision@{k:<3}     {self.precision_at_k[k]:.4f}")
        lines.append("-" * 55)
        lines.append(f"  MRR:              {self.mrr:.4f}")
        lines.append(f"  NDCG@10:          {self.ndcg_at_10:.4f}")
        lines.append("=" * 55)
        return "\n".join(lines)


class RetrievalEvaluator:
    """
    Evaluates hybrid search retrieval quality.

    Relevance is content-based: a retrieved chunk is relevant if
    the ground truth answer text appears within its content.
    """

    def __init__(
        self,
        hybrid_search: HybridSearch,
        embedding_generator: EmbeddingGenerator,
        queries_path: Path = QUERIES_PATH,
    ):
        self.hybrid_search = hybrid_search
        self.embedding_generator = embedding_generator
        self.queries_path = queries_path
        self._queries: list[dict] = []

    def _load_queries(self) -> list[dict]:
        if not self._queries:
            with open(self.queries_path, "r", encoding="utf-8") as f:
                self._queries = json.load(f)
        return self._queries

    @staticmethod
    def _is_relevant(chunk_content: str, ground_truth: str) -> bool:
        return ground_truth.lower() in chunk_content.lower()

    async def evaluate(self, top_k: int = 10) -> EvalResult:
        queries = self._load_queries()
        query_results: list[QueryResult] = []

        for i, q in enumerate(queries):
            qr = await self._evaluate_single(q, top_k)
            query_results.append(qr)
            if (i + 1) % 20 == 0:
                logger.info("Evaluated %d/%d queries", i + 1, len(queries))

        logger.info("Evaluated %d/%d queries", len(queries), len(queries))

        k_values = [1, 3, 5, 10]
        hit_count = sum(1 for qr in query_results if qr.is_hit)
        total = len(query_results)
        result = EvalResult(
            recall_at_k={k: self._compute_recall_at_k(query_results, k) for k in k_values},
            precision_at_k={k: self._compute_precision_at_k(query_results, k) for k in k_values},
            hit_rate=hit_count / total if total else 0.0,
            mrr=self._compute_mrr(query_results),
            ndcg_at_10=self._compute_ndcg_at_k(query_results, 10),
            total_queries=total,
            hit_count=hit_count,
            query_results=query_results,
        )
        return result

    async def _evaluate_single(self, query_data: dict, top_k: int) -> QueryResult:
        query = query_data["query"]
        ground_truth = str(query_data["ground_truth"])

        embedding = await self.embedding_generator.embed_query(query)
        results: list[HybridResult] = self.hybrid_search.search(
            query=query,
            query_embedding=embedding,
            top_k=top_k,
        )

        relevant_ranks = [
            rank
            for rank, r in enumerate(results, start=1)
            if self._is_relevant(r.content, ground_truth)
        ]

        return QueryResult(
            query_id=query_data["id"],
            query=query,
            ground_truth=ground_truth,
            relevant_ranks=relevant_ranks,
            num_retrieved=len(results),
        )

    @staticmethod
    def _compute_recall_at_k(results: list[QueryResult], k: int) -> float:
        """Fraction of queries where ground truth appears in top K."""
        if not results:
            return 0.0
        hits = sum(1 for qr in results if any(r <= k for r in qr.relevant_ranks))
        return hits / len(results)

    @staticmethod
    def _compute_precision_at_k(results: list[QueryResult], k: int) -> float:
        """Average fraction of top-K chunks that are relevant."""
        if not results:
            return 0.0
        precisions = []
        for qr in results:
            relevant_in_k = sum(1 for r in qr.relevant_ranks if r <= k)
            precisions.append(relevant_in_k / k)
        return sum(precisions) / len(precisions)

    @staticmethod
    def _compute_mrr(results: list[QueryResult]) -> float:
        """Mean Reciprocal Rank — average of 1/rank of first relevant chunk."""
        if not results:
            return 0.0
        rr_sum = 0.0
        for qr in results:
            if qr.relevant_ranks:
                rr_sum += 1.0 / min(qr.relevant_ranks)
        return rr_sum / len(results)

    @staticmethod
    def _compute_ndcg_at_k(results: list[QueryResult], k: int) -> float:
        """
        NDCG@K with binary relevance.

        DCG  = sum of 1/log2(rank+1) for relevant chunks in top K
        IDCG = 1/log2(2) = 1.0 (single relevant answer, best case rank=1)
        """
        if not results:
            return 0.0
        ndcg_sum = 0.0
        idcg = 1.0 / math.log2(2)  # ideal: one relevant doc at rank 1
        for qr in results:
            dcg = sum(
                1.0 / math.log2(r + 1)
                for r in qr.relevant_ranks
                if r <= k
            )
            ndcg_sum += dcg / idcg
        return ndcg_sum / len(results)


async def create_evaluator() -> RetrievalEvaluator:
    """Factory: connect to services and build evaluator."""
    # Qdrant store
    qdrant = QdrantStore(
        url=settings.qdrant.qdrant_url,
        api_key=settings.qdrant.qdrant_api_key,
        collection=settings.qdrant.qdrant_collection,
    )
    await qdrant.connect()

    # Fetch all docs for BM25 index
    all_docs = await qdrant.get_all_documents()
    logger.info("Loaded %d documents from Qdrant for BM25 indexing", len(all_docs))

    # BM25
    bm25 = BM25Search()
    bm25.index(all_docs)

    # Vector search
    vector_search = VectorSearch(
        collection_name=settings.qdrant.qdrant_collection,
        dimensions=settings.embedding.dimensions,
        url=settings.qdrant.qdrant_url,
        api_key=settings.qdrant.qdrant_api_key,
    )

    # Hybrid search
    hybrid = HybridSearch(vector_search=vector_search, bm25_search=bm25)

    # Embedding generator
    embedding_gen = EmbeddingGenerator(enable_cache=False)

    return RetrievalEvaluator(
        hybrid_search=hybrid,
        embedding_generator=embedding_gen,
    )
