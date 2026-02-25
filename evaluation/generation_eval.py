"""
Generation Evaluation Engine.

End-to-end RAG evaluation using RAGAS metrics:
Faithfulness, Answer Relevancy, Context Precision,
Context Recall, Answer Correctness.
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path

from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    AnswerCorrectness,
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from src.config import settings
from src.core.embedding.generator import EmbeddingGenerator
from src.core.generation.context_builder import ContextBuilder
from src.core.generation.llm_client import LLMClient
from src.core.generation.prompt_manager import build_prompt
from src.core.retrieval.bm25_search import BM25Search
from src.core.retrieval.hybrid_search import HybridSearch
from src.core.retrieval.vector_search import VectorSearch
from src.services.vector_store.qdrant import QdrantStore

logger = logging.getLogger(__name__)

QUERIES_PATH = Path(__file__).parent / "test_set" / "queries.json"
RESULTS_DIR = Path(__file__).parent / "results"


class GenerationEvaluator:
    """
    Evaluates the full RAG pipeline (retrieval + generation) using RAGAS.

    For each query: embed -> hybrid search -> build context -> LLM generate.
    Then scores with RAGAS metrics.
    """

    def __init__(
        self,
        hybrid_search: HybridSearch,
        embedding_generator: EmbeddingGenerator,
        llm_client: LLMClient,
        context_builder: ContextBuilder,
        queries_path: Path = QUERIES_PATH,
    ):
        self.hybrid_search = hybrid_search
        self.embedding_generator = embedding_generator
        self.llm_client = llm_client
        self.context_builder = context_builder
        self.queries_path = queries_path
        self._queries: list[dict] = []

    def _load_queries(self) -> list[dict]:
        if not self._queries:
            with open(self.queries_path, "r", encoding="utf-8") as f:
                self._queries = json.load(f)
        return self._queries

    async def _run_rag_pipeline(self, query: str) -> tuple[str, list[str]]:
        """Run the full RAG pipeline for a single query.

        Returns:
            (answer, retrieved_contexts) where retrieved_contexts is a list
            of chunk content strings.
        """
        # 1. Embed query
        embedding = await self.embedding_generator.embed_query(query)

        # 2. Hybrid search
        results = self.hybrid_search.search(
            query=query,
            query_embedding=embedding,
            top_k=10,
        )

        # 3. Build context
        chunks = [
            {
                "content": r.content,
                "document_id": r.document_id,
                "chunk_id": r.chunk_id,
                "score": r.score,
            }
            for r in results
        ]
        context = self.context_builder.build(chunks)

        # 4. Generate answer
        system_prompt, user_prompt = build_prompt(query, context.text)
        response = await self.llm_client.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=settings.llm.temperature,
            max_tokens=settings.llm.max_tokens,
        )

        retrieved_contexts = [r.content for r in results]
        return response.content, retrieved_contexts

    async def evaluate(self) -> dict:
        """Run all queries through the RAG pipeline and evaluate with RAGAS.

        Returns:
            Dict with per-metric scores and per-query details.
        """
        queries = self._load_queries()
        samples: list[SingleTurnSample] = []

        for i, q in enumerate(queries):
            query_text = q["query"]
            ground_truth = str(q["ground_truth"])

            logger.info("Processing query %d/%d: %s", i + 1, len(queries), query_text)

            answer, retrieved_contexts = await self._run_rag_pipeline(query_text)

            sample = SingleTurnSample(
                user_input=query_text,
                response=answer,
                retrieved_contexts=retrieved_contexts,
                reference=ground_truth,
            )
            samples.append(sample)

        # Build RAGAS dataset
        dataset = EvaluationDataset(samples=samples)

        # RAGAS evaluator LLM & embeddings (uses same OpenAI key)
        evaluator_llm = LangchainLLMWrapper(ChatOpenAI(
            model="gpt-4o-mini",
            api_key=settings.llm.api_key,
            base_url=settings.llm.base_url,
        ))
        evaluator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(
            model=settings.embedding.model_name,
            api_key=settings.llm.api_key,
        ))

        metrics = [
            Faithfulness(),
            ResponseRelevancy(),
            LLMContextPrecisionWithReference(),
            LLMContextRecall(),
            AnswerCorrectness(),
        ]

        logger.info("Running RAGAS evaluation with %d samples...", len(samples))
        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
        )

        saved_path = self._save_results(result)
        logger.info("Results saved to %s", saved_path)

        self._print_results(result)
        return result

    @staticmethod
    def _save_results(result) -> Path:
        """Save evaluation results to JSON."""
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = RESULTS_DIR / f"generation_eval_{timestamp}.json"

        # Build serialisable dict
        aggregate = {k: v for k, v in result.items() if isinstance(v, (int, float))}

        # Clean NaN values from per-sample scores
        per_sample = []
        if hasattr(result, "scores") and result.scores:
            for s in result.scores:
                cleaned = {}
                for k, v in s.items():
                    if v is not None and v == v:  # excludes NaN
                        cleaned[k] = v
                    else:
                        cleaned[k] = None
                per_sample.append(cleaned)

        data = {
            "timestamp": timestamp,
            "aggregate": aggregate,
            "per_sample": per_sample,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        return path

    @staticmethod
    def load_latest_results() -> dict | None:
        """Load the most recent saved results, or None if no results exist."""
        if not RESULTS_DIR.exists():
            return None
        files = sorted(RESULTS_DIR.glob("generation_eval_*.json"), reverse=True)
        if not files:
            return None
        with open(files[0], "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def print_saved_results(data: dict) -> None:
        """Print results from a saved JSON dict."""
        print()
        print("=" * 55)
        print("  Generation Evaluation Results (RAGAS)")
        print(f"  Run: {data['timestamp']}")
        print("=" * 55)

        for metric_name, score in data["aggregate"].items():
            print(f"  {metric_name:<35} {score:.4f}")

        print("-" * 55)

        per_sample = data.get("per_sample", [])
        if per_sample:
            num_samples = len(per_sample)
            print(f"  Total samples evaluated: {num_samples}")

            metrics = list(per_sample[0].keys())
            for metric_name in metrics:
                failed = sum(1 for s in per_sample if s.get(metric_name) is None)
                if failed > 0:
                    print(f"  {metric_name}: {failed} failed evaluations (NaN)")

        print("=" * 55)
        print()

    @staticmethod
    def _print_results(result) -> None:
        """Print formatted results table from a RAGAS EvaluationResult."""
        print()
        print("=" * 55)
        print("  Generation Evaluation Results (RAGAS)")
        print("=" * 55)

        for metric_name, score in result.items():
            if isinstance(score, (int, float)):
                print(f"  {metric_name:<35} {score:.4f}")

        print("-" * 55)

        if hasattr(result, "scores") and result.scores:
            num_samples = len(result.scores)
            print(f"  Total samples evaluated: {num_samples}")

            for metric_name in result.scores[0]:
                valid = [s[metric_name] for s in result.scores
                         if s.get(metric_name) is not None
                         and s[metric_name] == s[metric_name]]
                failed = num_samples - len(valid)
                if failed > 0:
                    print(f"  {metric_name}: {failed} failed evaluations (NaN)")

        print("=" * 55)
        print()


async def create_evaluator() -> GenerationEvaluator:
    """Factory: connect to services and build generation evaluator."""
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

    # LLM client
    llm_client = LLMClient(
        model=settings.llm.model,
        api_key=settings.llm.api_key,
        base_url=settings.llm.base_url,
        max_retries=settings.llm.max_retries,
        timeout=settings.llm.request_timeout,
    )

    # Context builder
    context_builder = ContextBuilder()

    return GenerationEvaluator(
        hybrid_search=hybrid,
        embedding_generator=embedding_gen,
        llm_client=llm_client,
        context_builder=context_builder,
    )
