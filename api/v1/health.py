"""Health check, stats, and Prometheus metrics endpoints."""

import logging

from fastapi import APIRouter, Depends
from fastapi.responses import PlainTextResponse
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from src.config import settings
from src.services.vector_store.qdrant import QdrantStore
from src.core.retrieval.bm25_search import BM25Search
from src.core.memory.conversation import ConversationMemory
from src.core.caching.semantic_cache import SemanticCache
from src.core.observability.metrics import METRICS
from src.core.observability.tracing import langfuse_client

from api.v1.schemas import HealthResponse, StatsResponse
from api.v1.dependencies import (
    get_qdrant_store,
    get_bm25_search,
    get_conversation_memory,
    get_semantic_cache,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Basic health check."""
    return HealthResponse(
        status="healthy",
        version=settings.app_version,
        environment=settings.APP_ENV,
    )


@router.get("/health/stats")
async def health_stats(
    qdrant_store: QdrantStore = Depends(get_qdrant_store),
    bm25_search: BM25Search = Depends(get_bm25_search),
    memory: ConversationMemory = Depends(get_conversation_memory),
    semantic_cache: SemanticCache | None = Depends(get_semantic_cache),
):
    """Detailed system statistics."""
    qdrant_stats = qdrant_store.stats()

    # Update gauges on each stats call
    bm25_size = bm25_search.size
    session_count = memory.get_session_count()
    METRICS.BM25_INDEX_SIZE.set(bm25_size)
    METRICS.ACTIVE_SESSIONS.set(session_count)

    result = {
        "qdrant_points": qdrant_stats.get("points_count", 0),
        "qdrant_status": qdrant_stats.get("status", "unknown"),
        "bm25_index_size": bm25_size,
        "active_sessions": session_count,
        "langfuse_enabled": langfuse_client is not None,
    }

    if semantic_cache:
        cache_stats = semantic_cache.get_stats()
        METRICS.CACHE_ENTRIES.set(cache_stats.get("cache_size", 0))
        result["cache"] = cache_stats

    return result


@router.get("/metrics")
async def prometheus_metrics(
    bm25_search: BM25Search = Depends(get_bm25_search),
    memory: ConversationMemory = Depends(get_conversation_memory),
    semantic_cache: SemanticCache | None = Depends(get_semantic_cache),
):
    """Expose Prometheus metrics for scraping."""
    # Refresh gauges before scrape
    METRICS.BM25_INDEX_SIZE.set(bm25_search.size)
    METRICS.ACTIVE_SESSIONS.set(memory.get_session_count())
    if semantic_cache:
        METRICS.CACHE_ENTRIES.set(len(semantic_cache._memory_cache))

    return PlainTextResponse(
        content=generate_latest().decode("utf-8"),
        media_type=CONTENT_TYPE_LATEST,
    )
