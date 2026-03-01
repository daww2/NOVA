"""Prometheus metrics for the RAG pipeline.

All metrics are defined once here and imported wherever needed.
"""

from prometheus_client import Counter, Gauge, Histogram

# ---------------------------------------------------------------------------
# Histograms — latency distributions
# ---------------------------------------------------------------------------
REQUEST_DURATION = Histogram(
    "rag_request_duration_seconds",
    "End-to-end request latency",
    labelnames=["route"],
    buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
)

EMBEDDING_DURATION = Histogram(
    "rag_embedding_duration_seconds",
    "Embedding generation latency",
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
)

SEARCH_DURATION = Histogram(
    "rag_search_duration_seconds",
    "Search latency by type",
    labelnames=["type"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0),
)

LLM_DURATION = Histogram(
    "rag_llm_duration_seconds",
    "LLM generation latency",
    buckets=(0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
)

CACHE_LOOKUP_DURATION = Histogram(
    "rag_cache_lookup_duration_seconds",
    "Semantic cache lookup latency",
    buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5),
)

# ---------------------------------------------------------------------------
# Counters
# ---------------------------------------------------------------------------
REQUESTS_TOTAL = Counter(
    "rag_requests_total",
    "Total requests by route and status",
    labelnames=["route", "status"],
)

CACHE_HITS = Counter(
    "rag_cache_hits_total",
    "Cache hits by layer (exact / semantic)",
    labelnames=["layer"],
)

CACHE_MISSES = Counter(
    "rag_cache_misses_total",
    "Cache misses",
)

LLM_TOKENS = Counter(
    "rag_llm_tokens_total",
    "LLM token usage by type",
    labelnames=["type"],
)

ERRORS_TOTAL = Counter(
    "rag_errors_total",
    "Errors by component",
    labelnames=["component"],
)

LLM_COST = Counter(
    "rag_llm_cost_dollars_total",
    "Estimated LLM cost in USD",
)

# ---------------------------------------------------------------------------
# Gauges
# ---------------------------------------------------------------------------
CACHE_ENTRIES = Gauge(
    "rag_cache_entries",
    "Current number of semantic cache entries",
)

BM25_INDEX_SIZE = Gauge(
    "rag_bm25_index_size",
    "Number of documents in BM25 index",
)

ACTIVE_SESSIONS = Gauge(
    "rag_active_sessions",
    "Number of active conversation sessions",
)


class _Metrics:
    """Namespace object so callers can do ``METRICS.REQUEST_DURATION``."""
    REQUEST_DURATION = REQUEST_DURATION
    EMBEDDING_DURATION = EMBEDDING_DURATION
    SEARCH_DURATION = SEARCH_DURATION
    LLM_DURATION = LLM_DURATION
    CACHE_LOOKUP_DURATION = CACHE_LOOKUP_DURATION
    REQUESTS_TOTAL = REQUESTS_TOTAL
    CACHE_HITS = CACHE_HITS
    CACHE_MISSES = CACHE_MISSES
    LLM_TOKENS = LLM_TOKENS
    ERRORS_TOTAL = ERRORS_TOTAL
    LLM_COST = LLM_COST
    CACHE_ENTRIES = CACHE_ENTRIES
    BM25_INDEX_SIZE = BM25_INDEX_SIZE
    ACTIVE_SESSIONS = ACTIVE_SESSIONS


METRICS = _Metrics()
