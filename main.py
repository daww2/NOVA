"""FastAPI application entry point for RAG Knowledge Assistant."""

import logging
import os
import sys
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Ensure project root is on sys.path so `from src.` and `from api.` imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import settings
from src.services.vector_store.qdrant import QdrantStore
from src.core.embedding.generator import create_embedding_generator
from src.core.retrieval.vector_search import VectorSearch
from src.core.retrieval.bm25_search import BM25Search
from src.core.retrieval.hybrid_search import create_hybrid_search
from src.core.generation.llm_client import create_llm_client
from src.core.generation.context_builder import create_context_builder
from src.core.query.classifier import create_classifier
from src.core.chunking.strategies import get_chunker
from src.services.document_processor import DocumentProcessor
from src.core.memory.conversation import ConversationMemory
from src.core.caching.semantic_cache import SemanticCache
from src.core.observability.tracing import langfuse_client, shutdown_langfuse
from src.core.observability.metrics import METRICS

from api.v1 import v1_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
# Only set YOUR app loggers to DEBUG, not third-party libs
if settings.debug:
    logging.getLogger("src").setLevel(logging.DEBUG)
    logging.getLogger("api").setLevel(logging.DEBUG)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: create all services. Shutdown: close connections."""
    logger.info("Starting %s v%s (%s)", settings.app_name, settings.app_version, settings.APP_ENV)

    # --- Langfuse ---
    if langfuse_client:
        logger.info("Langfuse tracing active")

    # --- Shared Redis Client (single connection pool for all caches) ---
    redis_client = None
    if settings.cache.redis_url:
        try:
            import redis
            redis_client = redis.from_url(settings.cache.redis_url, decode_responses=True)
            redis_client.ping()
            logger.info("Shared Redis client connected")
        except Exception as e:
            logger.warning("Redis unavailable (%s) — caches will use RAM", e)
            redis_client = None

    # --- Qdrant Store ---
    qdrant_store = QdrantStore(
        url=settings.qdrant.qdrant_url,
        api_key=settings.qdrant.qdrant_api_key,
        collection=settings.qdrant.qdrant_collection,
        dimension=settings.embedding.dimensions,
    )
    await qdrant_store.connect()

    # --- Embedding Generator (shares Redis client) ---
    embedding_generator = create_embedding_generator(redis_client=redis_client)

    # --- Vector Search (reuse Qdrant client — shared connection pool) ---
    vector_search = VectorSearch(
        collection_name=settings.qdrant.qdrant_collection,
        dimensions=settings.embedding.dimensions,
        client=qdrant_store.client,
    )

    # --- BM25 Search ---
    bm25_search = BM25Search()

    # Build BM25 index from existing Qdrant documents
    all_docs = await qdrant_store.get_all_documents()
    if all_docs:
        chunks_for_bm25 = [
            {
                "chunk_id": doc["chunk_id"],
                "content": doc["content"],
                "document_id": doc["document_id"],
                "metadata": doc["metadata"],
            }
            for doc in all_docs
        ]
        bm25_search.index(chunks_for_bm25)
        logger.info("BM25 index built with %d documents", len(chunks_for_bm25))
    else:
        logger.info("No existing documents — BM25 index is empty")

    # --- Hybrid Search ---
    hybrid_search = create_hybrid_search(vector_search, bm25_search)

    # --- LLM Client ---
    llm_client = create_llm_client(
        model=settings.llm.model,
        api_key=settings.llm.api_key,
        base_url=settings.llm.base_url,
        timeout=settings.llm.request_timeout,
        max_retries=settings.llm.max_retries,
    )

    # --- Context Builder ---
    context_builder = create_context_builder()

    # --- Query Classifier ---
    query_classifier = create_classifier()

    # --- Chunker ---
    chunker = get_chunker()

    # --- Document Processor ---
    document_processor = DocumentProcessor()

    # --- Conversation Memory ---
    conversation_memory = ConversationMemory()

    # --- Semantic Cache (shares embedding generator's OpenAI connection + Redis client) ---
    semantic_cache = None
    if settings.cache.semantic_cache_enabled:
        semantic_cache = SemanticCache(
            embedding_generator=embedding_generator,
            redis_client=redis_client,
        )
        logger.info("Semantic cache enabled")

    # --- Upload directory ---
    upload_dir = str(settings.data_dir / "uploads")
    os.makedirs(upload_dir, exist_ok=True)

    # Store all services on app.state
    app.state.qdrant_store = qdrant_store
    app.state.embedding_generator = embedding_generator
    app.state.bm25_search = bm25_search
    app.state.hybrid_search = hybrid_search
    app.state.llm_client = llm_client
    app.state.context_builder = context_builder
    app.state.query_classifier = query_classifier
    app.state.chunker = chunker
    app.state.document_processor = document_processor
    app.state.conversation_memory = conversation_memory
    app.state.semantic_cache = semantic_cache
    app.state.upload_dir = upload_dir

    # Set initial gauge values
    METRICS.BM25_INDEX_SIZE.set(bm25_search.size)
    METRICS.ACTIVE_SESSIONS.set(0)
    if semantic_cache:
        METRICS.CACHE_ENTRIES.set(len(semantic_cache._memory_cache))

    # --- Connection Warmup (eliminates cold-start latency on first request) ---
    try:
        logger.info("Warming up connections...")
        await embedding_generator.embed_query("warmup")
        logger.info("OpenAI embedding connection warmed")
    except Exception as e:
        logger.warning("Embedding warmup failed (non-fatal): %s", e)

    try:
        await llm_client.generate("Say OK", max_tokens=3)
        logger.info("OpenAI LLM connection warmed")
    except Exception as e:
        logger.warning("LLM warmup failed (non-fatal): %s", e)

    logger.info("All services initialized successfully")

    yield

    # --- Shutdown ---
    logger.info("Shutting down...")
    shutdown_langfuse()
    await qdrant_store.close()
    await llm_client.close()
    logger.info("Shutdown complete")


# --- Create FastAPI app ---
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan,
)

# --- CORS middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Mount static files (widget.js) ---
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")

# --- Mount API router ---
app.include_router(v1_router, prefix=settings.api_prefix)


# --- Root route: serve the website ---
@app.get("/")
async def root():
    return FileResponse(os.path.join(os.path.dirname(__file__), "test-website.html"))


# --- Global exception handler ---
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


if __name__ == "__main__":
    uvicorn.run(
        "backend.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.is_development,
    )
