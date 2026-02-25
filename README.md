# RAG Knowledge Assistant

A production-ready Retrieval-Augmented Generation (RAG) system built with FastAPI. Features hybrid search (vector + BM25), real-time SSE streaming, a 2-layer semantic cache, intelligent query classification with Arabic/English support, and an embeddable chat widget.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green)
![CI](https://img.shields.io/github/actions/workflow/status/YOUR_USERNAME/YOUR_REPO/ci.yml?label=CI)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Chat Widget (JS)                            │
│                     SSE Streaming / Markdown                        │
└───────────────────────────┬─────────────────────────────────────────┘
                            │ POST /api/v1/query
┌───────────────────────────▼─────────────────────────────────────────┐
│                        FastAPI Backend                               │
│                                                                      │
│  ┌───────────────┐  ┌───────────────────────┐  ┌─────────────────┐  │
│  │    Query       │  │  Conversation Memory  │  │   Query         │  │
│  │  Classifier    │  │  (Sliding Window)     │  │  Preprocessor   │  │
│  │  (Rule-based)  │  │                       │  │                 │  │
│  └───────┬───────┘  └───────────────────────┘  └────────┬────────┘  │
│          │                                               │           │
│          ▼                                               ▼           │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                  3-Layer Semantic Cache                       │   │
│  │  L1: Exact Hash ──► L2: Cosine Similarity ──► L3: Reranker  │   │
│  │     (~0.01ms)         (~5ms, >0.9)        (Cross-Encoder)   │   │
│  │                  Redis (persistent) / RAM fallback            │   │
│  └──────────────────────────┬───────────────────────────────────┘   │
│                    miss     │     hit → stream cached response       │
│                             ▼                                        │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │               Hybrid Search (Weighted RRF)                   │   │
│  │      Vector Search (HNSW)  +  BM25  +  Recency Boost        │   │
│  └──────────────┬──────────────────────┬───────────────────────┘   │
│                 │                      │                             │
│  ┌──────────────▼──────┐  ┌───────────▼────────────┐               │
│  │   Qdrant (Vectors)  │  │   BM25 (In-Memory)     │               │
│  │   HNSW Index        │  │   rank-bm25            │               │
│  └─────────────────────┘  └────────────────────────┘               │
│                 │                      │                             │
│                 └──────────┬───────────┘                             │
│                            ▼                                         │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                   Context Builder                            │   │
│  │           Chunk assembly within token limits                 │   │
│  └──────────────────────────┬───────────────────────────────────┘   │
│                              ▼                                       │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │               LLM Generation (OpenAI-compatible)             │   │
│  │               SSE Token-by-Token Streaming                   │   │
│  └──────────────────────────┬───────────────────────────────────┘   │
│                              │                                       │
│                    response stored in cache                          │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Evaluation

### Retrieval Evaluation

Measures hybrid search quality using a 250-query test set with ground truth answers. A chunk is relevant if the ground truth text appears in its content.

```bash
python -m tests.evaluation.run_retrieval_eval
```

#### Results — Before vs After Semantic Cache + Tuning

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| **Hit Rate** | 0.7200 | 0.8200 | +13.9% |
| **Recall@1** | 0.4800 | 0.6000 | +25.0% |
| **Recall@3** | 0.6200 | 0.7400 | +19.4% |
| **Recall@5** | 0.6600 | 0.7600 | +15.2% |
| **Recall@10** | 0.7200 | 0.8300 | +15.3% |
| **Precision@1** | 0.4800 | 0.6000 | +25.0% |
| **Precision@3** | 0.2267 | 0.2933 | +29.4% |
| **Precision@5** | 0.1520 | 0.1920 | +26.3% |
| **Precision@10** | 0.0840 | 0.1060 | +26.2% |
| **MRR** | 0.6147 | 0.7608 | +23.8% |
| **NDCG@10** | 0.7050 | 0.8290 | +17.6% |

> **Before**: Initial baseline with default weights and no cache.
> **After**: Tuned hybrid weights (Vector 5.0, BM25 3.0, Recency 0.2) + 3-layer semantic cache + preprocessing improvements.

### Generation Evaluation (RAGAS)

End-to-end RAG evaluation using [RAGAS](https://docs.ragas.io/) metrics. Each query runs through the full pipeline: embed → hybrid search → context building → LLM generation → RAGAS scoring.

```bash
# Full evaluation (makes API calls — costs money)
python -m tests.evaluation.run_generation_eval

# Print cached results (free, no API calls)
python -m tests.evaluation.run_generation_eval --cached
```

#### Estimated Metrics (250 queries, gpt-4o-mini)

| Metric | Score | Description |
|--------|-------|-------------|
| **Faithfulness** | ~0.85 | Are answers grounded in retrieved context? |
| **Answer Relevancy** | ~0.88 | Does the answer address the question? |
| **Context Precision** | ~0.78 | Are relevant chunks ranked higher? |
| **Context Recall** | ~0.82 | Does retrieved context cover the ground truth? |
| **Answer Correctness** | ~0.80 | Does the answer match the reference? |

> Estimates based on partial evaluation run (250 RAGAS jobs completed). Actual scores may vary by ~5% due to TimeoutError on some jobs during RAGAS internal LLM evaluation. Results are saved to `evaluation/results/` for future reference.

---

## Key Features

### Hybrid Search with Weighted RRF

Combines semantic and keyword search using Reciprocal Rank Fusion for higher recall and precision.

- **Vector search**: OpenAI `text-embedding-3-small` (1536 dimensions) → Qdrant with HNSW index
- **Keyword search**: BM25 (rank-bm25) with in-memory index
- **Fusion formula**: `score = vector_weight × 1/(k + rank_v) + bm25_weight × 1/(k + rank_b) + recency_weight × recency`
- **Default weights**: Vector 5.0, BM25 3.0, Recency 0.2
- **Retrieval pool**: Top 100 from each source → fused → return top K

### HNSW Vector Indexing (Qdrant)

- **Algorithm**: Hierarchical Navigable Small World graph
- **Parameters**: `m=16`, `ef_construct=100`
- **Distance metric**: Cosine similarity
- **Scale**: Optimized for 10M–500M vectors
- **Supports**: Qdrant Cloud and self-hosted instances

### 3-Layer Semantic Cache

Reduces latency and LLM costs by caching query-response pairs with a three-stage lookup.

| Layer | Method | Speed | Description |
|-------|--------|-------|-------------|
| 1 | Exact match | ~0.01ms | SHA-256 hash lookup (includes session context) |
| 2 | Semantic similarity | ~5ms | Embedding cosine similarity (threshold > 0.9) |
| 3 | Cross-encoder rerank | ~15ms | `ms-marco-MiniLM-L-6-v2` validates L2 candidates (threshold > 0.7) to filter false positives |

```
Query ──► L1 Exact Hash ──hit──► Return cached response
               │ miss
               ▼
          L2 Cosine Similarity (top candidate > 0.9)
               │ candidate found
               ▼
          L3 Cross-Encoder Rerank (score > 0.7)
               │ pass ──► Return cached response
               │ fail ──► Full RAG pipeline
```

- **Storage**: Redis (persistent) with automatic RAM fallback
- **TTL**: 3600s (1 hour) configurable
- **Max entries**: 10,000
- **Short query guard**: Queries under 4 words (e.g. "yes", "ok", "where") are never cached to prevent cross-topic collisions
- **Global cache**: Shared across all sessions — any user benefits from a previous identical/similar question, maximizing cost savings
- **L3 benefit**: The cross-encoder layer catches semantically similar but contextually different queries (e.g. "price of X" vs "price of Y") that fool cosine similarity alone

### SSE Real-Time Streaming

Token-by-token streaming using Server-Sent Events for instant UI feedback.

```
event: metadata   → {route, session_id, sources, cache}
event: token      → {content: "..."}
event: done       → {model, usage, latency_ms}
event: error      → {detail: "..."}
```

### Query Classification (Rule-Based)

Zero-cost, sub-millisecond query routing with Arabic and English support.

| Route | Trigger | Action |
|-------|---------|--------|
| `retrieval` | Factual/knowledge queries | Full RAG pipeline |
| `generation` | Creative tasks, greetings | LLM-only (no retrieval) |
| `clarification` | Vague/short queries | Ask for more details |
| `rejection` | Unsafe content | Block with explanation |

### Conversation Memory (Sliding Window)

- **Window size**: Last 3 messages kept in full
- **Older messages**: Summarized and truncated to ~150 tokens (600 chars)
- **Session-based**: Each conversation tracked independently

### Text Preprocessor

Full preprocessing pipeline applied before chunking:

1. Unicode normalization (NFKC)
2. Control character removal
3. Zero-width character cleanup
4. Quote and dash normalization
5. Whitespace collapsing
6. Optional: URL/email/phone removal, header/footer stripping

### Chunking Strategies

Six strategies available, configurable via API or environment:

| Strategy | Description |
|----------|-------------|
| `recursive` | Default — 512 tokens, 50 overlap (recommended) |
| `fixed` | Fixed-size character splits |
| `semantic` | Split by semantic boundaries |
| `sentence` | Sentence-level splitting |
| `document` | One chunk per document |
| `page` | Page-level splitting (PDF) |

### Embeddable Chat Widget

A self-contained JavaScript widget (`widget.js`) that can be embedded on any website.

- **Markdown rendering**: Bold, italic, code, headings
- **Chat history**: LocalStorage with 24-hour TTL
- **SSE streaming**: Real-time token display
- **Configurable**: Title, subtitle, color, position, suggestions
- **Responsive**: Mobile-friendly floating bubble UI

```html
<script src="https://your-domain.com/static/widget.js" data-api-url="https://your-domain.com"></script>
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/query` | Full RAG pipeline with SSE streaming |
| `POST` | `/api/v1/query/classify` | Debug: classify query without execution |
| `POST` | `/api/v1/search` | Hybrid search (vector + BM25) |
| `POST` | `/api/v1/search/vector` | Vector-only search |
| `POST` | `/api/v1/documents` | Upload and index a document |
| `GET` | `/api/v1/documents` | List all indexed documents |
| `DELETE` | `/api/v1/documents/{id}` | Delete a document |
| `DELETE` | `/api/v1/cache` | Clear semantic cache |
| `GET` | `/api/v1/health` | Health check |
| `GET` | `/api/v1/health/stats` | System statistics |

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| **Framework** | FastAPI + Uvicorn |
| **LLM** | OpenAI API (GPT-4o-mini default, any OpenAI-compatible) |
| **Embeddings** | OpenAI `text-embedding-3-small` (1536d) |
| **Vector Store** | Qdrant (HNSW index, cosine distance) |
| **Keyword Search** | BM25 (rank-bm25) |
| **Cache** | Redis (persistent) + in-memory fallback |
| **Database** | PostgreSQL + SQLAlchemy (async) |
| **Document Parsing** | PyPDF, python-docx, BeautifulSoup, pandas |
| **Tokenization** | tiktoken |
| **Orchestration** | LangChain |
| **Monitoring** | Prometheus client |
| **CI** | GitHub Actions (ruff + pytest) |

---

## Quick Start

### Prerequisites

- Python 3.12+
- Docker (optional)
- OpenAI API key
- Qdrant instance (cloud or local)
- Redis instance (optional, for persistent caching)

### 1. Clone and configure

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
cp .env.example .env
# Edit .env with your API keys and service URLs
```

### 2. Run with Docker

```bash
docker compose up --build
```

The API will be available at `http://localhost:8000`.

### 3. Run locally (without Docker)

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

---

## Configuration

All settings are configured via environment variables (`.env` file). See `.env.example` for the full list.

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | OpenAI API key (required) |
| `LLM_MODEL` | `gpt-4o-mini` | LLM model name |
| `LLM_BASE_URL` | — | Custom OpenAI-compatible endpoint |
| `EMBEDDING_MODEL_NAME` | `text-embedding-3-small` | Embedding model |
| `EMBEDDING_DIMENSIONS` | `1536` | Embedding dimensions |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant connection URL |
| `QDRANT_COLLECTION` | `documents` | Collection name |
| `REDIS_URL` | — | Redis URL for persistent cache |
| `CHUNK_STRATEGY` | `recursive` | Chunking strategy |
| `CHUNK_SIZE` | `512` | Chunk size in tokens |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks |
| `VECTOR_WEIGHT` | `5.0` | Hybrid search vector weight |
| `BM25_WEIGHT` | `3.0` | Hybrid search BM25 weight |
| `CACHE_SEMANTIC_THRESHOLD` | `0.9` | Semantic cache similarity threshold |

---

## Project Structure

```
├── main.py                          # FastAPI app entry point + lifespan
├── api/v1/
│   ├── query.py                     # RAG pipeline + SSE streaming
│   ├── search.py                    # Hybrid and vector search endpoints
│   ├── documents.py                 # Document upload/delete/list
│   ├── cache.py                     # Cache management
│   ├── health.py                    # Health check + stats
│   ├── schemas.py                   # Pydantic request/response models
│   └── dependencies.py              # FastAPI dependency injection
├── src/
│   ├── config/config.py             # Pydantic settings (env-driven)
│   ├── core/
│   │   ├── caching/
│   │   │   ├── semantic_cache.py    # 2-layer semantic cache
│   │   │   └── embedding_cache.py   # Embedding cache (Redis/RAM)
│   │   ├── chunking/
│   │   │   ├── strategies.py        # 6 chunking strategies
│   │   │   └── preprocessor.py      # Text preprocessing pipeline
│   │   ├── embedding/
│   │   │   ├── generator.py         # OpenAI embedding client + batching
│   │   │   └── models.py            # Embedding model registry
│   │   ├── generation/
│   │   │   ├── llm_client.py        # OpenAI LLM client + streaming
│   │   │   ├── context_builder.py   # Context assembly for prompts
│   │   │   └── prompt_manager.py    # Prompt templates
│   │   ├── memory/
│   │   │   └── conversation.py      # Sliding window memory
│   │   ├── query/
│   │   │   └── classifier.py        # Rule-based query router
│   │   └── retrieval/
│   │       ├── hybrid_search.py     # Weighted RRF fusion
│   │       ├── vector_search.py     # Qdrant vector search
│   │       └── bm25_search.py       # BM25 keyword search
│   └── services/
│       ├── document_processor.py    # PDF, DOCX, HTML, CSV, TXT parsing
│       └── vector_store/qdrant.py   # Qdrant client wrapper (HNSW config)
├── evaluation/
│   ├── retrieval_eval.py            # Retrieval metrics (Recall, Precision, MRR, NDCG)
│   ├── generation_eval.py           # RAGAS generation metrics (5 metrics)
│   ├── test_set/queries.json        # 250-query evaluation test set
│   └── results/                     # Saved evaluation results (JSON)
├── static/widget.js                 # Embeddable chat widget
├── tests/
│   ├── evaluation/
│   │   ├── run_retrieval_eval.py    # Retrieval evaluation runner
│   │   └── run_generation_eval.py   # Generation evaluation runner (--cached flag)
│   └── ...                          # 117 pytest tests
├── Dockerfile                       # Production container
├── docker-compose.yml               # Docker Compose setup
├── .github/workflows/ci.yml         # CI pipeline (lint + tests)
└── requirements.txt                 # Python dependencies
```

---

## Testing

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run a specific test file
pytest tests/test_hybrid_search.py
```

**117 tests** covering: chunking strategies, text preprocessing, hybrid search (RRF fusion), BM25 search, query classification, conversation memory, context building, document processing, and API schemas.

---

## CI/CD

**GitHub Actions** runs on every push and pull request to `main`:

1. **Lint** — `ruff check .` (pycodestyle + pyflakes rules)
2. **Test** — `pytest` (117 tests)

---

## Supported Document Formats

| Format | Extension | Parser |
|--------|-----------|--------|
| PDF | `.pdf` | PyPDF |
| Word | `.docx` | python-docx |
| HTML | `.html` | BeautifulSoup |
| CSV | `.csv` | pandas |
| Plain text | `.txt` | Built-in |

---

## License

MIT
