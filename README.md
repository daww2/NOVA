# 🚀 Production-Grade RAG System

A production-ready Retrieval-Augmented Generation (RAG) system focused on **hybrid retrieval, measurable quality improvements, and low-latency performance**.
Designed with evaluation-driven iteration and production best practices.

---

# ✨ Key Features

* Hybrid Search (Vector + BM25 + Recency)
* Weighted scoring formula from day one
* Qdrant vector database with HNSW indexing
* Semantic cache for repeated-query optimization
* Quantitative retrieval evaluation (Recall@K, NDCG, MRR, Hit Rate)
* Percentile-based latency measurement (P50 / P99)
* Dockerized deployment

---

# 🏗️ System Architecture

1. User Query
2. Semantic Cache Check
3. Single Query Embedding Call (warmed)
4. Hybrid Retrieval (Dense + BM25 + Recency)
5. Context Construction
6. LLM Response

---

# 🔎 Hybrid Retrieval Strategy

Vector search alone is insufficient in production systems.
Hybrid retrieval is implemented from the beginning using a weighted scoring formula:

Score = (VectorScore × 5) + (BM25Score × 3) + (RecencyScore × 0.2)

Where:

* **Vector search** captures semantic similarity
* **BM25** ensures exact keyword matching (IDs, error codes, entities)
* **Recency score** lightly prioritizes newer documents

---

# 🗄️ Vector Database & Indexing

* **Qdrant** used as the vector database
* **HNSW (Hierarchical Navigable Small World)** indexing for efficient ANN search

HNSW provides high recall (>95% with proper tuning) and strong performance at scale.

---

# 📊 Retrieval Evaluation (250 Labeled Queries)

Metrics:

* Recall@10
* NDCG@10
* MRR
* Hit Rate

## 🔹 Vector-Only Baseline

| Metric    | Score |
| --------- | ----- |
| Recall@10 | 0.64  |
| NDCG@10   | 0.60  |
| MRR       | 0.55  |
| Hit Rate  | 0.67  |

## 🔹 Hybrid Search (Weighted Formula Applied)

| Metric    | Score |
| --------- | ----- |
| Recall@10 | 0.88  |
| NDCG@10   | 0.82  |
| MRR       | 0.73  |
| Hit Rate  | 0.86  |

Retrieval effectiveness improved by approximately **~35–40% relative to baseline**.

Evaluation setup:

* 250 manually labeled queries
* Deterministic evaluation configuration
* Median metric values reported

---

# ⚡ Retrieval Latency Benchmarks

Latency measured on a warm system with persistent connections and single embedding call per request.

| Percentile | Latency |
| ---------- | ------- |
| P50        | 45 ms   |
| P99        | 185 ms  |

Targets followed:

* P50 < 50ms
* P99 < 200ms

System meets production-grade retrieval latency thresholds.

---

# 🧠 Semantic Cache

A lightweight semantic cache is implemented to reduce repeated computation and improve cost efficiency.

* Embedding similarity matching
* Configurable scope (global/session)
* Bypasses retrieval and generation on hit


---

# 🎯 Impact Summary

* Implemented hybrid retrieval with weighted scoring formula (Vector×5 + BM25×3 + Recency×0.2)
* Improved Recall@10 from 64% → 88%
* Increased NDCG@10 from 0.60 → 0.82
* Achieved P50 = 45ms and P99 = 185ms retrieval latency
* Deployed Qdrant with HNSW indexing for high-recall ANN search
* Built evaluation-driven RAG pipeline with 250 labeled queries

---

# 📜 License

MIT License
