"""Tests for HybridSearch — RRF fusion logic."""

import math
from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest
from src.core.retrieval.hybrid_search import HybridSearch
from src.core.retrieval.bm25_search import SearchResult as BM25Result


class FakeVectorResult:
    def __init__(self, chunk_id, score, content="", metadata=None, document_id=""):
        self.chunk_id = chunk_id
        self.score = score
        self.content = content
        self.metadata = metadata or {}
        self.document_id = document_id


@pytest.fixture
def hybrid():
    vector_search = MagicMock()
    bm25_search = MagicMock()

    vector_search.search.return_value = [
        FakeVectorResult("c1", 0.95, "chunk 1 content", {}, "d1"),
        FakeVectorResult("c2", 0.85, "chunk 2 content", {}, "d1"),
        FakeVectorResult("c3", 0.70, "chunk 3 content", {}, "d2"),
    ]
    bm25_search.search.return_value = [
        BM25Result("c2", 5.0, "chunk 2 content", {}, "d1"),
        BM25Result("c4", 4.0, "chunk 4 content", {}, "d2"),
        BM25Result("c1", 3.0, "chunk 1 content", {}, "d1"),
    ]

    hs = HybridSearch(vector_search, bm25_search)
    return hs


class TestRRFFusion:

    def test_returns_results(self, hybrid):
        results = hybrid.search("test query", [0.1, 0.2], top_k=10)
        assert len(results) > 0

    def test_results_sorted_by_score(self, hybrid):
        results = hybrid.search("test query", [0.1, 0.2], top_k=10)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_union_of_both_sources(self, hybrid):
        results = hybrid.search("test query", [0.1, 0.2], top_k=10)
        ids = {r.chunk_id for r in results}
        # c1, c2, c3 from vector; c2, c4, c1 from bm25 → union = {c1, c2, c3, c4}
        assert ids == {"c1", "c2", "c3", "c4"}

    def test_chunk_in_both_has_higher_score(self, hybrid):
        results = hybrid.search("test query", [0.1, 0.2], top_k=10)
        result_map = {r.chunk_id: r for r in results}
        # c2 appears in both vector (rank 1) and bm25 (rank 0), should have high score
        # c3 appears only in vector (rank 2)
        assert result_map["c2"].score > result_map["c3"].score

    def test_from_vector_flag(self, hybrid):
        results = hybrid.search("test query", [0.1, 0.2], top_k=10)
        result_map = {r.chunk_id: r for r in results}
        assert result_map["c1"].from_vector is True
        assert result_map["c4"].from_vector is False

    def test_from_bm25_flag(self, hybrid):
        results = hybrid.search("test query", [0.1, 0.2], top_k=10)
        result_map = {r.chunk_id: r for r in results}
        assert result_map["c2"].from_bm25 is True
        assert result_map["c3"].from_bm25 is False

    def test_respects_top_k(self, hybrid):
        results = hybrid.search("test query", [0.1, 0.2], top_k=2)
        assert len(results) == 2


class TestRecency:

    def test_recent_doc_gets_higher_recency(self, hybrid):
        now = datetime.now().isoformat()
        old = (datetime.now() - timedelta(days=365)).isoformat()

        score_recent = hybrid._calculate_recency({"created_at": now})
        score_old = hybrid._calculate_recency({"created_at": old})
        assert score_recent > score_old

    def test_no_timestamp_returns_zero(self, hybrid):
        assert hybrid._calculate_recency({}) == 0.0

    def test_invalid_timestamp_returns_zero(self, hybrid):
        assert hybrid._calculate_recency({"created_at": "not-a-date"}) == 0.0

    def test_numeric_timestamp(self, hybrid):
        ts = datetime.now().timestamp()
        score = hybrid._calculate_recency({"created_at": ts})
        assert score > 0.5  # Recent should be close to 1.0
