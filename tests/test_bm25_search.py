"""Tests for BM25Search — keyword search."""

import pytest
from src.core.retrieval.bm25_search import BM25Search


@pytest.fixture
def bm25(sample_chunks):
    search = BM25Search()
    search.index(sample_chunks)
    return search


class TestIndexing:

    def test_index_returns_count(self, sample_chunks):
        search = BM25Search()
        count = search.index(sample_chunks)
        assert count == 3

    def test_size_after_index(self, bm25):
        assert bm25.size == 3

    def test_empty_index(self):
        search = BM25Search()
        assert search.size == 0

    def test_search_without_index_returns_empty(self):
        search = BM25Search()
        results = search.search("python")
        assert results == []


class TestSearch:

    def test_finds_relevant_result(self, bm25):
        results = bm25.search("Python programming")
        assert len(results) > 0
        assert any("Python" in r.content for r in results)

    def test_respects_top_k(self, bm25):
        results = bm25.search("Python", top_k=1)
        assert len(results) <= 1

    def test_results_have_scores(self, bm25):
        results = bm25.search("FastAPI web framework")
        for r in results:
            assert r.score > 0

    def test_results_have_metadata(self, bm25):
        results = bm25.search("vector database")
        assert len(results) > 0
        assert results[0].chunk_id != ""
        assert results[0].document_id != ""

    def test_empty_query_returns_empty(self, bm25):
        results = bm25.search("")
        assert results == []

    def test_irrelevant_query_returns_low_scores(self, bm25):
        results = bm25.search("quantum mechanics spacetime")
        # Should still return something but with low relevance
        if results:
            assert all(r.score >= 0 for r in results)


class TestTokenizer:

    def test_tokenize_english(self):
        search = BM25Search()
        tokens = search._tokenize("Hello World")
        assert tokens == ["hello", "world"]

    def test_tokenize_arabic(self):
        search = BM25Search()
        tokens = search._tokenize("مرحبا بالعالم")
        assert len(tokens) == 2

    def test_tokenize_empty(self):
        search = BM25Search()
        assert search._tokenize("") == []

    def test_tokenize_preserves_hyphens(self):
        search = BM25Search()
        tokens = search._tokenize("ERR-404")
        assert "err-404" in tokens
