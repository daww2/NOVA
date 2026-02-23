"""Tests for Pydantic schemas — validation."""

import pytest
from pydantic import ValidationError
from api.v1.schemas import QueryRequest, SearchRequest, ClassifyRequest


class TestQueryRequest:

    def test_valid_request(self):
        req = QueryRequest(query="What is RAG?")
        assert req.query == "What is RAG?"
        assert req.top_k == 3
        assert req.temperature == 0.7
        assert req.max_tokens == 1024

    def test_custom_values(self):
        req = QueryRequest(query="test", top_k=10, temperature=0.2, max_tokens=512)
        assert req.top_k == 10
        assert req.temperature == 0.2
        assert req.max_tokens == 512

    def test_top_k_min_validation(self):
        with pytest.raises(ValidationError):
            QueryRequest(query="test", top_k=0)

    def test_top_k_max_validation(self):
        with pytest.raises(ValidationError):
            QueryRequest(query="test", top_k=51)

    def test_temperature_min_validation(self):
        with pytest.raises(ValidationError):
            QueryRequest(query="test", temperature=-0.1)

    def test_temperature_max_validation(self):
        with pytest.raises(ValidationError):
            QueryRequest(query="test", temperature=2.1)

    def test_max_tokens_min_validation(self):
        with pytest.raises(ValidationError):
            QueryRequest(query="test", max_tokens=0)

    def test_optional_session_id(self):
        req = QueryRequest(query="test")
        assert req.session_id is None

    def test_optional_filter(self):
        req = QueryRequest(query="test", filter={"doc_id": "abc"})
        assert req.filter == {"doc_id": "abc"}


class TestSearchRequest:

    def test_valid_request(self):
        req = SearchRequest(query="search term")
        assert req.query == "search term"
        assert req.top_k == 3

    def test_top_k_validation(self):
        with pytest.raises(ValidationError):
            SearchRequest(query="test", top_k=0)


class TestClassifyRequest:

    def test_valid_request(self):
        req = ClassifyRequest(query="hello world")
        assert req.query == "hello world"
