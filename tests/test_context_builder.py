"""Tests for ContextBuilder — context formatting from chunks."""

import pytest
from src.core.generation.context_builder import ContextBuilder


@pytest.fixture
def builder():
    return ContextBuilder(max_tokens=2000, max_chunks=10)


class TestBuild:

    def test_empty_chunks(self, builder):
        context = builder.build([])
        assert context.text == ""
        assert context.chunks_used == 0
        assert context.sources == []

    def test_single_chunk(self, builder):
        chunks = [{"content": "Hello world", "document_id": "d1", "chunk_id": "c1", "score": 0.9}]
        context = builder.build(chunks)
        assert "Hello world" in context.text
        assert context.chunks_used == 1
        assert len(context.sources) == 1

    def test_multiple_chunks(self, builder, sample_chunks):
        context = builder.build(sample_chunks)
        assert context.chunks_used == 3
        assert len(context.sources) == 3

    def test_source_numbering(self, builder):
        chunks = [
            {"content": "First chunk", "document_id": "d1", "chunk_id": "c1", "score": 0.9},
            {"content": "Second chunk", "document_id": "d1", "chunk_id": "c2", "score": 0.8},
        ]
        context = builder.build(chunks, include_sources=True)
        assert "[1]" in context.text
        assert "[2]" in context.text

    def test_no_sources_numbering(self, builder):
        chunks = [{"content": "Hello", "document_id": "d1", "chunk_id": "c1", "score": 0.9}]
        context = builder.build(chunks, include_sources=False)
        assert "[1]" not in context.text

    def test_respects_max_chunks(self):
        builder = ContextBuilder(max_tokens=10000, max_chunks=2)
        chunks = [
            {"content": f"Chunk {i}", "document_id": "d1", "chunk_id": f"c{i}", "score": 0.5}
            for i in range(5)
        ]
        context = builder.build(chunks)
        assert context.chunks_used == 2

    def test_respects_max_tokens(self):
        builder = ContextBuilder(max_tokens=10, max_chunks=100, chars_per_token=4)
        # max_chars = 10 * 4 = 40
        chunks = [
            {"content": "A" * 50, "document_id": "d1", "chunk_id": "c1", "score": 0.9},
            {"content": "B" * 50, "document_id": "d1", "chunk_id": "c2", "score": 0.8},
        ]
        context = builder.build(chunks)
        # Should truncate, not include everything
        assert context.chunks_used <= 2

    def test_sources_contain_required_fields(self, builder):
        chunks = [{"content": "Hello", "document_id": "d1", "chunk_id": "c1", "score": 0.95}]
        context = builder.build(chunks)
        source = context.sources[0]
        assert source["document_id"] == "d1"
        assert source["chunk_id"] == "c1"
        assert source["score"] == 0.95
        assert source["index"] == 1

    def test_estimated_tokens(self, builder):
        chunks = [{"content": "A" * 400, "document_id": "d1", "chunk_id": "c1", "score": 0.9}]
        context = builder.build(chunks)
        assert context.estimated_tokens == 400 // builder.chars_per_token
