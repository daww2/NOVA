"""Tests for Chunker — text splitting strategies."""

import pytest
from src.core.chunking.strategies import Chunker, Chunk, ChunkingStrategy


@pytest.fixture
def chunker():
    return Chunker(strategy="recursive", chunk_size=50, chunk_overlap=5)


@pytest.fixture
def long_text():
    return ("This is a test paragraph with enough words to be meaningful. " * 20).strip()


class TestChunking:

    def test_empty_text_returns_empty(self, chunker):
        assert chunker.chunk("") == []

    def test_whitespace_only_returns_empty(self, chunker):
        assert chunker.chunk("   \n\n  ") == []

    def test_produces_chunks(self, chunker, long_text):
        chunks = chunker.chunk(long_text, document_id="doc1")
        assert len(chunks) > 1

    def test_chunks_have_content(self, chunker, long_text):
        chunks = chunker.chunk(long_text)
        for c in chunks:
            assert c.content.strip() != ""

    def test_chunks_have_ids(self, chunker, long_text):
        chunks = chunker.chunk(long_text, document_id="doc1")
        for c in chunks:
            assert c.chunk_id.startswith("chunk_")

    def test_unique_chunk_ids(self, chunker, long_text):
        chunks = chunker.chunk(long_text, document_id="doc1")
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_document_id_preserved(self, chunker, long_text):
        chunks = chunker.chunk(long_text, document_id="my_doc")
        for c in chunks:
            assert c.document_id == "my_doc"

    def test_metadata_preserved(self, chunker, long_text):
        chunks = chunker.chunk(long_text, metadata={"key": "value"})
        for c in chunks:
            assert c.metadata.get("key") == "value"

    def test_total_chunks_correct(self, chunker, long_text):
        chunks = chunker.chunk(long_text)
        for c in chunks:
            assert c.total_chunks == len(chunks)

    def test_chunk_index_sequential(self, chunker, long_text):
        chunks = chunker.chunk(long_text)
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_token_count_positive(self, chunker, long_text):
        chunks = chunker.chunk(long_text)
        for c in chunks:
            assert c.token_count > 0


class TestDocumentStrategy:

    def test_document_strategy_single_chunk(self):
        chunker = Chunker(strategy="document", chunk_size=512, chunk_overlap=50)
        text = "Short text that should stay as one chunk."
        chunks = chunker.chunk(text, document_id="doc1")
        assert len(chunks) == 1
        assert chunks[0].total_chunks == 1


class TestChunkToDict:

    def test_to_dict(self):
        chunk = Chunk(content="test", document_id="d1", chunk_index=0, total_chunks=1, token_count=1)
        d = chunk.to_dict()
        assert d["content"] == "test"
        assert d["document_id"] == "d1"
        assert "chunk_id" in d
        assert "metadata" in d


class TestFixedStrategy:

    def test_fixed_produces_chunks(self, long_text):
        chunker = Chunker(strategy="fixed", chunk_size=50, chunk_overlap=5)
        chunks = chunker.chunk(long_text)
        assert len(chunks) >= 1
