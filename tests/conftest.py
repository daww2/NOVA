"""Shared fixtures for all tests."""

import os
import sys
import tempfile

import pytest

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def sample_chunks():
    """Sample chunk dicts for testing search/context."""
    return [
        {
            "chunk_id": "chunk_001",
            "content": "Python is a high-level programming language known for its readability.",
            "document_id": "doc_1",
            "metadata": {"filename": "python.txt"},
        },
        {
            "chunk_id": "chunk_002",
            "content": "FastAPI is a modern web framework for building APIs with Python.",
            "document_id": "doc_1",
            "metadata": {"filename": "python.txt"},
        },
        {
            "chunk_id": "chunk_003",
            "content": "Qdrant is a vector database for similarity search.",
            "document_id": "doc_2",
            "metadata": {"filename": "qdrant.txt"},
        },
    ]


@pytest.fixture
def tmp_text_file():
    """Create a temporary text file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write("This is a test document.\n\nIt has multiple paragraphs.\n\nThird paragraph here.")
        path = f.name
    yield path
    os.unlink(path)


@pytest.fixture
def tmp_html_file():
    """Create a temporary HTML file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False, encoding="utf-8") as f:
        f.write("<html><body><h1>Title</h1><p>Hello world</p><script>var x=1;</script></body></html>")
        path = f.name
    yield path
    os.unlink(path)


@pytest.fixture
def tmp_csv_file():
    """Create a temporary CSV file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8") as f:
        f.write("name,age,city\nAlice,30,Cairo\nBob,25,Alex\n")
        path = f.name
    yield path
    os.unlink(path)
