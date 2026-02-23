"""Tests for DocumentProcessor — file extraction."""

import pytest
from src.services.document_processor import DocumentProcessor


@pytest.fixture
def processor():
    return DocumentProcessor()


class TestTextExtraction:

    def test_extract_txt(self, processor, tmp_text_file):
        doc = processor.process(tmp_text_file)
        assert "test document" in doc.content
        assert "multiple paragraphs" in doc.content
        assert doc.page_count == 1

    def test_extract_html(self, processor, tmp_html_file):
        doc = processor.process(tmp_html_file)
        assert "Title" in doc.content
        assert "Hello world" in doc.content
        # Scripts should be removed
        assert "var x" not in doc.content

    def test_extract_csv(self, processor, tmp_csv_file):
        doc = processor.process(tmp_csv_file)
        assert "Alice" in doc.content
        assert "Bob" in doc.content
        assert "Cairo" in doc.content


class TestMetadata:

    def test_metadata_includes_filename(self, processor, tmp_text_file):
        doc = processor.process(tmp_text_file)
        assert "filename" in doc.metadata
        assert doc.metadata["filename"].endswith(".txt")

    def test_metadata_includes_file_type(self, processor, tmp_text_file):
        doc = processor.process(tmp_text_file)
        assert doc.metadata["file_type"] == ".txt"

    def test_metadata_includes_file_size(self, processor, tmp_text_file):
        doc = processor.process(tmp_text_file)
        assert doc.metadata["file_size"] > 0


class TestErrors:

    def test_file_not_found(self, processor):
        with pytest.raises(FileNotFoundError):
            processor.process("/nonexistent/file.txt")

    def test_unsupported_extension(self, processor, tmp_path):
        bad_file = tmp_path / "test.xyz"
        bad_file.write_text("content")
        with pytest.raises(ValueError, match="Unsupported"):
            processor.process(str(bad_file))


class TestSupportedTypes:

    def test_all_types_registered(self, processor):
        expected = {".pdf", ".docx", ".doc", ".txt", ".md", ".csv", ".xlsx", ".html"}
        assert set(processor.SUPPORTED_TYPES.keys()) == expected
