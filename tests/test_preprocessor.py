"""Tests for TextPreprocessor — text cleaning pipeline."""

import pytest
from src.core.chunking.preprocessor import TextPreprocessor, PreprocessingConfig


@pytest.fixture
def preprocessor():
    return TextPreprocessor()


class TestBasicPreprocessing:

    def test_empty_string(self, preprocessor):
        assert preprocessor.preprocess("") == ""

    def test_whitespace_only(self, preprocessor):
        assert preprocessor.preprocess("   \n\n  ") == ""

    def test_normalizes_whitespace(self, preprocessor):
        result = preprocessor.preprocess("hello    world")
        assert "    " not in result
        assert "hello" in result and "world" in result

    def test_collapses_newlines(self, preprocessor):
        result = preprocessor.preprocess("hello\n\n\n\n\nworld")
        # Max 2 consecutive newlines
        assert "\n\n\n" not in result

    def test_strips_result(self, preprocessor):
        result = preprocessor.preprocess("  hello world  ")
        assert result == "hello world"


class TestUnicodeNormalization:

    def test_normalizes_unicode(self, preprocessor):
        # NFKC normalization
        result = preprocessor.preprocess("ﬁ")  # fi ligature
        assert result == "fi"

    def test_removes_zero_width_chars(self, preprocessor):
        result = preprocessor.preprocess("hello\u200bworld")
        assert result == "helloworld"

    def test_removes_bom(self, preprocessor):
        result = preprocessor.preprocess("\ufeffhello")
        assert result == "hello"


class TestQuoteAndDashNormalization:

    def test_normalizes_smart_quotes(self, preprocessor):
        # Use angle quotes which survive NFKC normalization
        result = preprocessor.preprocess("\u00abhello\u00bb")
        assert '"hello"' == result

    def test_normalizes_em_dash(self, preprocessor):
        result = preprocessor.preprocess("hello\u2014world")
        assert "hello-world" == result


class TestContentCleaning:

    def test_removes_urls_when_enabled(self):
        config = PreprocessingConfig(remove_urls=True)
        preprocessor = TextPreprocessor(config)
        result = preprocessor.preprocess("visit https://example.com for info")
        assert "https://example.com" not in result

    def test_preserves_urls_by_default(self, preprocessor):
        result = preprocessor.preprocess("visit https://example.com for info")
        assert "https://example.com" in result

    def test_removes_emails_when_enabled(self):
        config = PreprocessingConfig(remove_emails=True)
        preprocessor = TextPreprocessor(config)
        result = preprocessor.preprocess("contact user@example.com today")
        assert "user@example.com" not in result


class TestCleanForEmbedding:

    def test_light_cleaning(self, preprocessor):
        result = preprocessor.clean_for_embedding("  hello\u200b  world  ")
        assert "hello" in result
        assert "\u200b" not in result

    def test_empty_input(self, preprocessor):
        assert preprocessor.clean_for_embedding("") == ""


class TestExtractMetadata:

    def test_extracts_urls(self, preprocessor):
        result = preprocessor.extract_metadata_text("visit https://example.com")
        assert "urls" in result
        assert "https://example.com" in result["urls"]

    def test_extracts_emails(self, preprocessor):
        result = preprocessor.extract_metadata_text("email user@example.com")
        assert "emails" in result
        assert "user@example.com" in result["emails"]

    def test_no_urls_no_key(self, preprocessor):
        result = preprocessor.extract_metadata_text("plain text here")
        assert "urls" not in result
