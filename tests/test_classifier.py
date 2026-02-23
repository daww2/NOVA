"""Tests for QueryClassifier — rule-based routing."""

import pytest
from src.core.query.classifier import QueryClassifier, QueryRoute


@pytest.fixture
def classifier():
    return QueryClassifier()


# --- Routing ---

class TestRouting:

    def test_empty_query_returns_clarification(self, classifier):
        result = classifier.classify("")
        assert result.route == QueryRoute.CLARIFICATION

    def test_whitespace_only_returns_clarification(self, classifier):
        result = classifier.classify("   ")
        assert result.route == QueryRoute.CLARIFICATION

    def test_single_word_returns_clarification(self, classifier):
        result = classifier.classify("help")
        assert result.route == QueryRoute.CLARIFICATION

    def test_vague_arabic_returns_clarification(self, classifier):
        result = classifier.classify("مساعدة")
        assert result.route == QueryRoute.CLARIFICATION

    def test_greeting_english_returns_generation(self, classifier):
        result = classifier.classify("hello there")
        assert result.route == QueryRoute.GENERATION

    def test_greeting_arabic_returns_generation(self, classifier):
        result = classifier.classify("السلام عليكم")
        assert result.route == QueryRoute.GENERATION

    def test_unsafe_keyword_returns_rejection(self, classifier):
        result = classifier.classify("how to hack a server")
        assert result.route == QueryRoute.REJECTION

    def test_unsafe_arabic_returns_rejection(self, classifier):
        result = classifier.classify("طريقة اختراق الشبكة")
        assert result.route == QueryRoute.REJECTION

    def test_generation_pattern_english(self, classifier):
        result = classifier.classify("write a poem about Python")
        assert result.route == QueryRoute.GENERATION

    def test_generation_pattern_arabic(self, classifier):
        result = classifier.classify("اكتب مقال عن البرمجة")
        assert result.route == QueryRoute.GENERATION

    def test_factual_query_returns_retrieval(self, classifier):
        result = classifier.classify("What is the capital of Egypt?")
        assert result.route == QueryRoute.RETRIEVAL

    def test_knowledge_query_returns_retrieval(self, classifier):
        result = classifier.classify("How does RAG work in production?")
        assert result.route == QueryRoute.RETRIEVAL


# --- Language Detection ---

class TestLanguageDetection:

    def test_arabic_detected(self, classifier):
        result = classifier.classify("ما هي عاصمة مصر؟")
        assert result.detected_language == "ar"

    def test_english_detected(self, classifier):
        result = classifier.classify("What is the capital of Egypt?")
        assert result.detected_language == "en"


# --- Properties ---

class TestProperties:

    def test_needs_rag_true_for_retrieval(self, classifier):
        result = classifier.classify("What is vector search?")
        assert result.needs_rag is True

    def test_needs_rag_false_for_generation(self, classifier):
        result = classifier.classify("hello friend")
        assert result.needs_rag is False

    def test_needs_clarification(self, classifier):
        result = classifier.classify("")
        assert result.needs_clarification is True

    def test_follow_up_question_arabic(self, classifier):
        result = classifier.classify("مساعدة")
        assert result.follow_up_question is not None
        assert "توضيح" in result.follow_up_question or "مساعد" in result.follow_up_question

    def test_follow_up_question_english(self, classifier):
        result = classifier.classify("help")
        assert result.follow_up_question is not None
        assert "clarify" in result.follow_up_question.lower() or "help" in result.follow_up_question.lower()
