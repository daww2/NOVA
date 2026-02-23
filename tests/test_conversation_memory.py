"""Tests for ConversationMemory — sliding window + summarization."""

import pytest
from src.core.memory.conversation import ConversationMemory


@pytest.fixture
def memory():
    return ConversationMemory(window_size=3, summary_max_chars=600)


class TestAddAndGet:

    def test_empty_session(self, memory):
        assert memory.get("nonexistent") == []

    def test_add_and_retrieve(self, memory):
        memory.add("s1", "user", "hello")
        messages = memory.get("s1")
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "hello"

    def test_multiple_messages(self, memory):
        memory.add("s1", "user", "hello")
        memory.add("s1", "assistant", "hi there")
        memory.add("s1", "user", "how are you?")
        messages = memory.get("s1")
        assert len(messages) == 3


class TestSlidingWindow:

    def test_within_window_no_summary(self, memory):
        memory.add("s1", "user", "msg1")
        memory.add("s1", "assistant", "msg2")
        memory.add("s1", "user", "msg3")
        messages = memory.get("s1")
        # 3 msgs = window_size, no summary needed
        assert len(messages) == 3
        assert messages[0]["role"] == "user"

    def test_beyond_window_creates_summary(self, memory):
        memory.add("s1", "user", "first question")
        memory.add("s1", "assistant", "first answer")
        memory.add("s1", "user", "second question")
        memory.add("s1", "assistant", "second answer")
        memory.add("s1", "user", "third question")

        messages = memory.get("s1")
        # Should be: [summary, msg3, msg4, msg5] = 4 messages
        assert len(messages) == 4
        assert messages[0]["role"] == "system"
        assert "summary" in messages[0]["content"].lower()

    def test_summary_truncated(self):
        memory = ConversationMemory(window_size=2, summary_max_chars=100)
        for i in range(10):
            memory.add("s1", "user", f"Message number {i} with lots of content " * 5)
        messages = memory.get("s1")
        summary = messages[0]["content"]
        assert len(summary) <= 103  # 100 + "..."


class TestSessionManagement:

    def test_separate_sessions(self, memory):
        memory.add("s1", "user", "hello from s1")
        memory.add("s2", "user", "hello from s2")
        assert len(memory.get("s1")) == 1
        assert len(memory.get("s2")) == 1
        assert memory.get("s1")[0]["content"] != memory.get("s2")[0]["content"]

    def test_clear_session(self, memory):
        memory.add("s1", "user", "hello")
        memory.clear("s1")
        assert memory.get("s1") == []

    def test_clear_all(self, memory):
        memory.add("s1", "user", "hello")
        memory.add("s2", "user", "hello")
        memory.clear_all()
        assert memory.get("s1") == []
        assert memory.get("s2") == []

    def test_session_count(self, memory):
        assert memory.get_session_count() == 0
        memory.add("s1", "user", "hi")
        memory.add("s2", "user", "hi")
        assert memory.get_session_count() == 2

    def test_clear_nonexistent_session(self, memory):
        # Should not raise
        memory.clear("nonexistent")
