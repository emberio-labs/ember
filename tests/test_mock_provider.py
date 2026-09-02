"""Тесты MockProvider — работают без сети и API-ключей."""

import pytest

from ember.providers import get_provider
from ember.providers.mock import MockProvider
from ember.types import ChatRequest, ChatResponse, Message, Usage


def _request() -> ChatRequest:
    return ChatRequest(messages=[Message(role="user", content="Здравствуй")], model="mock-1")


def test_mock_complete() -> None:
    provider = MockProvider(response_text="Привет, мир")
    response = provider.complete(_request())

    assert isinstance(response, ChatResponse)
    assert response.message.role == "assistant"
    assert response.message.content == "Привет, мир"
    assert response.model == "mock-1"


def test_mock_complete_usage() -> None:
    provider = MockProvider(response_text="Привет, мир")
    response = provider.complete(_request())

    assert isinstance(response.usage, Usage)
    assert response.usage.prompt_tokens > 0
    assert response.usage.completion_tokens > 0
    assert response.usage.total_tokens == (
        response.usage.prompt_tokens + response.usage.completion_tokens
    )


def test_mock_stream_words() -> None:
    provider = MockProvider(response_text="Привет мир")
    chunks = list(provider.stream(_request()))

    assert len(chunks) == 2
    assert "".join(c.delta for c in chunks).strip() == "Привет мир"


def test_mock_stream_finish_reason() -> None:
    provider = MockProvider(response_text="Привет мир")
    chunks = list(provider.stream(_request()))

    assert chunks[0].finish_reason is None
    assert chunks[-1].finish_reason == "stop"


def test_registry_get_mock() -> None:
    provider = get_provider("mock", response_text="из реестра")

    assert isinstance(provider, MockProvider)
    assert provider.response_text == "из реестра"


def test_registry_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Неизвестный провайдер"):
        get_provider("no-such-provider")
