"""Мок-провайдер — работает без сети и API-ключей."""

from __future__ import annotations

from collections.abc import Iterator

from ember.providers.base import Provider
from ember.types import ChatRequest, ChatResponse, Message, StreamChunk, Usage


class MockProvider(Provider):
    """Детерминированный провайдер, возвращающий заранее заданный текст.

    Полезен для юнит-тестов, примеров и документации: не требует
    API-ключа и доступа к сети.

    Attributes:
        response_text: Текст, который возвращается на любой запрос.
        model: Имя модели, указываемое в ответе.
    """

    def __init__(
        self,
        response_text: str = "Привет! Я мок-провайдер.",
        model: str = "mock-1",
    ) -> None:
        self.response_text = response_text
        self.model = model

    def complete(self, request: ChatRequest) -> ChatResponse:
        """Возвращает фиксированный ответ, игнорируя содержимое запроса."""
        prompt_tokens = sum(len(m.content.split()) for m in request.messages)
        completion_tokens = len(self.response_text.split())
        return ChatResponse(
            message=Message(role="assistant", content=self.response_text),
            model=self.model,
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )

    def stream(self, request: ChatRequest) -> Iterator[StreamChunk]:
        """Отдаёт ответ по одному слову за фрагмент."""
        words = self.response_text.split()
        for index, word in enumerate(words):
            last = index == len(words) - 1
            yield StreamChunk(
                delta=word if last else word + " ",
                model=self.model,
                finish_reason="stop" if last else None,
            )
