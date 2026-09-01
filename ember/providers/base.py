"""Абстрактный интерфейс LLM-провайдера."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator

from ember.types import ChatRequest, ChatResponse, StreamChunk


class Provider(ABC):
    """Единый интерфейс для LLM-провайдеров.

    Каждый адаптер (OpenAI, Anthropic, Gemini, локальные модели) реализует
    методы complete и stream, конвертируя свой формат в модели ядра ember.
    """

    @abstractmethod
    def complete(self, request: ChatRequest) -> ChatResponse:
        """Получить полный ответ модели на запрос.

        Args:
            request: Запрос к модели.

        Returns:
            Полный ответ модели.
        """

    @abstractmethod
    def stream(self, request: ChatRequest) -> Iterator[StreamChunk]:
        """Получить ответ модели потоком — по одному фрагменту за раз.

        Args:
            request: Запрос к модели.

        Returns:
            Итератор фрагментов ответа.
        """
