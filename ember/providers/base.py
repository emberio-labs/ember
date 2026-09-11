"""Абстрактный интерфейс LLM-провайдера."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator

from ember.types import ChatRequest, ChatResponse, StreamChunk


class ProviderError(Exception):
    """Ошибка при обращении к LLM-провайдеру.

    Адаптеры оборачивают ошибки SDK (сеть, неверный ключ, код ответа)
    в этот тип, чтобы пользователь работал с единым интерфейсом ошибок.
    """


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

        Текст отдаётся по мере генерации. Если модель решила вызвать
        инструменты, адаптер обязан сам собрать вызовы из фрагментов потока
        (SDK присылает их по частям: id и имя в начале, JSON-аргументы —
        накоплением) и отдать ровно один финальный ``StreamChunk`` с пустым
        ``delta`` и заполненным ``tool_calls``. Разбор формата дельт — дело
        адаптера, а не потребителя: ``Agent.stream_run()`` получает уже
        готовые ``ToolCall`` и работает с ними так же, как в ``complete()``.

        Args:
            request: Запрос к модели.

        Returns:
            Итератор фрагментов ответа.
        """
