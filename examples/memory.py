"""Память агента: персистентные сессии и recall — исполняемый пример без API-ключей.

Демонстрирует: диалог автоматически сохраняется в ``FileMemory``, пересозданный
агент с тем же ``session_id`` продолжает беседу с сохранённой истории, а новая
сессия получает recall-контекст из прошлых разговоров (мок печатает его —
видно, что модель «вспомнила» релевантное).
"""

from __future__ import annotations

import tempfile
from collections.abc import Iterator
from pathlib import Path

from ember import Agent, FileMemory
from ember.providers.base import Provider
from ember.types import ChatRequest, ChatResponse, Message, StreamChunk

RECALL_PREFIX = "Из прошлых сессий:"


class MemoryMockProvider(Provider):
    """Мок: печатает recall-контекст из запроса и отвечает фиксированным текстом."""

    def complete(self, request: ChatRequest) -> ChatResponse:
        for message in request.messages:
            if message.role == "system" and message.content.startswith(RECALL_PREFIX):
                print(f"[память] {message.content}")
        return ChatResponse(
            message=Message(role="assistant", content="Записал в память и ответил."),
            model="mock-memory",
        )

    def stream(self, request: ChatRequest) -> Iterator[StreamChunk]:
        raise NotImplementedError("Потоковый режим в примере не используется")


def main() -> None:
    """Прогнать сценарий: сессия → пересоздание агента → recall → reset."""
    with tempfile.TemporaryDirectory(prefix="ember-memory-") as tmp:
        memory = FileMemory(Path(tmp))
        provider = MemoryMockProvider()

        # Сессия 1: обычный диалог; после run() он сам сохраняется в память.
        agent = Agent(provider=provider, memory=memory, session_id="alex")
        print("Сессия 1:", agent.run("Меня зовут Алекс, я люблю Python"))

        # «Перезапуск приложения»: тот же session_id — история восстановлена.
        restored = Agent(provider=provider, memory=memory, session_id="alex")
        print("История после пересоздания:", [m.content for m in restored.messages])

        # Сессия 2 с новым session_id: агент вспоминает релевантное из прошлой.
        second = Agent(provider=provider, memory=memory, session_id="second")
        print("Сессия 2:", second.run("Как меня зовут и что я люблю?"))

        # reset(): текущая сессия начинается заново (архив прошлых сессий цел).
        restored.reset()
        print("История после reset:", [m.content for m in restored.messages])
        print("Пример завершён.")


if __name__ == "__main__":
    main()
