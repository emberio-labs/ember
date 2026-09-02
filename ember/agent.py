"""Простой агент поверх Provider — минимум кода для чат-диалога."""

from __future__ import annotations

from collections.abc import Iterator

from ember.providers.base import Provider
from ember.types import ChatRequest, Message


class Agent:
    """Чат-агент, работающий поверх любого ``Provider``.

    Хранит историю диалога и системный промпт, формирует ``ChatRequest``
    и возвращает текстовый ответ модели.

    Attributes:
        provider: Провайдер, через который агент общается с моделью.
        model: Модель по умолчанию. Если не задана, провайдер использует
            свою модель по умолчанию.
        messages: Текущая история диалога (включая system-сообщение).
    """

    def __init__(
        self,
        provider: Provider,
        system_prompt: str = "",
        model: str | None = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.messages: list[Message] = []
        if system_prompt:
            self.messages.append(Message(role="system", content=system_prompt))

    def run(self, user_input: str) -> str:
        """Отправить сообщение пользователя и вернуть текстовый ответ.

        Сообщение пользователя и ответ модели добавляются в историю диалога.

        Args:
            user_input: Текст сообщения пользователя.

        Returns:
            Текст ответа модели.
        """
        self.messages.append(Message(role="user", content=user_input))
        response = self.provider.complete(self._request())
        self.messages.append(response.message)
        return response.message.content

    def stream_run(self, user_input: str) -> Iterator[str]:
        """Отправить сообщение пользователя и получить ответ потоком.

        Поведение аналогично ``run``, но ответ возвращается по одному
        фрагменту за раз. По завершении полный ответ попадает в историю.

        Args:
            user_input: Текст сообщения пользователя.

        Yields:
            Фрагменты текста ответа модели.
        """
        self.messages.append(Message(role="user", content=user_input))
        request = ChatRequest(messages=list(self.messages), model=self.model or "", stream=True)
        full_text = ""
        for chunk in self.provider.stream(request):
            full_text += chunk.delta
            yield chunk.delta
        self.messages.append(Message(role="assistant", content=full_text))

    def reset(self) -> None:
        """Очистить историю диалога, оставив только system-промпт."""
        system = next((m for m in self.messages if m.role == "system"), None)
        self.messages = [system] if system is not None else []

    def _request(self) -> ChatRequest:
        # Копия списка: запрос не должен разделять состояние с историей агента,
        # иначе последующие append в self.messages мутируют переданный запрос.
        # Пустая model означает «модель провайдера по умолчанию»: адаптеры
        # (например, OpenAIProvider) подставляют свою дефолтную модель.
        return ChatRequest(messages=list(self.messages), model=self.model or "")
