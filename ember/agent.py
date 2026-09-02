"""Простой агент поверх Provider — минимум кода для чат-диалога."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

from ember.providers.base import Provider
from ember.types import ChatRequest, FunctionTool, Message, ToolCall


class ToolCallLimitError(RuntimeError):
    """Модель превысила лимит раундов tool calling.

    Поднимается, когда модель снова и снова запрашивает инструменты вместо
    финального текстового ответа: она зациклилась либо задача не решается
    доступными инструментами за отведённое число шагов.
    """


class Agent:
    """Чат-агент, работающий поверх любого ``Provider``.

    Хранит историю диалога и системный промпт, формирует ``ChatRequest``
    и возвращает текстовый ответ модели.

    Если агенту заданы инструменты (``tools``), ``run()`` исполняет
    запрошенные моделью вызовы в цикле: результат каждого инструмента
    возвращается модели как tool-сообщение, и диалог продолжается до тех
    пор, пока модель не даст финальный текстовый ответ.

    Attributes:
        provider: Провайдер, через который агент общается с моделью.
        model: Модель по умолчанию. Если не задана, провайдер использует
            свою модель по умолчанию.
        tools: Инструменты, доступные модели (описание + функция).
        max_tool_steps: Максимум раундов исполнения инструментов за один
            ``run()``. Один раунд — один запрос к модели и исполнение всех
            запрошенных в ответе вызовов.
        messages: Текущая история диалога (включая system-сообщение).
    """

    def __init__(
        self,
        provider: Provider,
        system_prompt: str = "",
        model: str | None = None,
        tools: list[FunctionTool] | None = None,
        max_tool_steps: int = 10,
    ) -> None:
        self.provider = provider
        self.model = model
        self.tools = tools
        self.max_tool_steps = max_tool_steps
        self.messages: list[Message] = []
        if system_prompt:
            self.messages.append(Message(role="system", content=system_prompt))
        self._validate()
        self._tool_index: dict[str, FunctionTool] = (
            {tool.name: tool for tool in tools} if tools else {}
        )

    def _validate(self) -> None:
        """Проверить параметры конструктора, чтобы ошибки падали сразу."""
        if self.max_tool_steps < 1:
            raise ValueError(
                f"max_tool_steps должен быть положительным числом, получено: {self.max_tool_steps}"
            )
        if self.tools is not None:
            seen: set[str] = set()
            for tool in self.tools:
                if tool.name in seen:
                    raise ValueError(f"Дубликат имени инструмента: {tool.name!r}")
                seen.add(tool.name)

    def run(self, user_input: str) -> str:
        """Отправить сообщение пользователя и вернуть текстовый ответ.

        Если у агента заданы инструменты и модель запрашивает их вызов,
        ``run()`` исполняет инструменты и возвращает результат модели в
        диалог до тех пор, пока модель не даст финальный текстовый ответ
        (без ``tool_calls``). Все промежуточные сообщения (assistant с
        вызовами, tool-результаты) попадают в историю.

        Args:
            user_input: Текст сообщения пользователя.

        Returns:
            Финальный текст ответа модели.

        Raises:
            ToolCallLimitError: Если модель не завершила диалог за
                ``max_tool_steps`` раундов исполнения инструментов.
        """
        self.messages.append(Message(role="user", content=user_input))
        tool_steps = 0
        while True:
            response = self.provider.complete(self._request())
            assistant_message = response.message
            self.messages.append(assistant_message)
            if not assistant_message.tool_calls:
                return assistant_message.content
            if tool_steps >= self.max_tool_steps:
                names = ", ".join(call.name for call in assistant_message.tool_calls)
                raise ToolCallLimitError(
                    f"Модель не завершила диалог: превышен лимит раундов tool calling "
                    f"({self.max_tool_steps}). Модель продолжает запрашивать инструменты "
                    f"({names or 'без имён'}) вместо финального текстового ответа."
                )
            self._execute_tool_calls(assistant_message.tool_calls)
            tool_steps += 1

    def _execute_tool_calls(self, tool_calls: list[ToolCall]) -> None:
        """Исполнить инструменты и добавить результаты в историю как tool-сообщения.

        Любая ошибка (неизвестное имя, битые аргументы, исключение в функции)
        не роняет диалог, а уходит модели текстом ошибки — модель может
        скорректировать запрос.
        """
        for call in tool_calls:
            tool = self._tool_index.get(call.name)
            if tool is None:
                available = ", ".join(sorted(self._tool_index)) or "нет"
                content = (
                    f"Ошибка: инструмент {call.name!r} не найден. "
                    f"Доступные инструменты: {available}."
                )
            else:
                try:
                    arguments = json.loads(call.arguments or "{}")
                    if not isinstance(arguments, dict):
                        raise ValueError("аргументы должны быть JSON-объектом")
                    content = self._stringify_result(tool.func(**arguments))
                except Exception as exc:
                    content = f"Ошибка при вызове {call.name!r}: {exc}"
            self.messages.append(Message(role="tool", content=content, tool_call_id=call.id))

    @staticmethod
    def _stringify_result(result: Any) -> str:
        """Привести результат функции к строке для tool-сообщения.

        Структуры сериализуются в JSON (читаемо, без \\u-escape), None означает
        «успех без данных» и превращается в пустую строку.
        """
        if result is None:
            return ""
        if isinstance(result, dict | list):
            return json.dumps(result, ensure_ascii=False, default=str)
        return str(result)

    def stream_run(self, user_input: str) -> Iterator[str]:
        """Отправить сообщение пользователя и получить ответ потоком.

        Поведение аналогично ``run``, но ответ возвращается по одному
        фрагменту за раз. По завершении полный ответ попадает в историю.

        Args:
            user_input: Текст сообщения пользователя.

        Yields:
            Фрагменты текста ответа модели.

        Raises:
            ValueError: Если у агента заданы инструменты — потоковый ответ
                не разбирает ``tool_calls``, используйте ``run()``.
        """
        if self.tools:
            raise ValueError(
                "stream_run() не поддерживает инструменты: потоковый ответ провайдера "
                "не разбирает tool_calls. Используйте run() для сценариев с tools."
            )
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
        return ChatRequest(
            messages=list(self.messages),
            model=self.model or "",
            tools=self.tools,
        )
