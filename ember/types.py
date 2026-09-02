"""Модели данных ядра ember — общий язык для всех провайдеров.

Эти типы не зависят от конкретных SDK: каждый адаптер (OpenAI, Anthropic,
Gemini, локальные модели) конвертирует свой формат в модели ядра и обратно.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, get_args

Role = Literal["system", "user", "assistant", "tool"]
"""Допустимые роли участников диалога."""

_VALID_ROLES: tuple[str, ...] = get_args(Role)


@dataclass(slots=True)
class Message:
    """Одно сообщение в диалоге.

    Attributes:
        role: Роль отправителя: system, user, assistant или tool.
        content: Текст сообщения. Для assistant-сообщений с вызовами
            инструментов может быть пустой строкой.
        name: Опциональное имя участника — позволяет модели различать
            нескольких участников с одной ролью (например, нескольких
            пользователей или ассистентов в одном диалоге).
        tool_calls: Запрошенные моделью вызовы инструментов (только для
            роли assistant).
        tool_call_id: Идентификатор вызова инструмента, на который отвечает
            это сообщение (обязателен для роли tool).
    """

    role: Role
    content: str
    name: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None

    def __post_init__(self) -> None:
        if self.role not in _VALID_ROLES:
            raise ValueError(
                f"Недопустимая роль: {self.role!r}. Ожидается одна из: {', '.join(_VALID_ROLES)}"
            )
        if self.role == "tool" and not self.tool_call_id:
            raise ValueError("Сообщение с ролью 'tool' должно содержать tool_call_id")


@dataclass(slots=True)
class ToolCall:
    """Запрос модели на вызов инструмента.

    Attributes:
        id: Уникальный идентификатор вызова — по нему сопоставляется
            результат выполнения (см. Message.tool_call_id).
        name: Имя вызываемой функции.
        arguments: Аргументы функции в виде JSON-строки.
    """

    id: str
    name: str
    arguments: str = "{}"


@dataclass(slots=True)
class Tool:
    """Описание функции, доступной модели для вызова (tool calling).

    Attributes:
        name: Имя функции, которое модель укажет в tool_calls.
        description: Описание: для чего функция, какие параметры — помогает
            модели выбрать подходящий инструмент.
        parameters: JSON Schema параметров функции (например,
            ``{"type": "object", "properties": {...}}``).
    """

    name: str
    description: str = ""
    parameters: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Имя инструмента не может быть пустым")


@dataclass(slots=True)
class Usage:
    """Счётчики токенов за один запрос.

    Attributes:
        prompt_tokens: Токенов потрачено на вход (промпт).
        completion_tokens: Токенов потрачено на ответ.
        total_tokens: Суммарное количество токенов.
    """

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass(slots=True)
class ChatRequest:
    """Запрос к LLM-провайдеру.

    Attributes:
        messages: История диалога. Не может быть пустой.
        model: Имя модели у провайдера (например, "gpt-4o-mini").
        temperature: Креативность ответа, от 0.0 (детерминированно) и выше.
        max_tokens: Максимум токенов в ответе. None — ограничение провайдера.
        stream: Использовать ли потоковый режим ответа.
        tools: Описания функций, которые модель может вызвать.
    """

    messages: list[Message]
    model: str
    temperature: float = 1.0
    max_tokens: int | None = None
    stream: bool = False
    tools: list[Tool] | None = None

    def __post_init__(self) -> None:
        if not self.messages:
            raise ValueError("Список сообщений не может быть пустым")
        if self.temperature < 0:
            raise ValueError("temperature не может быть отрицательной")
        if self.max_tokens is not None and self.max_tokens <= 0:
            raise ValueError("max_tokens должен быть положительным числом")


@dataclass(slots=True)
class ChatResponse:
    """Ответ модели на ChatRequest.

    Attributes:
        message: Сообщение с ответом модели (роль assistant). Если модель
            решила вызвать инструменты, заполнено поле message.tool_calls.
        model: Модель, которая сформировала ответ.
        usage: Счётчики токенов, если провайдер их вернул.
    """

    message: Message
    model: str
    usage: Usage | None = None


@dataclass(slots=True)
class StreamChunk:
    """Фрагмент потокового ответа модели.

    Attributes:
        delta: Часть текста ответа, пришедшая в этом фрагменте.
        model: Модель, из которой пришёл фрагмент.
        finish_reason: Причина завершения ("stop", "length", ...) для
            последнего фрагмента, иначе None.
    """

    delta: str
    model: str
    finish_reason: str | None = None
