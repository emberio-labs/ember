"""Тесты tool calling в агенте: цикл исполнения инструментов и защита от зацикливания."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator

import pytest

from ember.agent import Agent, ToolCallLimitError
from ember.providers.base import Provider
from ember.types import ChatRequest, ChatResponse, FunctionTool, Message, StreamChunk, ToolCall


class ScriptedProvider(Provider):
    """Провайдер-сценарий: возвращает заранее заданные ответы по порядку.

    Позволяет симулировать модель, которая вызывает инструменты: каждый
    вызов ``complete`` отдаёт следующий Message из ``script`` и записывает
    запрос в ``requests``.
    """

    def __init__(self, script: list[Message]) -> None:
        self.script = script
        self.index = 0
        self.requests: list[ChatRequest] = []

    def complete(self, request: ChatRequest) -> ChatResponse:
        self.requests.append(request)
        if self.index >= len(self.script):
            raise AssertionError("ScriptedProvider: сценарий ответов исчерпан")
        message = self.script[self.index]
        self.index += 1
        return ChatResponse(message=message, model="scripted-1")

    def stream(self, request: ChatRequest) -> Iterator[StreamChunk]:
        raise AssertionError("stream не используется в тестах tool calling")


def _tool_call(name: str, arguments: str, call_id: str) -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments=arguments)


def _assistant_with_calls(*calls: ToolCall) -> Message:
    return Message(role="assistant", content="", tool_calls=list(calls))


def _function_tool(
    name: str,
    func: Callable[..., object],
    parameters: dict[str, object],
) -> FunctionTool:
    return FunctionTool(
        name=name,
        description=f"Инструмент {name}",
        parameters=parameters,
        func=func,
    )


def _tool_results(agent: Agent) -> list[Message]:
    return [m for m in agent.messages if m.role == "tool"]


def test_run_executes_tool_call_and_returns_final_text() -> None:
    calls: list[dict[str, str]] = []

    def get_weather(city: str) -> dict[str, object]:
        calls.append({"city": city})
        return {"temperature": 18, "condition": "ясно"}

    provider = ScriptedProvider(
        script=[
            _assistant_with_calls(_tool_call("get_weather", '{"city": "Париж"}', "c1")),
            Message(role="assistant", content="В Париже сейчас +18 °C"),
        ]
    )
    agent = Agent(
        provider,
        tools=[
            _function_tool(
                "get_weather",
                get_weather,
                {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            )
        ],
    )

    result = agent.run("Какая погода в Париже?")

    assert result == "В Париже сейчас +18 °C"
    assert calls == [{"city": "Париж"}]
    # История: user, assistant с вызовом, tool-результат, финальный assistant.
    assert [m.role for m in agent.messages] == ["user", "assistant", "tool", "assistant"]
    tool_result = json.loads(_tool_results(agent)[0].content)
    assert tool_result == {"temperature": 18, "condition": "ясно"}


def test_multiple_tool_calls_in_one_round() -> None:
    def add(a: int, b: int) -> int:
        return a + b

    provider = ScriptedProvider(
        script=[
            _assistant_with_calls(
                _tool_call("add", '{"a": 2, "b": 3}', "c1"),
                _tool_call("add", '{"a": 10, "b": 20}', "c2"),
            ),
            Message(role="assistant", content="Сумма: 35"),
        ]
    )
    agent = Agent(
        provider,
        tools=[
            _function_tool(
                "add",
                add,
                {
                    "type": "object",
                    "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                    "required": ["a", "b"],
                },
            )
        ],
    )

    assert agent.run("Сложи числа") == "Сумма: 35"

    # Оба вызова исполнены за один раунд — второго запроса к модели не было.
    assert len(provider.requests) == 2
    results = _tool_results(agent)
    assert [m.content for m in results] == ["5", "30"]
    assert [m.tool_call_id for m in results] == ["c1", "c2"]


def test_arguments_passed_as_kwargs() -> None:
    seen: list[dict[str, object]] = []

    def summarize(text: str, limit: int) -> str:
        seen.append({"text": text, "limit": limit})
        return f"OK:{limit}"

    provider = ScriptedProvider(
        script=[
            _assistant_with_calls(
                _tool_call("summarize", '{"text": "длинный текст", "limit": 10}', "c1")
            ),
            Message(role="assistant", content="Готово"),
        ]
    )
    agent = Agent(
        provider,
        tools=[
            _function_tool(
                "summarize",
                summarize,
                {
                    "type": "object",
                    "properties": {"text": {"type": "string"}, "limit": {"type": "integer"}},
                    "required": ["text", "limit"],
                },
            )
        ],
    )

    agent.run("Сократи")

    assert seen == [{"text": "длинный текст", "limit": 10}]


def test_chain_of_multiple_rounds() -> None:
    weather_calls: list[str] = []
    time_calls: list[str] = []

    def get_weather(city: str) -> dict[str, int]:
        weather_calls.append(city)
        return {"temperature": 18}

    def get_time(city: str) -> str:
        time_calls.append(city)
        return "12:00"

    provider = ScriptedProvider(
        script=[
            _assistant_with_calls(_tool_call("get_weather", '{"city": "Париж"}', "c1")),
            _assistant_with_calls(_tool_call("get_time", '{"city": "Париж"}', "c2")),
            Message(role="assistant", content="Погода +18 °C, время 12:00"),
        ]
    )
    agent = Agent(
        provider,
        tools=[
            _function_tool("get_weather", get_weather, {"type": "object"}),
            _function_tool("get_time", get_time, {"type": "object"}),
        ],
    )

    result = agent.run("Что в Париже?")

    assert result == "Погода +18 °C, время 12:00"
    assert weather_calls == ["Париж"]
    assert time_calls == ["Париж"]
    assert len(provider.requests) == 3
    assert [m.role for m in agent.messages] == [
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
        "assistant",
    ]


def test_unknown_tool_name_becomes_error_result_and_continues() -> None:
    provider = ScriptedProvider(
        script=[
            _assistant_with_calls(_tool_call("no_such_tool", "{}", "c1")),
            Message(role="assistant", content="Извини, такого инструмента нет"),
        ]
    )
    agent = Agent(provider, tools=[_function_tool("get_weather", lambda: "ok", {"type": "object"})])

    result = agent.run("Позови неизвестный инструмент")

    assert result == "Извини, такого инструмента нет"
    error_result = _tool_results(agent)[0]
    assert "no_such_tool" in error_result.content
    assert "не найден" in error_result.content


def test_function_exception_becomes_error_result_and_continues() -> None:
    def broken() -> None:
        raise ValueError("сервис недоступен")

    provider = ScriptedProvider(
        script=[
            _assistant_with_calls(_tool_call("broken", "{}", "c1")),
            Message(role="assistant", content="Попробую иначе"),
        ]
    )
    agent = Agent(provider, tools=[_function_tool("broken", broken, {"type": "object"})])

    result = agent.run("Вызови сломанный инструмент")

    assert result == "Попробую иначе"
    error_result = _tool_results(agent)[0]
    assert "сервис недоступен" in error_result.content


def test_broken_json_arguments_become_error_result() -> None:
    def get_weather(city: str) -> str:
        return f"погода в {city}"

    provider = ScriptedProvider(
        script=[
            _assistant_with_calls(_tool_call("get_weather", "{не json", "c1")),
            Message(role="assistant", content="Ок"),
        ]
    )
    agent = Agent(
        provider,
        tools=[
            _function_tool(
                "get_weather",
                get_weather,
                {"type": "object", "properties": {"city": {"type": "string"}}},
            )
        ],
    )

    agent.run("Погода")

    error_result = _tool_results(agent)[0]
    assert "Ошибка" in error_result.content


def test_max_tool_steps_raises_with_diagnostics() -> None:
    def add(a: int, b: int) -> int:
        return a + b

    tool_call = _tool_call("add", '{"a": 1, "b": 2}', "c1")
    provider = ScriptedProvider(script=[_assistant_with_calls(tool_call) for _ in range(3)])
    agent = Agent(
        provider,
        tools=[_function_tool("add", add, {"type": "object"})],
        max_tool_steps=2,
    )

    with pytest.raises(ToolCallLimitError) as exc_info:
        agent.run("Сложи")

    message = str(exc_info.value)
    assert "2" in message
    assert "add" in message


def test_tools_are_sent_in_chat_request() -> None:
    def add(a: int, b: int) -> int:
        return a + b

    tool = _function_tool("add", add, {"type": "object"})
    provider = ScriptedProvider(
        script=[
            _assistant_with_calls(_tool_call("add", '{"a": 1, "b": 2}', "c1")),
            Message(role="assistant", content="3"),
        ]
    )
    agent = Agent(provider, tools=[tool])

    agent.run("Сложи")

    assert len(provider.requests) == 2
    for request in provider.requests:
        assert request.tools is not None
        assert [t.name for t in request.tools] == ["add"]
        assert request.tools[0] is tool


def test_run_without_tools_works_as_before() -> None:
    provider = ScriptedProvider(script=[Message(role="assistant", content="просто ответ")])
    agent = Agent(provider)

    assert agent.run("Привет") == "просто ответ"
    assert len(provider.requests) == 1
    assert provider.requests[0].tools is None


def test_stream_run_with_tools_raises_value_error() -> None:
    agent = Agent(
        ScriptedProvider(script=[Message(role="assistant", content="текст")]),
        tools=[_function_tool("get_weather", lambda: "ok", {"type": "object"})],
    )

    with pytest.raises(ValueError, match="stream_run"):
        list(agent.stream_run("Привет"))


def test_duplicate_tool_names_raise_value_error() -> None:
    provider = ScriptedProvider(script=[Message(role="assistant", content="ответ")])
    tools = [
        _function_tool("add", lambda a, b: a + b, {"type": "object"}),
        _function_tool("add", lambda a, b: a + b, {"type": "object"}),
    ]

    with pytest.raises(ValueError, match="Дубликат"):
        Agent(provider, tools=tools)


def test_invalid_max_tool_steps_raise_value_error() -> None:
    provider = ScriptedProvider(script=[Message(role="assistant", content="ответ")])

    with pytest.raises(ValueError, match="max_tool_steps"):
        Agent(provider, max_tool_steps=0)


def test_none_result_becomes_empty_tool_message() -> None:
    def noop() -> None:
        return None

    provider = ScriptedProvider(
        script=[
            _assistant_with_calls(_tool_call("noop", "{}", "c1")),
            Message(role="assistant", content="Сделано"),
        ]
    )
    agent = Agent(provider, tools=[_function_tool("noop", noop, {"type": "object"})])

    agent.run("Сделай ничего")

    assert _tool_results(agent)[0].content == ""
