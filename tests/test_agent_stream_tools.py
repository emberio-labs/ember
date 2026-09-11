"""Тесты stream_run с инструментами: цикл раундов, история, лимит, память."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from ember.agent import Agent, ToolCallLimitError
from ember.memory import FileMemory
from ember.providers.base import Provider
from ember.types import (
    ChatRequest,
    ChatResponse,
    FunctionTool,
    Message,
    StreamChunk,
    ToolCall,
)


def _text_chunks(text: str, model: str = "stream-1", finish: bool = True) -> list[StreamChunk]:
    """Разбить текст на потоковые фрагменты (по словам)."""
    words = text.split()
    return [
        StreamChunk(
            delta=word if index == len(words) - 1 else word + " ",
            model=model,
            finish_reason="stop" if finish and index == len(words) - 1 else None,
        )
        for index, word in enumerate(words)
    ]


def _tool_round(*calls: ToolCall, text: str = "", model: str = "stream-1") -> list[StreamChunk]:
    """Раунд потока: необязательный текст + финальный фрагмент с tool_calls."""
    chunks = _text_chunks(text, model=model, finish=False) if text else []
    chunks.append(
        StreamChunk(delta="", model=model, finish_reason="tool_calls", tool_calls=list(calls))
    )
    return chunks


class ScriptedStreamProvider(Provider):
    """Провайдер-сценарий: каждый раунд — заранее заданный список фрагментов."""

    def __init__(self, rounds: list[list[StreamChunk]]) -> None:
        self.rounds = rounds
        self.requests: list[ChatRequest] = []

    def complete(self, request: ChatRequest) -> ChatResponse:
        raise AssertionError("complete не используется в потоковых тестах tool calling")

    def stream(self, request: ChatRequest) -> Iterator[StreamChunk]:
        self.requests.append(request)
        if len(self.requests) > len(self.rounds):
            raise AssertionError("ScriptedStreamProvider: сценарий фрагментов исчерпан")
        yield from self.rounds[len(self.requests) - 1]


def _tool_call(name: str, arguments: str, call_id: str = "c1") -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments=arguments)


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


def test_stream_run_executes_tool_call_and_returns_final_text() -> None:
    calls: list[dict[str, str]] = []

    def get_weather(city: str) -> dict[str, object]:
        calls.append({"city": city})
        return {"temperature": 18, "condition": "ясно"}

    provider = ScriptedStreamProvider(
        [
            _tool_round(_tool_call("get_weather", '{"city": "Париж"}'), text="Сейчас посмотрю."),
            _text_chunks("В Париже +18 °C"),
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

    chunks = list(agent.stream_run("Какая погода в Париже?"))

    # Текст обоих раундов уходит в один поток без разделителей.
    assert "".join(chunks) == "Сейчас посмотрю.В Париже +18 °C"
    assert calls == [{"city": "Париж"}]
    assert [m.role for m in agent.messages] == ["user", "assistant", "tool", "assistant"]
    tool_result = json.loads(_tool_results(agent)[0].content)
    assert tool_result == {"temperature": 18, "condition": "ясно"}


def test_stream_run_passes_tools_and_stream_flag() -> None:
    provider = ScriptedStreamProvider([_text_chunks("ответ")])
    tool = _function_tool("add", lambda a, b: a + b, {"type": "object"})
    agent = Agent(provider, model="gpt-4o-mini", tools=[tool])

    list(agent.stream_run("Сложи"))

    request = provider.requests[0]
    assert request.stream is True
    assert request.model == "gpt-4o-mini"
    assert request.tools is not None
    assert request.tools[0] is tool


def test_stream_run_chain_of_multiple_rounds() -> None:
    order: list[str] = []

    def get_weather(city: str) -> str:
        order.append("weather")
        return "18"

    def get_time(city: str) -> str:
        order.append("time")
        return "12:00"

    provider = ScriptedStreamProvider(
        [
            _tool_round(_tool_call("get_weather", '{"city": "Париж"}', "c1")),
            _tool_round(_tool_call("get_time", '{"city": "Париж"}', "c2")),
            _text_chunks("Погода +18, время 12:00"),
        ]
    )
    agent = Agent(
        provider,
        tools=[
            _function_tool("get_weather", get_weather, {"type": "object"}),
            _function_tool("get_time", get_time, {"type": "object"}),
        ],
    )

    assert "".join(agent.stream_run("Что в Париже?")) == "Погода +18, время 12:00"
    assert order == ["weather", "time"]
    assert len(provider.requests) == 3
    assert [m.role for m in agent.messages] == [
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
        "assistant",
    ]


def test_stream_run_stores_tool_calls_in_assistant_message() -> None:
    provider = ScriptedStreamProvider(
        [
            _tool_round(_tool_call("noop", "{}")),
            _text_chunks("Готово"),
        ]
    )
    agent = Agent(provider, tools=[_function_tool("noop", lambda: None, {"type": "object"})])

    list(agent.stream_run("Сделай"))

    assistant = agent.messages[1]
    assert assistant.role == "assistant"
    assert assistant.content == ""
    assert assistant.tool_calls == [_tool_call("noop", "{}")]


def test_stream_run_unknown_tool_becomes_error_result() -> None:
    provider = ScriptedStreamProvider(
        [
            _tool_round(_tool_call("no_such_tool", "{}")),
            _text_chunks("Извини, такого инструмента нет"),
        ]
    )
    agent = Agent(
        provider,
        tools=[_function_tool("get_weather", lambda: "ok", {"type": "object"})],
    )

    assert "".join(agent.stream_run("Позови неизвестный инструмент")) == (
        "Извини, такого инструмента нет"
    )
    error_result = _tool_results(agent)[0]
    assert "no_such_tool" in error_result.content
    assert "не найден" in error_result.content


def test_stream_run_broken_json_arguments_become_error_result() -> None:
    provider = ScriptedStreamProvider(
        [
            _tool_round(_tool_call("get_weather", "{не json")),
            _text_chunks("Ок"),
        ]
    )
    agent = Agent(
        provider,
        tools=[
            _function_tool(
                "get_weather",
                lambda city: f"погода в {city}",
                {"type": "object", "properties": {"city": {"type": "string"}}},
            )
        ],
    )

    list(agent.stream_run("Погода"))

    assert "Ошибка" in _tool_results(agent)[0].content


def test_stream_run_respects_max_tool_steps() -> None:
    call = _tool_call("add", '{"a": 1, "b": 2}')
    provider = ScriptedStreamProvider([_tool_round(call) for _ in range(3)])
    agent = Agent(
        provider,
        tools=[_function_tool("add", lambda a, b: a + b, {"type": "object"})],
        max_tool_steps=2,
    )

    with pytest.raises(ToolCallLimitError) as exc_info:
        list(agent.stream_run("Сложи"))

    message = str(exc_info.value)
    assert "2" in message
    assert "add" in message


def test_stream_run_without_tools_appends_assistant_message() -> None:
    provider = ScriptedStreamProvider([_text_chunks("Привет мир")])
    agent = Agent(provider)

    assert "".join(agent.stream_run("Привет")) == "Привет мир"
    assert [(m.role, m.content) for m in agent.messages] == [
        ("user", "Привет"),
        ("assistant", "Привет мир"),
    ]
    assert agent.messages[-1].tool_calls is None
    assert provider.requests[0].tools is None


def test_stream_run_saves_session_after_tool_rounds(tmp_path: Path) -> None:
    memory = FileMemory(tmp_path)
    provider = ScriptedStreamProvider(
        [
            _tool_round(_tool_call("noop", "{}")),
            _text_chunks("Готово"),
        ]
    )
    agent = Agent(
        provider,
        tools=[_function_tool("noop", lambda: None, {"type": "object"})],
        memory=memory,
        session_id="s",
    )

    list(agent.stream_run("Сделай"))

    stored = memory.load_session("s")
    assert [m.role for m in stored] == ["user", "assistant", "tool", "assistant"]
    assert stored[1].tool_calls is not None


def test_stream_run_saves_session_on_partial_consumption(tmp_path: Path) -> None:
    memory = FileMemory(tmp_path)
    provider = ScriptedStreamProvider([_text_chunks("Привет мир")])
    agent = Agent(provider, memory=memory, session_id="s")

    stream = agent.stream_run("Привет")
    next(stream)
    stream.close()

    # Обрыв потока до первого фрагмента assistant: в хранилище остаётся user.
    assert memory.load_session("s") == [Message(role="user", content="Привет")]
