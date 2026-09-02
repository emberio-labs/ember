"""Юнит-тесты моделей данных ядра ember."""

import pytest

from ember.types import ChatRequest, ChatResponse, Message, StreamChunk, Tool, ToolCall, Usage


def test_message_defaults() -> None:
    msg = Message(role="user", content="Привет")
    assert msg.role == "user"
    assert msg.content == "Привет"
    assert msg.name is None
    assert msg.tool_calls is None
    assert msg.tool_call_id is None


def test_message_with_name() -> None:
    msg = Message(role="assistant", content="Здравствуйте", name="alice")
    assert msg.name == "alice"


def test_message_with_tool_calls() -> None:
    tool_call = ToolCall(id="call_1", name="get_weather", arguments='{"city": "Moscow"}')
    msg = Message(role="assistant", content="", tool_calls=[tool_call])
    assert msg.tool_calls == [tool_call]


def test_tool_message_with_tool_call_id() -> None:
    msg = Message(role="tool", content="+15C", tool_call_id="call_1")
    assert msg.tool_call_id == "call_1"


def test_tool_message_without_tool_call_id_raises() -> None:
    with pytest.raises(ValueError, match="tool_call_id"):
        Message(role="tool", content="+15C")


def test_message_invalid_role_raises() -> None:
    with pytest.raises(ValueError, match="Недопустимая роль"):
        Message(role="admin", content="x")  # type: ignore[arg-type]


def test_tool_call_defaults() -> None:
    tool_call = ToolCall(id="call_1", name="get_weather")
    assert tool_call.arguments == "{}"


def test_tool_defaults() -> None:
    tool = Tool(name="get_weather")
    assert tool.description == ""
    assert tool.parameters is None


def test_tool_full() -> None:
    parameters = {"type": "object", "properties": {"city": {"type": "string"}}}
    tool = Tool(name="get_weather", description="Погода в городе", parameters=parameters)
    assert tool.description == "Погода в городе"
    assert tool.parameters == parameters


def test_tool_empty_name_raises() -> None:
    with pytest.raises(ValueError, match="Имя инструмента"):
        Tool(name="")


def test_chat_request_defaults() -> None:
    request = ChatRequest(messages=[Message(role="user", content="Привет")], model="mock-1")
    assert request.model == "mock-1"
    assert request.temperature == 1.0
    assert request.max_tokens is None
    assert request.stream is False
    assert request.tools is None


def test_chat_request_full() -> None:
    request = ChatRequest(
        messages=[Message(role="user", content="Привет")],
        model="mock-1",
        temperature=0.5,
        max_tokens=128,
        stream=True,
        tools=[Tool(name="get_weather")],
    )
    assert request.temperature == 0.5
    assert request.max_tokens == 128
    assert request.stream is True
    assert request.tools == [Tool(name="get_weather")]


def test_chat_request_empty_messages_raises() -> None:
    with pytest.raises(ValueError, match="не может быть пустым"):
        ChatRequest(messages=[], model="mock-1")


def test_chat_request_negative_temperature_raises() -> None:
    with pytest.raises(ValueError, match="temperature"):
        ChatRequest(messages=[Message(role="user", content="x")], model="mock-1", temperature=-0.1)


def test_chat_request_non_positive_max_tokens_raises() -> None:
    with pytest.raises(ValueError, match="max_tokens"):
        ChatRequest(messages=[Message(role="user", content="x")], model="mock-1", max_tokens=0)


def test_usage_fields() -> None:
    usage = Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    assert usage.prompt_tokens == 10
    assert usage.completion_tokens == 5
    assert usage.total_tokens == 15


def test_chat_response_defaults() -> None:
    response = ChatResponse(message=Message(role="assistant", content="ответ"), model="mock-1")
    assert response.usage is None


def test_chat_response_with_usage() -> None:
    usage = Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2)
    response = ChatResponse(
        message=Message(role="assistant", content="ответ"),
        model="mock-1",
        usage=usage,
    )
    assert response.usage == usage


def test_stream_chunk_defaults() -> None:
    chunk = StreamChunk(delta="часть", model="mock-1")
    assert chunk.finish_reason is None


def test_stream_chunk_finish_reason() -> None:
    chunk = StreamChunk(delta="конец", model="mock-1", finish_reason="stop")
    assert chunk.finish_reason == "stop"
