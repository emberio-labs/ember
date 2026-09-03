"""Тесты MCPClient: подключение к реальному MCP-серверу и интеграция с агентом.

Интеграционные тесты запускают тестовый MCP-сервер
(``tests/mcp_server_script.py``) как дочерний процесс и общаются с ним по
stdio или streamable HTTP — так проверяется реальный транспорт, а не
заглушки.
"""

from __future__ import annotations

import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from ember import Agent, MCPClient, MCPError
from ember.providers.base import Provider
from ember.types import ChatRequest, ChatResponse, FunctionTool, Message, StreamChunk, ToolCall

SERVER_SCRIPT = Path(__file__).parent / "mcp_server_script.py"


def make_stdio_client(timeout: float = 60.0) -> MCPClient:
    """Клиент к тестовому MCP-серверу (запускается как дочерний процесс)."""
    return MCPClient.stdio(command=sys.executable, args=[str(SERVER_SCRIPT)], timeout=timeout)


@pytest.fixture(scope="module")
def mcp_client() -> Iterator[MCPClient]:
    """Один подключённый stdio-клиент на весь модуль."""
    with make_stdio_client() as client:
        yield client


# -- helpers для streamable HTTP ----------------------------------------


def _free_port() -> int:
    """Найти свободный TCP-порт на 127.0.0.1."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _start_http_server(port: int) -> subprocess.Popen[Any]:
    """Запустить тестовый MCP-сервер по streamable HTTP на 127.0.0.1."""
    return subprocess.Popen(
        [sys.executable, str(SERVER_SCRIPT), "--http", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _wait_http_ready(url: str, process: subprocess.Popen[Any]) -> None:
    """Дождаться, пока сервер начнёт отвечать на MCP-запросы."""
    deadline = time.monotonic() + 20.0
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"HTTP MCP-сервер завершился при запуске (код {process.returncode})"
            )
        try:
            with MCPClient.http(url, timeout=2.0) as client:
                client.list_tools()
            return
        except MCPError:
            time.sleep(0.25)
    raise RuntimeError("HTTP MCP-сервер не ответил за отведённое время")


@pytest.fixture(scope="module")
def http_server_url() -> Iterator[str]:
    """URL streamable HTTP endpoint тестового сервера (один на модуль)."""
    port = _free_port()
    process = _start_http_server(port)
    url = f"http://127.0.0.1:{port}/mcp"
    try:
        _wait_http_ready(url, process)
        yield url
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()


# -- список инструментов -------------------------------------------------


def test_connects_and_lists_tools(mcp_client: MCPClient) -> None:
    """tools/list возвращает все инструменты сервера как FunctionTool."""
    tools = mcp_client.list_tools()

    assert isinstance(tools, list)
    assert all(isinstance(tool, FunctionTool) for tool in tools)
    names = {tool.name for tool in tools}
    assert {"plus", "echo", "get_weather", "ping", "fail"} <= names


def test_tool_schema_and_description_are_preserved(mcp_client: MCPClient) -> None:
    """JSON Schema параметров и описание доезжают до FunctionTool."""
    tools = mcp_client.list_tools()
    plus_tool = next(tool for tool in tools if tool.name == "plus")

    assert plus_tool.description == "Сложить два числа."
    assert plus_tool.parameters is not None
    assert plus_tool.parameters["type"] == "object"
    assert set(plus_tool.parameters["properties"]) == {"a", "b"}
    assert plus_tool.parameters["required"] == ["a", "b"]
    assert callable(plus_tool.func)


# -- вызов инструментов --------------------------------------------------


def test_call_tool_returns_text_result(mcp_client: MCPClient) -> None:
    """tools/call возвращает текст результата."""
    assert mcp_client.call_tool("plus", {"a": 2, "b": 3}) == "5"
    assert mcp_client.call_tool("echo", {"text": "привет мир"}) == "привет мир"
    assert mcp_client.call_tool("get_weather", {"city": "Париж"}) == "В городе Париж: +18 °C, ясно"


def test_call_tool_without_arguments(mcp_client: MCPClient) -> None:
    """Инструмент без параметров вызывается без arguments."""
    assert mcp_client.call_tool("ping") == "pong"


def test_function_tool_func_proxies_to_server(mcp_client: MCPClient) -> None:
    """func у FunctionTool — рабочая обёртка над tools/call."""
    tools = mcp_client.list_tools()
    plus_tool = next(tool for tool in tools if tool.name == "plus")
    ping_tool = next(tool for tool in tools if tool.name == "ping")

    assert plus_tool.func(a=10, b=32) == "42"
    assert ping_tool.func() == "pong"


def test_call_tool_server_error_raises_mcp_error(mcp_client: MCPClient) -> None:
    """Ошибка на стороне сервера (is_error) превращается в MCPError."""
    with pytest.raises(MCPError) as exc_info:
        mcp_client.call_tool("fail")

    assert "fail" in str(exc_info.value)
    assert "ошибк" in str(exc_info.value).lower()


def test_call_tool_unknown_name_raises_mcp_error(mcp_client: MCPClient) -> None:
    """Вызов несуществующего инструмента — MCPError."""
    with pytest.raises(MCPError):
        mcp_client.call_tool("no_such_tool")


# -- интеграция с агентом ------------------------------------------------


class ToolCallingMock(Provider):
    """Сценарный мок: первый ответ запрашивает MCP-инструмент, второй — текст.

    Имитирует модель, которая увидела в истории результат инструмента
    (role=tool) и сформулировала ответ пользователю.
    """

    def __init__(self) -> None:
        self._step = 0

    def complete(self, request: ChatRequest) -> ChatResponse:
        self._step += 1
        if self._step == 1:
            return ChatResponse(
                message=Message(
                    role="assistant",
                    content="",
                    tool_calls=[
                        ToolCall(id="call_1", name="get_weather", arguments='{"city": "Париж"}')
                    ],
                ),
                model="mock",
            )
        return ChatResponse(
            message=Message(role="assistant", content="В Париже сейчас +18 °C, ясно."),
            model="mock",
        )

    def stream(self, request: ChatRequest) -> Iterator[StreamChunk]:
        raise AssertionError("stream не используется в тестах MCPClient")


def test_agent_executes_mcp_tool(mcp_client: MCPClient) -> None:
    """Инструменты MCP-сервера исполняются агентом как обычные FunctionTool."""
    agent = Agent(provider=ToolCallingMock(), tools=mcp_client.list_tools())

    result = agent.run("Какая погода в Париже?")

    assert result == "В Париже сейчас +18 °C, ясно."
    tool_messages = [m for m in agent.messages if m.role == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0].content == "В городе Париж: +18 °C, ясно"


def test_agent_mixes_mcp_and_local_tools(mcp_client: MCPClient) -> None:
    """MCP-инструменты и локальные FunctionTool работают в одном агенте."""

    def say_hi(name: str) -> str:
        return f"Привет, {name}!"

    class MixedToolsMock(Provider):
        """Мок: MCP get_weather → локальный say_hi → финальный текст."""

        def __init__(self) -> None:
            self._step = 0

        def complete(self, request: ChatRequest) -> ChatResponse:
            self._step += 1
            if self._step == 1:
                return ChatResponse(
                    message=Message(
                        role="assistant",
                        content="",
                        tool_calls=[
                            ToolCall(
                                id="call_1",
                                name="get_weather",
                                arguments='{"city": "Париж"}',
                            )
                        ],
                    ),
                    model="mock",
                )
            if self._step == 2:
                return ChatResponse(
                    message=Message(
                        role="assistant",
                        content="",
                        tool_calls=[
                            ToolCall(
                                id="call_2",
                                name="say_hi",
                                arguments='{"name": "Ember"}',
                            )
                        ],
                    ),
                    model="mock",
                )
            return ChatResponse(
                message=Message(role="assistant", content="Готово."),
                model="mock",
            )

        def stream(self, request: ChatRequest) -> Iterator[StreamChunk]:
            raise AssertionError("stream не используется в тестах MCPClient")

    local_tool = FunctionTool(
        name="say_hi",
        description="Поприветствовать пользователя по имени.",
        parameters={
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
        func=say_hi,
    )
    agent = Agent(
        provider=MixedToolsMock(),
        tools=[*mcp_client.list_tools(), local_tool],
    )

    result = agent.run("Проверь всё")

    assert result == "Готово."
    tool_contents = [m.content for m in agent.messages if m.role == "tool"]
    assert "В городе Париж: +18 °C, ясно" in tool_contents
    assert "Привет, Ember!" in tool_contents


# -- streamable HTTP транспорт -------------------------------------------


def test_http_transport_lists_and_calls_tools(http_server_url: str) -> None:
    """streamable HTTP: подключение, tools/list и tools/call работают."""
    with MCPClient.http(http_server_url, timeout=15.0) as client:
        tools = client.list_tools()

        names = {tool.name for tool in tools}
        assert {"plus", "echo", "get_weather", "ping", "fail"} <= names
        assert client.call_tool("plus", {"a": 40, "b": 2}) == "42"
        assert client.call_tool("echo", {"text": "http"}) == "http"


def test_http_tools_work_with_agent(http_server_url: str) -> None:
    """Инструменты по HTTP доезжают до агента и исполняются в цикле."""
    with MCPClient.http(http_server_url, timeout=15.0) as client:
        agent = Agent(provider=ToolCallingMock(), tools=client.list_tools())

        result = agent.run("Какая погода в Париже?")

        assert result == "В Париже сейчас +18 °C, ясно."
        tool_messages = [m for m in agent.messages if m.role == "tool"]
        assert len(tool_messages) == 1
        assert tool_messages[0].content == "В городе Париж: +18 °C, ясно"


# -- ошибки использования до подключения ---------------------------------


def test_list_tools_before_connect_raises() -> None:
    """Запрос инструментов без connect() — понятная MCPError."""
    client = MCPClient.http("http://127.0.0.1:9/mcp")

    with pytest.raises(MCPError, match="не подключён"):
        client.list_tools()
    with pytest.raises(MCPError, match="не подключён"):
        client.call_tool("ping")


def test_close_without_connect_is_safe() -> None:
    """close() на неподключённом клиенте не падает (и повторный close тоже)."""
    client = MCPClient.http("http://127.0.0.1:9/mcp")
    client.close()
    client.close()


# -- валидация параметров ------------------------------------------------


def test_invalid_timeout_raises_value_error() -> None:
    with pytest.raises(ValueError, match="timeout"):
        MCPClient.http("http://127.0.0.1:9/mcp", timeout=0)
    with pytest.raises(ValueError, match="timeout"):
        MCPClient.http("http://127.0.0.1:9/mcp", timeout=-5)


def test_invalid_http_url_raises_value_error() -> None:
    with pytest.raises(ValueError, match="URL"):
        MCPClient.http("не-url")


def test_unreachable_http_server_raises_mcp_error() -> None:
    """Подключение к недоступному серверу — MCPError (а не сырое исключение)."""
    client = MCPClient.http("http://127.0.0.1:1/mcp", timeout=3)

    with pytest.raises(MCPError):
        client.connect()

    # После неудачного connect клиент полностью закрыт и переиспользуем.
    client.close()
