"""Интеграция агента с MCP-сервером — исполняемый пример без API-ключей.

Файл умеет работать в двух режимах:

* ``python examples/mcp_client.py --serve`` — запустить локальный MCP-сервер
  (stdio-транспорт) с инструментом ``ping``;
* ``python examples/mcp_client.py`` (без аргументов) — подключиться к этому
  серверу как к дочернему процессу, получить инструменты через
  ``tools/list`` и исполнить их агентом (мок-провайдер, ключи не нужны).

Режим ``--serve`` не импортирует ``ember`` вовсе — это честный пример
интеграции: серверная часть живёт отдельным процессом и не знает
о клиенте. Поэтому импорты ``ember`` (клиент, агент, типы) находятся
внутри ``main()``, а не на верхнем уровне модуля.

Файл намеренно НЕ называется ``mcp.py``: имя ``mcp`` занято SDK (пакет
``mcp``), и одноимённый скрипт в ``examples/`` перекрывал бы его при
импорте из дочернего процесса.

В реальном приложении сервер — отдельный процесс (свой или сторонний),
а клиент подключается к нему той же фабрикой ``MCPClient.stdio``.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from mcp.server import MCPServer

SERVER = MCPServer(name="ember-example-mcp", version="0.1.0")


@SERVER.tool()
def ping() -> str:
    """Проверка доступности сервера."""
    return "pong"


def _serve() -> None:
    """Запустить MCP-сервер (stdio) — режим ``--serve``."""
    asyncio.run(SERVER.run_stdio_async())


def main() -> None:
    """Запустить пример: агент исполняет инструмент MCP-сервера.

    Сервер запускается этим же файлом как дочерний процесс
    (``python examples/mcp_client.py --serve``); агент подключается к нему
    по stdio и вызывает инструмент ``ping`` в цикле ``agent.run()``.
    """
    from collections.abc import Iterator

    from ember import Agent, MCPClient
    from ember.providers.base import Provider
    from ember.types import ChatRequest, ChatResponse, Message, StreamChunk, ToolCall

    class PingToolMock(Provider):
        """Сценарный мок: первый ответ просит инструмент ``ping``, второй — текст.

        Имитирует модель: после исполнения инструмента она «видит» результат
        (сообщение role=tool) в истории и формулирует ответ пользователю.
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
                        tool_calls=[ToolCall(id="call_1", name="ping", arguments="{}")],
                    ),
                    model="mock-mcp",
                )
            tool_result = next(
                m.content for m in reversed(request.messages) if m.role == "tool"
            )
            return ChatResponse(
                message=Message(
                    role="assistant",
                    content=f"MCP-сервер доступен: {tool_result}",
                ),
                model="mock-mcp",
            )

        def stream(self, request: ChatRequest) -> Iterator[StreamChunk]:
            raise NotImplementedError("Потоковый режим в примере не используется")

    script = Path(__file__).resolve()
    with MCPClient.stdio(command=sys.executable, args=[str(script), "--serve"]) as mcp:
        agent = Agent(
            provider=PingToolMock(),
            system_prompt="Ты — помощник, проверяющий доступность сервисов.",
            tools=mcp.list_tools(),
        )
        print(agent.run("Проверь, доступен ли MCP-сервер."))


if __name__ == "__main__":
    if "--serve" in sys.argv:
        _serve()
    else:
        main()
