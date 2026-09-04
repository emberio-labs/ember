"""Вспомогательный MCP-сервер для интеграционных тестов MCPClient.

Запускается как дочерний процесс самими тестами::

    python tests/mcp_server_script.py            # stdio transport
    python tests/mcp_server_script.py --http 9000  # streamable HTTP
    python tests/mcp_server_script.py --http 9000 \\
        --http-require-header Authorization "Bearer test-token"

Сервер регистрирует несколько инструментов, имитирующих реальный сервис:
арифметику, «погоду», эхо, проверку доступности и инструмент, который
всегда падает с ошибкой.

``--http-require-header`` (только вместе с ``--http``) заставляет сервер
отвечать 401 на любой запрос без ожидаемого HTTP-заголовка — так тесты
проверяют, что клиент реально отправляет заголовки (аутентификация,
API-ключи и т.п.).
"""

from __future__ import annotations

import argparse
import asyncio
from typing import Any

from mcp.server import MCPServer

server = MCPServer(name="ember-test-mcp", version="0.1.0")


@server.tool()
def plus(a: int, b: int) -> int:
    """Сложить два числа."""
    return a + b


@server.tool()
def get_weather(city: str) -> str:
    """Узнать текущую погоду в городе."""
    return f"В городе {city}: +18 °C, ясно"


@server.tool()
def echo(text: str) -> str:
    """Вернуть переданную строку как есть."""
    return text


@server.tool()
def ping() -> str:
    """Проверка доступности сервера."""
    return "pong"


@server.tool()
def fail() -> str:
    """Инструмент, который всегда завершается ошибкой на стороне сервера."""
    raise RuntimeError("инструмент сломан")


class _RequireHeader:
    """ASGI-обёртка над MCP-приложением: без нужного заголовка отвечает 401.

    Используется только в тестах: проверяет заголовок на уровне HTTP до
    того, как запрос попадёт в MCP-обработчик.
    """

    def __init__(self, app: Any, name: str, value: str) -> None:
        self.app = app
        self.name = name.lower().encode("ascii")
        self.value = value.encode("utf-8")

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] == "http":
            headers = {key.lower(): val for key, val in scope.get("headers") or []}
            if headers.get(self.name) != self.value:
                body = b'{"error": "missing or invalid required header"}'
                response_headers = [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode("ascii")),
                ]
                await send(
                    {
                        "type": "http.response.start",
                        "status": 401,
                        "headers": response_headers,
                    }
                )
                await send({"type": "http.response.body", "body": body})
                return
        await self.app(scope, receive, send)


def main() -> None:
    parser = argparse.ArgumentParser(description="Тестовый MCP-сервер ember")
    parser.add_argument("--http", type=int, metavar="PORT", help="запустить по streamable HTTP")
    parser.add_argument(
        "--http-require-header",
        nargs=2,
        metavar=("NAME", "VALUE"),
        help="отвечать 401 без указанного HTTP-заголовка (только с --http)",
    )
    args = parser.parse_args()

    if args.http is not None:
        import uvicorn

        app = server.streamable_http_app()
        if args.http_require_header is not None:
            name, value = args.http_require_header
            app = _RequireHeader(app, name, value)
        uvicorn.run(app, host="127.0.0.1", port=args.http, log_level="warning")
    else:
        asyncio.run(server.run_stdio_async())


if __name__ == "__main__":
    main()
