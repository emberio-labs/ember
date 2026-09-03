"""Вспомогательный MCP-сервер для интеграционных тестов MCPClient.

Запускается как дочерний процесс самими тестами::

    python tests/mcp_server_script.py            # stdio transport
    python tests/mcp_server_script.py --http 9000  # streamable HTTP

Сервер регистрирует несколько инструментов, имитирующих реальный сервис:
арифметику, «погоду», эхо, проверку доступности и инструмент, который
всегда падает с ошибкой.
"""

from __future__ import annotations

import argparse
import asyncio

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Тестовый MCP-сервер ember")
    parser.add_argument("--http", type=int, metavar="PORT", help="запустить по streamable HTTP")
    args = parser.parse_args()

    if args.http is not None:
        asyncio.run(server.run_streamable_http_async(host="127.0.0.1", port=args.http))
    else:
        asyncio.run(server.run_stdio_async())


if __name__ == "__main__":
    main()
