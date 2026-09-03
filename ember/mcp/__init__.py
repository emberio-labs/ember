"""MCP-клиент ember: инструменты внешних MCP-серверов для агента.

Подключение к MCP-серверу (stdio или streamable HTTP), получение его
инструментов как ``FunctionTool`` и исполнение ``tools/call`` из цикла
``Agent.run()``. Пакет ``mcp`` (официальный MCP SDK) — optional зависимость:
``pip install "emberio-labs-ember[mcp]"``.
"""

from ember.mcp.client import MCPClient, MCPError

__all__ = ["MCPClient", "MCPError"]
