"""MCP-клиент: инструменты внешних MCP-серверов для агента ember.

Официальный MCP SDK (пакет ``mcp``) асинхронный; модуль даёт тонкую
синхронную обёртку: клиент держит фоновый поток с event loop, а методы
``list_tools``/``call_tool`` проксируют вызовы в этот поток и ждут ответа.
Инструменты сервера возвращаются как обычные ``FunctionTool`` — их можно
передавать в ``Agent`` наряду с локальными функциями, агент не знает
о транспорте.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import threading
from collections.abc import Callable, Coroutine
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import TYPE_CHECKING, Any, TypeVar

from ember.types import FunctionTool

if TYPE_CHECKING:
    from types import TracebackType

    from mcp import StdioServerParameters

_T = TypeVar("_T")

_MCP_IMPORT_HINT = (
    "MCPClient требует пакет 'mcp'. Установите его: pip install 'emberio-labs-ember[mcp]'"
)


class MCPError(Exception):
    """Ошибка при работе с MCP-сервером.

    Единый тип для ошибок транспорта и протокола — запуск сервера,
    ``tools/list``, ``tools/call``, таймауты — по образцу ``ProviderError``:
    пользователь не зависит от внутренних исключений MCP SDK.
    """


class MCPClient:
    """Синхронный клиент одного MCP-сервера.

    Подключается к MCP-серверу (stdio или streamable HTTP), получает список
    инструментов (``tools/list``) и умеет вызывать их (``tools/call``).
    Инструменты возвращаются как ``FunctionTool``: их ``func`` — обёртка,
    отправляющая вызов на сервер, поэтому результат ``list_tools()`` можно
    передать в ``Agent`` наряду с локальными инструментами.

    Использование::

        with MCPClient.stdio(command="python", args=["server.py"]) as mcp:
            agent = Agent(provider=provider, tools=mcp.list_tools())
            print(agent.run("..."))

    Args:
        server: Адрес сервера: ``StdioServerParameters`` (дочерний процесс)
            или URL streamable HTTP endpoint.
        timeout: Таймаут на подключение и на каждый вызов (секунды).
    """

    def __init__(
        self,
        server: StdioServerParameters | str,
        *,
        timeout: float = 60.0,
    ) -> None:
        if timeout <= 0:
            raise ValueError("timeout должен быть положительным числом")
        self._server = server
        self._timeout = timeout
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._client: Any = None
        self._connected = False

    # -- фабрики --------------------------------------------------------

    @classmethod
    def stdio(
        cls,
        command: str,
        args: list[str] | None = None,
        *,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        timeout: float = 60.0,
    ) -> MCPClient:
        """Клиент для сервера, запускаемого как дочерний процесс.

        Args:
            command: Команда запуска сервера (например, ``"python"`` или
                ``"npx"``).
            args: Аргументы команды (например, ``["server.py"]``).
            env: Дополнительные переменные окружения процесса.
            cwd: Рабочая директория процесса.
            timeout: Таймаут на подключение и на каждый вызов (секунды).

        Returns:
            Ещё не подключённый ``MCPClient`` — вызовите ``connect()`` или
            используйте как context manager.

        Raises:
            ImportError: Если пакет ``mcp`` не установлен.
        """
        try:
            from mcp import StdioServerParameters
        except ImportError as exc:
            raise ImportError(_MCP_IMPORT_HINT) from exc
        return cls(
            server=StdioServerParameters(command=command, args=args or [], env=env, cwd=cwd),
            timeout=timeout,
        )

    @classmethod
    def http(cls, url: str, *, timeout: float = 60.0) -> MCPClient:
        """Клиент для streamable HTTP endpoint сервера.

        Args:
            url: URL endpoint (например, ``"http://127.0.0.1:8000/mcp"``).
            timeout: Таймаут на подключение и на каждый вызов (секунды).

        Returns:
            Ещё не подключённый ``MCPClient``.

        Raises:
            ValueError: Если URL не похож на HTTP(S).
        """
        if not url.startswith(("http://", "https://")):
            raise ValueError(f"Некорректный URL MCP-сервера: {url!r}")
        return cls(server=url, timeout=timeout)

    # -- жизненный цикл --------------------------------------------------

    @property
    def connected(self) -> bool:
        """Подключён ли клиент к серверу."""
        return self._connected

    def connect(self) -> None:
        """Подключиться к серверу: запустить транспорт и handshake.

        Может вызываться повторно (повторный вызов без ``close()`` ничего
        не делает). Обычно проще использовать context manager:
        ``with MCPClient.stdio(...) as mcp:``.

        Raises:
            MCPError: Если сервер не запустился, недоступен или не ответил
                за ``timeout``.
            ImportError: Если пакет ``mcp`` не установлен.
        """
        if self._connected:
            return
        try:
            import mcp  # noqa: F401
        except ImportError as exc:
            raise ImportError(_MCP_IMPORT_HINT) from exc

        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, name="ember-mcp", daemon=True)
        self._thread.start()
        try:
            self._submit(self._open())
        except BaseException:
            self._shutdown()
            raise

    def close(self) -> None:
        """Закрыть соединение и остановить фоновый поток.

        Безопасно вызывать несколько раз и на неподключённом клиенте.
        """
        if self._thread is None:
            return
        if self._connected:
            with contextlib.suppress(MCPError):
                # Сервер мог упасть раньше нас — это не мешает закрытию.
                self._submit(self._close_session())
        self._shutdown()

    def __enter__(self) -> MCPClient:
        self.connect()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def _run_loop(self) -> None:
        """Тело фонового потока: крутить event loop до остановки."""
        assert self._loop is not None
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _submit(self, coro: Coroutine[Any, Any, _T]) -> _T:
        """Запустить корутину в фоновом loop и дождаться результата.

        Исключения MCP-сессии (таймауты, разрыв соединения, ошибки сервера)
        оборачиваются в ``MCPError``, чтобы пользователь не зависел от SDK.
        Если корутина не была передана в loop (клиент не подключён, loop
        закрыт), она закрывается, а не остаётся висеть незапущенной.
        """
        loop = self._loop
        if loop is None or self._thread is None:
            coro.close()
            raise MCPError("MCPClient не подключён: вызовите connect()")
        try:
            future = asyncio.run_coroutine_threadsafe(coro, loop)
        except Exception as exc:
            coro.close()
            raise MCPError(f"Ошибка MCP: {exc}") from exc
        try:
            return future.result(timeout=self._timeout)
        except FutureTimeoutError:
            raise MCPError(f"MCP-запрос превысил таймаут ({self._timeout:g} c)") from None
        except MCPError:
            raise
        except Exception as exc:
            raise MCPError(f"Ошибка MCP: {exc}") from exc

    def _shutdown(self) -> None:
        """Остановить фоновый поток и закрыть loop (после close или сбоя)."""
        loop = self._loop
        thread = self._thread
        self._loop = None
        self._thread = None
        self._client = None
        self._connected = False
        if loop is not None:
            loop.call_soon_threadsafe(loop.stop)
        if thread is not None:
            thread.join(timeout=5)
        if loop is not None and not loop.is_closed():
            loop.close()

    # -- инструменты -----------------------------------------------------

    def list_tools(self) -> list[FunctionTool]:
        """Получить инструменты сервера как ``FunctionTool``.

        Каждый инструмент (``tools/list``) превращается в ``FunctionTool``:
        JSON Schema параметров сохраняется, а ``func`` — обёртка, которая
        при вызове отправляет ``tools/call`` на сервер. Список можно
        передать в ``Agent`` наряду с локальными инструментами.

        Returns:
            Инструменты сервера, готовые к использованию агентом.

        Raises:
            MCPError: Если клиент не подключён или сервер не ответил.
        """
        tools = self._submit(self._list_tools())
        return [self._to_function_tool(tool) for tool in tools]

    def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> str:
        """Вызвать инструмент сервера и вернуть текстовый результат.

        Args:
            name: Имя инструмента (из ``list_tools()``).
            arguments: Аргументы вызова (JSON-объект). По умолчанию —
                вызов без аргументов.

        Returns:
            Текст результата: текстовые блоки ответа, склеенные
            переносами строк (нетекстовые — JSON-представлением).

        Raises:
            MCPError: Если инструмент вернул ошибку (``is_error``), сервер
                не ответил за таймаут или соединение потеряно.
        """
        return self._submit(self._call_tool(name, arguments))

    async def _open(self) -> None:
        """Создать MCP-сессию (выполняется в фоновом loop)."""
        import mcp

        client = mcp.Client(server=self._server, read_timeout_seconds=self._timeout)
        await client.__aenter__()
        self._client = client
        self._connected = True

    async def _close_session(self) -> None:
        """Закрыть MCP-сессию (выполняется в фоновом loop)."""
        client = self._client
        self._client = None
        self._connected = False
        if client is not None:
            await client.__aexit__(None, None, None)

    async def _list_tools(self) -> list[Any]:
        client = self._client
        if client is None:
            raise MCPError("MCPClient не подключён: вызовите connect()")
        result = await client.list_tools()
        return list(result.tools)

    async def _call_tool(self, name: str, arguments: dict[str, Any] | None) -> str:
        client = self._client
        if client is None:
            raise MCPError("MCPClient не подключён: вызовите connect()")
        result = await client.call_tool(name, arguments)
        message = self._result_text(result)
        if result.is_error:
            raise MCPError(f"Инструмент {name!r} вернул ошибку: {message}")
        return message

    def _to_function_tool(self, tool: Any) -> FunctionTool:
        """Сконвертировать инструмент MCP в ``FunctionTool`` агента."""
        return FunctionTool(
            name=tool.name,
            description=tool.description or "",
            parameters=tool.input_schema or None,
            func=self._make_remote(tool.name),
        )

    def _make_remote(self, name: str) -> Callable[..., Any]:
        """Обёртка: вызов MCP-инструмента из цикла исполнения агента.

        Агент вызывает ``func(**arguments)`` с аргументами, распарсенными из
        JSON модели, — проксируем их в ``tools/call`` как есть.
        """

        def remote(**arguments: Any) -> str:
            return self.call_tool(name, arguments)

        return remote

    @staticmethod
    def _result_text(result: Any) -> str:
        """Привести ответ ``tools/call`` к тексту для tool-сообщения."""
        parts: list[str] = []
        for block in result.content or []:
            text = getattr(block, "text", None)
            if isinstance(text, str):
                parts.append(text)
                continue
            # Нетекстовый блок (изображение, аудио, ресурс): отдаём JSON.
            parts.append(MCPClient._block_json(block))
        structured = getattr(result, "structured_content", None)
        if structured is not None and not parts:
            parts.append(MCPClient._json_text(structured))
        return "\n".join(parts)

    @staticmethod
    def _block_json(block: Any) -> str:
        """Сериализовать нетекстовый блок ответа в JSON-строку."""
        dump = getattr(block, "model_dump", None)
        if callable(dump):
            return MCPClient._json_text(dump())
        return MCPClient._json_text(str(block))

    @staticmethod
    def _json_text(value: Any) -> str:
        """Сериализовать произвольное значение в читаемый JSON."""
        dump = getattr(value, "model_dump", None)
        if callable(dump):
            value = dump()
        return json.dumps(value, ensure_ascii=False, default=str)
