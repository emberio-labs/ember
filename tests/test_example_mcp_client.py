"""Тест: пример из examples/mcp_client.py реально исполняется."""

from examples.mcp_client import main


def test_mcp_client_example_runs(capsys) -> None:
    """Пример поднимает MCP-сервер, исполняет его инструмент и печатает ответ."""
    main()

    captured = capsys.readouterr()
    assert "MCP-сервер доступен: pong" in captured.out
