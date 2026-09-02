"""Тест: пример из examples/tools.py реально исполняется."""

from examples.tools import main


def test_tools_example_runs(capsys) -> None:
    """Пример не должен падать и должен выводить финальный ответ агента."""
    main()

    captured = capsys.readouterr()
    assert "В Париже сейчас +18 °C, ясно." in captured.out
