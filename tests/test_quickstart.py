"""Тест: пример из examples/quickstart.py реально исполняется."""

from examples.quickstart import main


def test_quickstart_runs(capsys) -> None:
    """Пример не должен падать и должен выводить ответ мок-провайдера."""
    main()

    captured = capsys.readouterr()
    assert "Привет! Я мок-провайдер." in captured.out
