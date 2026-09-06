"""Тест: пример из examples/memory.py реально исполняется."""

from examples.memory import main


def test_memory_example_runs(capsys) -> None:
    """Пример сохраняет диалог, восстанавливает его и показывает recall."""
    main()

    captured = capsys.readouterr()
    assert "Пример завершён." in captured.out
    # Recall из прошлой сессии попал в запрос второй сессии.
    assert "Из прошлых сессий:" in captured.out
