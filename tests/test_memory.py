"""Тесты памяти агента: FileMemory (round-trip, поиск) и Agent с memory/session_id."""

from collections.abc import Iterator
from pathlib import Path

import pytest

from ember import Agent, FileMemory, InvalidSessionIdError, Memory, MockProvider
from ember.memory import validate_session_id
from ember.providers.base import Provider
from ember.types import ChatRequest, ChatResponse, Message, StreamChunk, ToolCall, Usage


class RecordingProvider(Provider):
    """Провайдер-шпион: записывает все запросы и возвращает фиксированный ответ."""

    def __init__(self, response_text: str = "ответ агента") -> None:
        self.response_text = response_text
        self.requests: list[ChatRequest] = []

    def complete(self, request: ChatRequest) -> ChatResponse:
        self.requests.append(request)
        return ChatResponse(
            message=Message(role="assistant", content=self.response_text),
            model="rec-1",
            usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )

    def stream(self, request: ChatRequest) -> Iterator[StreamChunk]:
        self.requests.append(request)
        words = self.response_text.split()
        for index, word in enumerate(words):
            last = index == len(words) - 1
            yield StreamChunk(
                delta=word if last else word + " ",
                model="rec-1",
                finish_reason="stop" if last else None,
            )


class FailingProvider(Provider):
    """Провайдер, который падает на первом же запросе."""

    def complete(self, request: ChatRequest) -> ChatResponse:
        raise RuntimeError("провайдер упал")

    def stream(self, request: ChatRequest) -> Iterator[StreamChunk]:
        raise RuntimeError("провайдер упал")


def test_memory_is_abstract_interface() -> None:
    """Интерфейс Memory нельзя инстанцировать напрямую."""
    with pytest.raises(TypeError):
        Memory()  # type: ignore[abstract]


def test_file_memory_round_trip(tmp_path: Path) -> None:
    """Сообщения (включая tool_calls) переживают save → load без потерь."""
    memory = FileMemory(tmp_path)
    messages = [
        Message(role="user", content="Какая погода?"),
        Message(
            role="assistant",
            content="",
            tool_calls=[ToolCall(id="call_1", name="get_weather", arguments='{"city": "Париж"}')],
        ),
        Message(role="tool", content="+18 °C", tool_call_id="call_1"),
        Message(role="user", content="Спасибо!", name="пользователь"),
    ]

    memory.save_session("demo", messages)
    loaded = memory.load_session("demo")

    assert [(m.role, m.content) for m in loaded] == [
        ("user", "Какая погода?"),
        ("assistant", ""),
        ("tool", "+18 °C"),
        ("user", "Спасибо!"),
    ]
    assert loaded[1].tool_calls is not None
    assert loaded[1].tool_calls[0] == ToolCall(
        id="call_1",
        name="get_weather",
        arguments='{"city": "Париж"}',
    )
    assert loaded[2].tool_call_id == "call_1"
    assert loaded[3].name == "пользователь"


def test_file_memory_load_missing_session_returns_empty(tmp_path: Path) -> None:
    memory = FileMemory(tmp_path)

    assert memory.load_session("missing") == []


def test_file_memory_save_empty_session(tmp_path: Path) -> None:
    memory = FileMemory(tmp_path)

    memory.save_session("empty", [])
    assert memory.load_session("empty") == []


def test_file_memory_delete_session(tmp_path: Path) -> None:
    memory = FileMemory(tmp_path)
    memory.save_session("temp", [Message(role="user", content="привет")])

    memory.delete_session("temp")

    assert memory.load_session("temp") == []
    # Повторное удаление отсутствующей сессии — не ошибка.
    memory.delete_session("temp")


@pytest.mark.parametrize("session_id", ["../outside", "a/b", "/etc/passwd", "..", "."])
def test_file_memory_rejects_session_id_escaping_directory(tmp_path: Path, session_id: str) -> None:
    """id, пытающийся выйти за пределы каталога, — отказ, а не санитизация."""
    memory = FileMemory(tmp_path)

    with pytest.raises(InvalidSessionIdError):
        memory.save_session(session_id, [Message(role="user", content="секрет")])

    assert list(tmp_path.iterdir()) == []
    assert not (tmp_path.parent / ".._outside.json").exists()
    assert not (tmp_path.parent / "outside.json").exists()


def test_file_memory_search_ranks_by_overlap(tmp_path: Path) -> None:
    memory = FileMemory(tmp_path)
    memory.save_session(
        "alex",
        [
            Message(role="user", content="Меня зовут Алекс, я люблю Python"),
            Message(role="assistant", content="Приятно познакомиться, Алекс!"),
        ],
    )
    memory.save_session(
        "weather",
        [Message(role="user", content="Какая погода в Париже?")],
    )

    # Сообщение с двумя совпавшими словами выше, чем с одним.
    found = memory.search("Алекс Python")
    assert [m.content for m in found] == [
        "Меня зовут Алекс, я люблю Python",
        "Приятно познакомиться, Алекс!",
    ]

    found_weather = memory.search("погода в Париже")
    assert [m.content for m in found_weather] == ["Какая погода в Париже?"]


def test_file_memory_search_excludes_session(tmp_path: Path) -> None:
    memory = FileMemory(tmp_path)
    memory.save_session("only", [Message(role="user", content="люблю питон")])

    assert memory.search("питон", exclude_session_id="only") == []
    assert len(memory.search("питон")) == 1


def test_file_memory_search_ignores_noise(tmp_path: Path) -> None:
    """Стоп-слова и короткие слова не дают ложных совпадений."""
    memory = FileMemory(tmp_path)
    memory.save_session("s", [Message(role="user", content="и в на по для")])

    assert memory.search("и в на по для") == []
    assert memory.search("") == []


def test_agent_requires_session_id_with_memory(tmp_path: Path) -> None:
    memory = FileMemory(tmp_path)
    with pytest.raises(ValueError, match="вместе"):
        Agent(MockProvider(), memory=memory)
    with pytest.raises(ValueError, match="вместе"):
        Agent(MockProvider(), session_id="solo")


def test_agent_continues_after_recreation(tmp_path: Path) -> None:
    """Пересозданный агент с тем же session_id продолжает диалог с истории."""
    memory = FileMemory(tmp_path)
    agent = Agent(MockProvider(response_text="ответ"), memory=memory, session_id="demo")
    agent.run("первый вопрос")
    agent.run("второй вопрос")

    restored = Agent(MockProvider(response_text="ответ"), memory=memory, session_id="demo")

    assert [m.content for m in restored.messages] == [
        "первый вопрос",
        "ответ",
        "второй вопрос",
        "ответ",
    ]


def test_agent_does_not_store_system_in_memory(tmp_path: Path) -> None:
    memory = FileMemory(tmp_path)
    agent = Agent(
        MockProvider(response_text="ответ"),
        system_prompt="Ты полезный помощник",
        memory=memory,
        session_id="demo",
    )
    agent.run("привет")

    stored = memory.load_session("demo")
    assert all(m.role != "system" for m in stored)
    assert [m.content for m in stored] == ["привет", "ответ"]


def test_agent_reset_clears_current_session_in_store(tmp_path: Path) -> None:
    memory = FileMemory(tmp_path)
    agent = Agent(
        MockProvider(response_text="ответ"),
        system_prompt="Ты помощник",
        memory=memory,
        session_id="demo",
    )
    agent.run("привет")
    agent.run("пока")

    agent.reset()

    assert [m.role for m in agent.messages] == ["system"]
    assert memory.load_session("demo") == []
    # Повторный run после reset пишет только новую порцию диалога.
    agent.run("снова")
    assert [m.content for m in memory.load_session("demo")] == ["снова", "ответ"]


def test_agent_recall_injects_past_context(tmp_path: Path) -> None:
    """Новая сессия получает recall из прошлых, не мутируя историю и store."""
    memory = FileMemory(tmp_path)
    past = Agent(MockProvider(response_text="ок"), memory=memory, session_id="old")
    past.run("Меня зовут Алекс, я люблю Python")

    recorder = RecordingProvider(response_text="Привет, Алекс!")
    agent = Agent(recorder, memory=memory, session_id="new")
    agent.run("Какое у меня имя?")

    request = recorder.requests[0]
    recall = [
        m
        for m in request.messages
        if m.role == "system" and m.content.startswith("Из прошлых сессий")
    ]
    assert len(recall) == 1
    assert "люблю Python" in recall[0].content
    # Recall-сообщение не попадает ни в историю агента, ни в хранилище.
    assert not any(m.content.startswith("Из прошлых сессий") for m in agent.messages)
    assert all(not m.content.startswith("Из прошлых сессий") for m in memory.load_session("new"))


def test_agent_recall_skips_current_session(tmp_path: Path) -> None:
    """Если других сессий нет — recall пуст, запрос без system-вставки."""
    memory = FileMemory(tmp_path)
    recorder = RecordingProvider(response_text="ответ")
    agent = Agent(recorder, memory=memory, session_id="solo")
    agent.run("уникальное слово квазидракон")

    request = recorder.requests[0]
    assert all(not m.content.startswith("Из прошлых сессий") for m in request.messages)


def test_agent_saves_history_on_error(tmp_path: Path) -> None:
    """При падении провайдера диалог всё равно сохраняется (try/finally)."""
    memory = FileMemory(tmp_path)
    agent = Agent(FailingProvider(), memory=memory, session_id="s")
    with pytest.raises(RuntimeError):
        agent.run("привет")

    assert [m.content for m in memory.load_session("s")] == ["привет"]


def test_agent_stream_run_saves_history(tmp_path: Path) -> None:
    memory = FileMemory(tmp_path)
    agent = Agent(MockProvider(response_text="Привет мир"), memory=memory, session_id="s")

    chunks = list(agent.stream_run("Привет"))

    assert "".join(chunks) == "Привет мир"
    assert [m.content for m in memory.load_session("s")] == ["Привет", "Привет мир"]


@pytest.mark.parametrize(
    "session_id",
    ["", ".hidden", "abc.", "a b", "a#b", "ручная", "café", "x" * 129],
)
def test_file_memory_rejects_unportable_session_id(tmp_path: Path, session_id: str) -> None:
    """Нелатинские и «составные» id отклоняются явно — раньше портили данные молча."""
    memory = FileMemory(tmp_path)

    with pytest.raises(InvalidSessionIdError):
        memory.save_session(session_id, [Message(role="user", content="секрет")])

    assert list(tmp_path.iterdir()) == []


def test_file_memory_invalid_session_id_fails_in_every_method(tmp_path: Path) -> None:
    """Проверка одна на весь API: load/delete/search тоже отказывают."""
    memory = FileMemory(tmp_path)
    memory.save_session("demo", [Message(role="user", content="привет")])

    with pytest.raises(InvalidSessionIdError):
        memory.load_session("ручная")
    with pytest.raises(InvalidSessionIdError):
        memory.delete_session("ручная")
    with pytest.raises(InvalidSessionIdError):
        memory.search("привет", exclude_session_id="ручная")

    # Сессия с валидным id не пострадала.
    assert [m.content for m in memory.load_session("demo")] == ["привет"]


def test_file_memory_similar_ids_no_longer_collide(tmp_path: Path) -> None:
    """Регрессия #44: id, схлопывавшиеся в одно имя файла, больше не затирают друг друга."""
    memory = FileMemory(tmp_path)
    memory.save_session("a-b", [Message(role="user", content="первый")])
    memory.save_session("a_b", [Message(role="user", content="второй")])

    assert [m.content for m in memory.load_session("a-b")] == ["первый"]
    assert [m.content for m in memory.load_session("a_b")] == ["второй"]
    assert sorted(path.name for path in tmp_path.iterdir()) == ["a-b.json", "a_b.json"]

    # «ручная» и «ручной» раньше давали один "______.json" — теперь обе отклоняются.
    for bad_id in ("ручная", "ручной"):
        with pytest.raises(InvalidSessionIdError):
            memory.save_session(bad_id, [Message(role="user", content="потеряно")])
    assert sorted(path.name for path in tmp_path.iterdir()) == ["a-b.json", "a_b.json"]


@pytest.mark.parametrize("session_id", ["demo", "user-42", "a_b", "2026.09.10", "A"])
def test_file_memory_accepts_portable_session_id(tmp_path: Path, session_id: str) -> None:
    """Допустимый id и есть имя файла без расширения — предсказуемо и обратимо."""
    memory = FileMemory(tmp_path)
    memory.save_session(session_id, [Message(role="user", content="привет")])

    assert (tmp_path / f"{session_id}.json").exists()
    assert [m.content for m in memory.load_session(session_id)] == ["привет"]


def test_file_memory_search_ignores_foreign_files(tmp_path: Path) -> None:
    """Посторонние и легаси-файлы (имена не-ids) не ломают поиск и не видны в нём."""
    memory = FileMemory(tmp_path)
    memory.save_session("demo", [Message(role="user", content="люблю питон")])
    legacy = '{"role": "user", "content": "люблю питон"}\n'
    (tmp_path / ".._outside.json").write_text(legacy, encoding="utf-8")
    (tmp_path / "заметки.json").write_text(legacy, encoding="utf-8")
    (tmp_path / "notes.txt").write_text(legacy, encoding="utf-8")

    assert [m.content for m in memory.search("питон")] == ["люблю питон"]


def test_agent_validates_session_id_at_construction(tmp_path: Path) -> None:
    """Непригодный session_id рвётся сразу при создании агента, а не при первом run()."""
    memory = FileMemory(tmp_path)

    with pytest.raises(InvalidSessionIdError):
        Agent(MockProvider(), memory=memory, session_id="ручная")


def test_invalid_session_id_error_is_value_error() -> None:
    """Обратная совместимость: прежний ``except ValueError`` продолжает ловить."""
    with pytest.raises(ValueError):
        validate_session_id("ручная")
    assert validate_session_id("user-42") == "user-42"
