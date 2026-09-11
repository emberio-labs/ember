"""Тесты скиллов: разбор SKILL.md, скан каталогов, SkillStore и интеграция с Agent."""

from __future__ import annotations

import json
import warnings
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TypeVar

import pytest

from ember.agent import Agent
from ember.providers.base import Provider
from ember.providers.mock import MockProvider
from ember.skills.base import validate_description, validate_skill_name
from ember.skills.loader import _reset_warnings, parse_skill, scan_skills
from ember.skills.tools import SkillStore
from ember.types import ChatRequest, ChatResponse, FunctionTool, Message, StreamChunk, ToolCall

T = TypeVar("T")


@pytest.fixture(autouse=True)
def _clear_skill_warnings() -> Iterator[None]:
    """Предупреждения дедуплицируются в модуле — между тестами сбрасываем."""
    _reset_warnings()
    yield
    _reset_warnings()


def _with_warnings(func: Callable[[], T]) -> tuple[T, list[str]]:
    """Выполнить ``func`` и вернуть результат вместе с текстами предупреждений."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = func()
    return result, [str(item.message) for item in caught]


def _write_skill(
    root: Path,
    name: str,
    *,
    description: str = "Описание скилла",
    body: str = "Тело скилла",
    extra_frontmatter: str = "",
) -> Path:
    skill_dir = root / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text(
        f"---\nname: {name}\ndescription: {description}\n{extra_frontmatter}---\n\n{body}\n",
        encoding="utf-8",
    )
    return skill_md


def _tool(store: SkillStore, name: str) -> FunctionTool:
    return next(tool for tool in store.tools() if tool.name == name)


class ScriptedProvider(Provider):
    """Провайдер-сценарий: отдаёт заранее заданные ответы по порядку."""

    def __init__(self, script: list[Message]) -> None:
        self.script = script
        self.index = 0
        self.requests: list[ChatRequest] = []

    def complete(self, request: ChatRequest) -> ChatResponse:
        self.requests.append(request)
        if self.index >= len(self.script):
            raise AssertionError("ScriptedProvider: сценарий ответов исчерпан")
        message = self.script[self.index]
        self.index += 1
        return ChatResponse(message=message, model="scripted-1")

    def stream(self, request: ChatRequest) -> Iterator[StreamChunk]:
        raise AssertionError("stream не используется в тестах скиллов")


def _assistant_with_calls(*calls: ToolCall) -> Message:
    return Message(role="assistant", content="", tool_calls=list(calls))


# --- Парсинг SKILL.md -------------------------------------------------------


def test_parse_valid_skill_with_optional_fields(tmp_path: Path) -> None:
    skill_md = _write_skill(
        tmp_path,
        "code-review",
        description="Провести code review изменений",
        body="## Шаги\n1. Прочитать diff",
        extra_frontmatter="license: MIT\nmetadata:\n  owner: platform\n",
    )

    skill = parse_skill(skill_md)

    assert skill is not None
    assert skill.name == "code-review"
    assert skill.description == "Провести code review изменений"
    assert skill.body == "## Шаги\n1. Прочитать diff"
    assert skill.path == skill_md
    assert skill.license == "MIT"
    assert skill.metadata == {"owner": "platform"}
    assert skill.allowed_tools is None


def test_parse_allowed_tools_from_string_and_list(tmp_path: Path) -> None:
    as_string = _write_skill(tmp_path, "one", extra_frontmatter="allowed-tools: Read, Grep\n")
    as_list = _write_skill(
        tmp_path,
        "two",
        extra_frontmatter="allowed-tools:\n  - Read\n  - Grep\n",
    )

    first = parse_skill(as_string)
    second = parse_skill(as_list)

    assert first is not None and first.allowed_tools == ["Read", "Grep"]
    assert second is not None and second.allowed_tools == ["Read", "Grep"]


def test_parse_without_frontmatter_returns_none(tmp_path: Path) -> None:
    skill_dir = tmp_path / "plain"
    skill_dir.mkdir()
    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text("# Просто markdown без frontmatter\n", encoding="utf-8")

    skill, messages = _with_warnings(lambda: parse_skill(skill_md))

    assert skill is None
    assert any("name и description" in message for message in messages)


def test_parse_missing_description_returns_none(tmp_path: Path) -> None:
    skill_dir = tmp_path / "half"
    skill_dir.mkdir()
    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text("---\nname: half\n---\n\nТело\n", encoding="utf-8")

    skill, messages = _with_warnings(lambda: parse_skill(skill_md))

    assert skill is None
    assert messages


def test_parse_broken_yaml_falls_back_to_crude_parse(tmp_path: Path) -> None:
    skill_dir = tmp_path / "tricky"
    skill_dir.mkdir()
    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text(
        "---\nname: tricky\ndescription: [unclosed\n---\n\nТело\n",
        encoding="utf-8",
    )

    skill, messages = _with_warnings(lambda: parse_skill(skill_md))

    assert skill is not None
    assert skill.name == "tricky"
    assert skill.description == "[unclosed"
    assert any("YAML" in message for message in messages)


def test_parse_name_mismatch_warns_but_loads(tmp_path: Path) -> None:
    skill_dir = tmp_path / "folder-name"
    skill_dir.mkdir()
    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text(
        "---\nname: other-name\ndescription: Описание\n---\n\nТело\n",
        encoding="utf-8",
    )

    skill, messages = _with_warnings(lambda: parse_skill(skill_md))

    assert skill is not None
    assert skill.name == "other-name"
    assert any("не совпадает" in message for message in messages)


def test_parse_unreadable_file_returns_none(tmp_path: Path) -> None:
    skill, messages = _with_warnings(lambda: parse_skill(tmp_path / "nope" / "SKILL.md"))

    assert skill is None
    assert any("Не удалось прочитать" in message for message in messages)


# --- Сканирование каталогов -------------------------------------------------


def test_scan_priority_first_dir_wins_on_collision(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_skill(first, "shared", description="из первого")
    _write_skill(second, "shared", description="из второго")

    skills, messages = _with_warnings(lambda: scan_skills([first, second]))

    assert [skill.name for skill in skills] == ["shared"]
    assert skills[0].description == "из первого"
    assert any("пропущен" in message for message in messages)


def test_scan_missing_dir_warns_but_continues(tmp_path: Path) -> None:
    existing = tmp_path / "existing"
    _write_skill(existing, "known")

    skills, messages = _with_warnings(lambda: scan_skills([tmp_path / "nope", existing]))

    assert [skill.name for skill in skills] == ["known"]
    assert any("не найден" in message for message in messages)


def test_scan_ignores_files_and_dirs_without_skill_md(tmp_path: Path) -> None:
    _write_skill(tmp_path, "real")
    (tmp_path / "just-file.txt").write_text("не скилл", encoding="utf-8")
    (tmp_path / "empty-dir").mkdir()
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "SKILL.md").write_text("---\nname: broken\n---\n", encoding="utf-8")

    skills, _ = _with_warnings(lambda: scan_skills([tmp_path]))

    assert [skill.name for skill in skills] == ["real"]


# --- SkillStore: каталог и инструменты --------------------------------------


def test_catalog_is_none_without_skills(tmp_path: Path) -> None:
    assert SkillStore([tmp_path]).catalog() is None


def test_catalog_lists_name_and_description(tmp_path: Path) -> None:
    _write_skill(tmp_path, "code-review", description="Ревью изменений")

    catalog = SkillStore([tmp_path]).catalog()

    assert catalog is not None
    assert "code-review" in catalog
    assert "Ревью изменений" in catalog
    assert "read_skill" in catalog


def test_read_skill_returns_body_and_unknown_raises(tmp_path: Path) -> None:
    _write_skill(tmp_path, "code-review", body="## Шаги")
    read_skill = _tool(SkillStore([tmp_path]), "read_skill")

    assert read_skill.func(name="code-review") == "## Шаги"
    with pytest.raises(ValueError, match="не найден"):
        read_skill.func(name="unknown")


def test_save_creates_skill_with_optional_fields(tmp_path: Path) -> None:
    store = SkillStore([tmp_path])

    message = store.save(
        "code-review",
        "Провести code review изменений",
        "## Шаги\n1. Прочитать diff",
        license="MIT",
        metadata={"owner": "platform"},
        allowed_tools=["read_file", "search_in_project"],
    )

    assert "создан" in message
    skill = parse_skill(tmp_path / "code-review" / "SKILL.md")
    assert skill is not None
    assert skill.name == "code-review"
    assert skill.license == "MIT"
    assert skill.metadata == {"owner": "platform"}
    assert skill.allowed_tools == ["read_file", "search_in_project"]
    assert skill.body == "## Шаги\n1. Прочитать diff"


def test_save_updates_existing_skill(tmp_path: Path) -> None:
    store = SkillStore([tmp_path])
    store.save("code-review", "Первое описание", "Первое тело")

    message = store.save("code-review", "Второе описание", "Второе тело")

    assert "обновлён" in message
    skill = parse_skill(tmp_path / "code-review" / "SKILL.md")
    assert skill is not None and skill.description == "Второе описание"
    assert [path.name for path in (tmp_path / "code-review").iterdir()] == ["SKILL.md"]


def test_save_writes_only_first_dir_and_reports_shadowing(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    shadowed_md = _write_skill(second, "shared", description="старый")

    message = SkillStore([first, second]).save("shared", "новый", "тело")

    assert "перекрывает" in message
    assert (first / "shared" / "SKILL.md").is_file()
    assert "старый" in shadowed_md.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "bad_name",
    ["", "  ", "Foo", "a--b", "-a", "a-", "a b", "код", "x" * 65, "../evil", "a/b"],
)
def test_save_rejects_invalid_name(tmp_path: Path, bad_name: str) -> None:
    with pytest.raises(ValueError):
        SkillStore([tmp_path]).save(bad_name, "описание", "тело")

    assert list(tmp_path.iterdir()) == []


def test_save_rejects_long_description_and_empty_body(tmp_path: Path) -> None:
    store = SkillStore([tmp_path])

    with pytest.raises(ValueError, match="1024"):
        store.save("code-review", "x" * 1025, "тело")
    with pytest.raises(ValueError, match="body"):
        store.save("code-review", "описание", "   ")


def test_validate_skill_name_accepts_valid() -> None:
    assert validate_skill_name("  code-review-2 ") == "code-review-2"


def test_validate_description_rejects_blank() -> None:
    with pytest.raises(ValueError, match="description"):
        validate_description("   ")


# --- Интеграция с Agent -----------------------------------------------------


def test_agent_adds_skill_catalog_to_request(tmp_path: Path) -> None:
    _write_skill(tmp_path, "code-review", description="Ревью изменений")
    provider = ScriptedProvider([Message(role="assistant", content="ок")])
    agent = Agent(provider, system_prompt="Ты помощник", skills_dirs=[tmp_path])

    agent.run("Привет")

    messages = provider.requests[0].messages
    assert messages[0].content == "Ты помощник"
    assert messages[1].role == "system"
    assert "code-review" in messages[1].content
    assert [tool.name for tool in provider.requests[0].tools or []] == [
        "read_skill",
        "save_skill",
    ]
    # Каталог не оседает в истории агента — только в сообщениях запроса.
    assert [m.role for m in agent.messages] == ["system", "user", "assistant"]


def test_agent_without_skills_has_no_tools_and_no_catalog() -> None:
    provider = ScriptedProvider([Message(role="assistant", content="ок")])

    Agent(provider).run("Привет")

    assert provider.requests[0].tools is None
    assert [m.role for m in provider.requests[0].messages] == ["user"]


def test_agent_with_skills_works_on_mock_provider(tmp_path: Path) -> None:
    _write_skill(tmp_path, "code-review", description="Ревью изменений")
    agent = Agent(provider=MockProvider(response_text="ок"), skills_dirs=[tmp_path])

    assert agent.run("Привет") == "ок"
    assert [tool.name for tool in agent.tools or []] == ["read_skill", "save_skill"]


def test_agent_executes_read_skill_tool_call(tmp_path: Path) -> None:
    _write_skill(tmp_path, "code-review", body="## Шаги")
    provider = ScriptedProvider(
        [
            _assistant_with_calls(
                ToolCall(id="c1", name="read_skill", arguments='{"name": "code-review"}')
            ),
            Message(role="assistant", content="Загрузил скилл"),
        ]
    )
    agent = Agent(provider, skills_dirs=[tmp_path])

    assert agent.run("Сделай ревью") == "Загрузил скилл"

    tool_messages = [m for m in agent.messages if m.role == "tool"]
    assert tool_messages[0].content == "## Шаги"


def test_agent_saves_skill_and_catalog_updates_next_run(tmp_path: Path) -> None:
    arguments = json.dumps(
        {
            "name": "reliable-http",
            "description": "Ретраи httpx: backoff, retry на 429/5xx",
            "body": "## Шаги\n1. transport с retries",
        }
    )
    provider = ScriptedProvider(
        [
            _assistant_with_calls(ToolCall(id="c1", name="save_skill", arguments=arguments)),
            Message(role="assistant", content="Сохранил"),
            Message(role="assistant", content="Второй ответ"),
        ]
    )
    agent = Agent(provider, skills_dirs=[tmp_path])

    assert agent.run("Запомни процедуру") == "Сохранил"
    assert (tmp_path / "reliable-http" / "SKILL.md").is_file()

    first_request = provider.requests[0]
    assert not any("reliable-http" in m.content for m in first_request.messages)

    agent.run("Что дальше?")

    second_request = provider.requests[1]
    assert any("reliable-http" in m.content for m in second_request.messages)


def test_agent_rejects_collision_with_user_skill_tool(tmp_path: Path) -> None:
    provider = ScriptedProvider([Message(role="assistant", content="ок")])
    user_tool = FunctionTool(
        name="save_skill", description="свой", parameters={"type": "object"}, func=lambda: None
    )

    with pytest.raises(ValueError, match="Дубликат"):
        Agent(provider, tools=[user_tool], skills_dirs=[tmp_path])


def test_user_save_skill_tool_allowed_without_skills() -> None:
    provider = ScriptedProvider([Message(role="assistant", content="ок")])
    user_tool = FunctionTool(
        name="save_skill", description="свой", parameters={"type": "object"}, func=lambda: None
    )

    assert Agent(provider, tools=[user_tool]).tools == [user_tool]


def test_stream_run_with_skills_executes_read_skill(tmp_path: Path) -> None:
    """Инструменты скиллов (#16) работают и в потоковом режиме."""
    _write_skill(tmp_path, "code-review", body="## Шаги")

    class StreamingProvider(Provider):
        rounds = [
            [
                StreamChunk(
                    delta="",
                    model="s",
                    finish_reason="tool_calls",
                    tool_calls=[
                        ToolCall(id="c1", name="read_skill", arguments='{"name": "code-review"}')
                    ],
                )
            ],
            [StreamChunk(delta="Готово", model="s", finish_reason="stop")],
        ]

        def __init__(self) -> None:
            self.requests: list[ChatRequest] = []

        def complete(self, request: ChatRequest) -> ChatResponse:
            raise AssertionError("complete не используется в потоковом тесте")

        def stream(self, request: ChatRequest) -> Iterator[StreamChunk]:
            self.requests.append(request)
            yield from self.rounds[len(self.requests) - 1]

    provider = StreamingProvider()
    agent = Agent(provider, skills_dirs=[tmp_path])

    assert "".join(agent.stream_run("Сделай ревью")) == "Готово"

    tool_messages = [m for m in agent.messages if m.role == "tool"]
    assert tool_messages[0].content == "## Шаги"
    assert [tool.name for tool in provider.requests[0].tools or []] == [
        "read_skill",
        "save_skill",
    ]
