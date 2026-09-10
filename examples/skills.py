"""Скиллы агента (Agent Skills) — исполняемый пример без API-ключей.

Демонстрирует полный цикл работы со скиллами:

1. по явной просьбе пользователя агент сохраняет процедуру инструментом
   ``save_skill`` — в первый каталог из ``skills_dirs``;
2. в следующем запросе в промпт попадает tier-1 каталог (``name`` +
   ``description``), а тело скилла модель ещё не видит;
3. когда описание подходит к задаче, модель подгружает тело инструментом
   ``read_skill`` и действует по инструкции.

Скиллы пишутся в реальные файлы ``<skills_dir>/<name>/SKILL.md``, поэтому
пример создаёт временный каталог и печатает его содержимое. Модель —
сценарный мок: он печатает каталог из запроса, чтобы было видно, что именно
получает модель.
"""

from __future__ import annotations

import tempfile
from collections.abc import Iterator
from pathlib import Path

from ember import Agent
from ember.providers.base import Provider
from ember.skills import SkillStore
from ember.types import ChatRequest, ChatResponse, Message, StreamChunk, ToolCall

CATALOG_HEADER = "## Available skills"

SAVE_CALL = ToolCall(
    id="call_1",
    name="save_skill",
    arguments=(
        '{"name": "code-review", '
        '"description": "Провести code review изменений: риски, стиль, тесты", '
        '"body": "## Шаги\\n1. Прочитать diff\\n'
        '2. Отметить риски и отсутствующие тесты"}'
    ),
)
READ_CALL = ToolCall(
    id="call_2",
    name="read_skill",
    arguments='{"name": "code-review"}',
)


class SkillScenarioMock(Provider):
    """Сценарный мок: играет роли «сохранил скилл» → «загрузил скилл».

    Имитирует модель: сначала сохраняет процедуру по просьбе пользователя,
    затем подгружает её тело, когда описание подходит к задаче.
    """

    def __init__(self) -> None:
        self._step = 0

    def complete(self, request: ChatRequest) -> ChatResponse:
        self._print_catalog(request)
        self._step += 1
        if self._step == 1:
            return self._reply(SAVE_CALL)
        if self._step == 2:
            return self._reply(None, "Запомнил процедуру code review.")
        if self._step == 3:
            return self._reply(READ_CALL)
        return self._reply(None, "Провёл ревью по загруженной процедуре.")

    @staticmethod
    def _reply(call: ToolCall | None, text: str = "") -> ChatResponse:
        return ChatResponse(
            message=Message(role="assistant", content=text, tool_calls=[call] if call else []),
            model="mock-skills",
        )

    @staticmethod
    def _print_catalog(request: ChatRequest) -> None:
        """Печатать каталог один раз за ход диалога: он в каждом запросе цикла."""
        if request.messages[-1].role != "user":
            return
        for message in request.messages:
            if message.role == "system" and message.content.startswith(CATALOG_HEADER):
                print("[каталог, который видит модель]")
                print(message.content)
                print()

    def stream(self, request: ChatRequest) -> Iterator[StreamChunk]:
        raise NotImplementedError("Потоковый режим в примере не используется")


def main() -> None:
    """Прогнать сценарий: сохранение скилла → каталог в промпте → загрузка тела."""
    with tempfile.TemporaryDirectory(prefix="ember-skills-") as tmp:
        skills_dir = Path(tmp) / "skills"
        skills_dir.mkdir()  # каталог записи; вообще-то его создаст и save_skill
        agent = Agent(
            provider=SkillScenarioMock(),
            system_prompt="Ты — ассистент разработчика.",
            skills_dirs=[skills_dir],
        )

        print("Пользователь: Запомни процедуру code review.")
        print("Агент:", agent.run("Запомни процедуру code review."))
        print()

        print("Пользователь: Сделай ревью последних изменений.")
        print("Агент:", agent.run("Сделай ревью последних изменений."))
        print()

        print("Инструменты агента:", [tool.name for tool in agent.tools or []])
        saved = sorted(path.parent.name for path in skills_dir.glob("*/SKILL.md"))
        print("Скиллы на диске:", saved)
        print()
        print("Файл, который записал агент —", skills_dir / "code-review" / "SKILL.md")
        print("Каталог скиллов, который строится для промпта:")
        print(SkillStore([skills_dir]).catalog())


if __name__ == "__main__":
    main()
