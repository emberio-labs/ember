"""Скиллы агента по стандарту Agent Skills: модель данных и правила стандарта.

Скилл — каталог ``<name>/SKILL.md`` с YAML-frontmatter и markdown-телом.
Здесь живут разобранный ``Skill``, константы стандарта и строгие проверки,
применяемые при **записи** (чтение чужих скиллов — лояльное, см.
``ember.skills.loader``). Загрузка — в ``loader``, инструменты — в ``tools``.

Стандарт: agentskills.io (репозиторий ``anthropics/skills``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

#: Максимальная длина имени скилла по стандарту.
MAX_NAME_LENGTH = 64
#: Максимальная длина описания скилла по стандарту.
MAX_DESCRIPTION_LENGTH = 1024
#: Имя скилла: строчные латинские буквы и цифры, одиночные дефисы внутри.
SKILL_NAME_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


@dataclass(slots=True)
class Skill:
    """Разобранный скилл: метаданные frontmatter + тело ``SKILL.md``.

    Attributes:
        name: Имя скилла из frontmatter (по стандарту совпадает с именем
            каталога, в котором лежит ``SKILL.md``).
        description: Краткое описание: что делает скилл и когда его применять.
        body: Markdown-тело ``SKILL.md`` — инструкции, которые модель
            подгружает отдельным инструментом, когда описание релевантно.
        path: Путь к исходному ``SKILL.md``.
        license: Опциональная лицензия скилла.
        compatibility: Опциональная информация о совместимости.
        metadata: Опциональные произвольные метаданные (ключ → значение).
        allowed_tools: Опциональный список инструментов, разрешённых скиллу.
            В v1 поле только читается и хранится (enforcement — см. issue #34).
    """

    name: str
    description: str
    body: str
    path: Path
    license: str | None = None
    compatibility: str | None = None
    metadata: dict[str, str] | None = None
    allowed_tools: list[str] | None = None


def validate_skill_name(name: str) -> str:
    """Проверить имя скилла по стандарту и вернуть его без пробелов по краям.

    Raises:
        ValueError: Если имя пустое, длиннее ``MAX_NAME_LENGTH`` или не
            соответствует ``SKILL_NAME_PATTERN`` (строчные буквы/цифры,
            одиночные дефисы внутри — значит, без ``--`` и дефисов по краям).
    """
    candidate = name.strip()
    if not candidate:
        raise ValueError("Имя скилла (name) не может быть пустым")
    if len(candidate) > MAX_NAME_LENGTH:
        raise ValueError(f"Имя скилла длиннее {MAX_NAME_LENGTH} символов: {len(candidate)}")
    if not SKILL_NAME_PATTERN.match(candidate):
        raise ValueError(
            "Имя скилла должно состоять из строчных латинских букв и цифр, "
            f"разделённых одиночными дефисами: {candidate!r}"
        )
    return candidate


def validate_description(description: str) -> str:
    """Проверить описание скилла по стандарту и вернуть его без пробелов по краям.

    Raises:
        ValueError: Если описание пустое или длиннее ``MAX_DESCRIPTION_LENGTH``.
    """
    text = description.strip()
    if not text:
        raise ValueError("Описание скилла (description) не может быть пустым")
    if len(text) > MAX_DESCRIPTION_LENGTH:
        raise ValueError(
            f"Описание скилла длиннее {MAX_DESCRIPTION_LENGTH} символов: {len(text)}"
        )
    return text
