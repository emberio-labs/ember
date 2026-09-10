"""Загрузка скиллов с диска: разбор ``SKILL.md`` и сканирование каталогов.

Чтение намеренно **лояльное** (lenient): чужие скиллы могут быть написаны
другим клиентом или руками, и кривой frontmatter не должен ронять агента.
Строгие проверки применяются только при записи (см. ``ember.skills.tools``).

Прогрессивное раскрытие (progressive disclosure) стандарта: каталог отдаёт
модели только ``name`` + ``description``, а тело ``SKILL.md`` подгружается
отдельным инструментом — см. ``SkillStore.catalog``.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml

from ember.skills.base import Skill

_FRONTMATTER_DELIMITER = "---"

#: Уже выданные предупреждения: повторный скан (каждый запрос к модели)
#: не должен заново спамить одинаковыми сообщениями.
_WARNED: set[tuple[str, str]] = set()


def _reset_warnings() -> None:
    """Забыть выданные предупреждения (нужно тестам)."""
    _WARNED.clear()


def _warn(event: str, key: str, message: str) -> None:
    cache_key = (event, key)
    if cache_key in _WARNED:
        return
    _WARNED.add(cache_key)
    warnings.warn(message, stacklevel=3)


def split_frontmatter(text: str) -> tuple[str, str]:
    """Отделить YAML-frontmatter от markdown-тела ``SKILL.md``.

    Frontmatter — блок между первой строкой ``---`` и следующей строкой
    ``---``. Если файл не начинается с разделителя (или блок не закрыт),
    frontmatter пустой, а телом считается весь текст: разбирать такое
    дальше — задача ``parse_skill``.

    Args:
        text: Полное содержимое ``SKILL.md``.

    Returns:
        Пару ``(frontmatter, body)`` без окружающих пробелов у тела.
    """
    normalized = text.lstrip("﻿")
    if not normalized.startswith(_FRONTMATTER_DELIMITER):
        return "", normalized.strip()
    lines = normalized.splitlines(keepends=True)
    for index in range(1, len(lines)):
        if lines[index].strip() == _FRONTMATTER_DELIMITER:
            return "".join(lines[1:index]), "".join(lines[index + 1 :]).strip()
    return normalized, ""


def _crude_frontmatter(frontmatter: str) -> dict[str, str]:
    """Грубый разбор frontmatter построчно, когда YAML не распарсился.

    Понимает только простые скалярные пары ``key: value``: этого достаточно,
    чтобы вытащить ``name``/``description`` из чуть подпорченного файла.
    """
    data: dict[str, str] = {}
    for line in frontmatter.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        key, separator, value = stripped.partition(":")
        if not separator:
            continue
        data[key.strip()] = value.strip().strip("'\"")
    return data


def _load_yaml(frontmatter: str) -> tuple[dict[str, Any], bool]:
    """Разобрать frontmatter, при ошибке YAML — грубый построчный фолбэк.

    Returns:
        Пару ``(данные, использован_фолбэк)``.
    """
    if not frontmatter.strip():
        return {}, False
    try:
        data: Any = yaml.safe_load(frontmatter)
    except yaml.YAMLError:
        return _crude_frontmatter(frontmatter), True
    if isinstance(data, dict):
        return data, False
    # Пустой frontmatter (None) — норма; список или скаляр — уже аномалия.
    return {}, data is not None


def _optional_str(value: Any) -> str | None:
    if isinstance(value, str):
        text = value.strip()
        return text or None
    return None


def _optional_metadata(value: Any) -> dict[str, str] | None:
    if isinstance(value, dict):
        items = {str(key): str(item) for key, item in value.items()}
        return items or None
    return None


def _optional_allowed_tools(value: Any) -> list[str] | None:
    """Нормализовать ``allowed-tools`` — строку (через запятую/пробел) или список."""
    parts: list[str]
    if isinstance(value, str):
        parts = [part for chunk in value.split(",") for part in chunk.split()]
    elif isinstance(value, list):
        parts = [str(item).strip() for item in value]
    else:
        return None
    cleaned = [part for part in parts if part]
    return cleaned or None


def parse_skill(skill_md: str | Path) -> Skill | None:
    """Прочитать ``SKILL.md`` и собрать ``Skill`` (лояльно, без падений).

    Лояльность: битый YAML разбирается грубо, расхождение ``name`` с именем
    каталога даёт предупреждение, но скилл грузится. Файл без ``name`` или
    ``description`` пропускается с предупреждением — такой скилл бесполезен
    для каталога (модель не поймёт, когда его применять).

    Args:
        skill_md: Путь к файлу ``SKILL.md``.

    Returns:
        Разобранный ``Skill`` или ``None``, если файл нечитаем либо в нём нет
        обязательных полей.
    """
    path = Path(skill_md)
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        _warn("unreadable", str(path), f"Не удалось прочитать скилл {path}: {exc}")
        return None
    frontmatter, body = split_frontmatter(raw)
    data, used_fallback = _load_yaml(frontmatter)
    if used_fallback:
        _warn(
            "broken-yaml",
            str(path),
            f"Некорректный YAML в {path}: разбираю frontmatter грубо, построчно",
        )
    name = _optional_str(data.get("name"))
    description = _optional_str(data.get("description"))
    if name is None or description is None:
        _warn(
            "missing-fields",
            str(path),
            f"Скилл {path} пропущен: нужны непустые поля name и description",
        )
        return None
    if name != path.parent.name:
        _warn(
            "name-mismatch",
            str(path),
            f"У скилла {path} name={name!r} не совпадает с именем каталога {path.parent.name!r}",
        )
    return Skill(
        name=name,
        description=description,
        body=body,
        path=path,
        license=_optional_str(data.get("license")),
        compatibility=_optional_str(data.get("compatibility")),
        metadata=_optional_metadata(data.get("metadata")),
        allowed_tools=_optional_allowed_tools(data.get("allowed-tools")),
    )


def scan_skills(directories: Iterable[str | Path]) -> list[Skill]:
    """Собрать скиллы из каталогов ``<каталог>/<name>/SKILL.md``.

    Каталоги перебираются по порядку, и **порядок задаёт приоритет**: при
    коллизии имён побеждает скилл из более раннего каталога, остальные
    пропускаются с предупреждением. Вглубь более одного уровня не смотрим.
    Несуществующий каталог — предупреждение, а не ошибка.

    Args:
        directories: Каталоги скиллов (пути в порядке приоритета).

    Returns:
        Скиллы в порядке обнаружения (без дублей по имени).
    """
    skills: list[Skill] = []
    winners: dict[str, Path] = {}
    for directory in directories:
        root = Path(directory)
        if not root.is_dir():
            _warn("missing-dir", str(root), f"Каталог скиллов не найден: {root}")
            continue
        for entry in sorted(root.iterdir()):
            skill_md = entry / "SKILL.md"
            if not entry.is_dir() or not skill_md.is_file():
                continue
            skill = parse_skill(skill_md)
            if skill is None:
                continue
            if skill.name in winners:
                _warn(
                    "collision",
                    skill.name,
                    f"Скилл {skill.name!r} из {skill_md} пропущен: "
                    f"приоритет у {winners[skill.name]}",
                )
                continue
            winners[skill.name] = skill.path
            skills.append(skill)
    return skills
