"""Встроенные инструменты скиллов: чтение тела и запись по явной просьбе.

``SkillStore`` связывает каталоги скиллов и агентом: отдаёт tier-1 каталог
(``name`` + ``description``) для системного промпта и два инструмента —
``read_skill`` (подгрузить тело) и ``save_skill`` (создать/обновить скилл).

Запись строгая (по стандарту Agent Skills) и только в **первый** каталог
списка; чтение — лояльное (см. ``ember.skills.loader``). Инструмент
``save_skill`` не создаётся, если скиллы у агента выключены, — модель
физически не может ничего записать.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

from ember.skills.base import Skill, validate_description, validate_skill_name
from ember.skills.loader import scan_skills
from ember.types import FunctionTool


def _require_str(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"Поле {field} должно быть строкой, получено: {type(value).__name__}")
    if not value.strip():
        raise ValueError(f"Поле {field} не может быть пустым")
    return value


def _clean_optional_str(value: Any, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"Поле {field} должно быть строкой, получено: {type(value).__name__}")
    text = value.strip()
    if not text:
        raise ValueError(f"Поле {field} не может быть пустым")
    return text


def _clean_metadata(value: Any) -> dict[str, str] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"Поле metadata должно быть объектом, получено: {type(value).__name__}")
    items = {str(key): str(item) for key, item in value.items()}
    return items or None


def _clean_allowed_tools(value: Any) -> list[str] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError(
            f"Поле allowed_tools должно быть списком, получено: {type(value).__name__}"
        )
    cleaned = [str(item).strip() for item in value]
    tools = [item for item in cleaned if item]
    return tools or None


class SkillStore:
    """Каталог скиллов агента: скан, tier-1 каталог и встроенные инструменты.

    Args:
        directories: Каталоги скиллов в порядке приоритета. Первый —
            цель записи ``save_skill`` и победитель при коллизии имён.
    """

    def __init__(self, directories: Sequence[str | Path]) -> None:
        if not directories:
            raise ValueError("Нужен хотя бы один каталог скиллов")
        self.directories: list[Path] = [Path(directory) for directory in directories]

    def scan(self) -> list[Skill]:
        """Свежий скан каталогов: скилл, сохранённый в этом же диалоге, уже виден."""
        return scan_skills(self.directories)

    def catalog(self) -> str | None:
        """Tier-1 каталог скиллов для системного промпта: ``name`` + ``description``.

        Returns:
            Текст секции или ``None``, если валидных скиллов нет.
        """
        skills = self.scan()
        if not skills:
            return None
        lines = [
            "## Available skills",
            "Call read_skill(name) to load a skill's full instructions when its "
            "description matches the task.",
            "",
            *[f"- {skill.name}: {skill.description}" for skill in skills],
        ]
        return "\n".join(lines)

    def tools(self) -> list[FunctionTool]:
        """Встроенные инструменты скиллов: ``read_skill`` и ``save_skill``."""
        return [self._read_tool(), self._save_tool()]

    def _read_tool(self) -> FunctionTool:
        def read_skill(name: str) -> str:
            skills = {skill.name: skill for skill in self.scan()}
            skill = skills.get(name)
            if skill is None:
                available = ", ".join(sorted(skills)) or "нет"
                raise ValueError(f"Скилл {name!r} не найден. Доступные скиллы: {available}.")
            return skill.body

        return FunctionTool(
            name="read_skill",
            description=(
                "Load the full instructions (body) of a skill by name. Call it when a "
                "skill's description matches the task, before acting on it."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Skill name exactly as listed in the available skills.",
                    }
                },
                "required": ["name"],
            },
            func=read_skill,
        )

    def _save_tool(self) -> FunctionTool:
        def save_skill(
            name: str,
            description: str,
            body: str,
            license: str | None = None,
            compatibility: str | None = None,
            metadata: dict[str, str] | None = None,
            allowed_tools: list[str] | None = None,
        ) -> str:
            return self.save(
                name,
                description,
                body,
                license=license,
                compatibility=compatibility,
                metadata=metadata,
                allowed_tools=allowed_tools,
            )

        return FunctionTool(
            name="save_skill",
            description=(
                "Create or update a skill: a reusable instruction set the agent can load "
                "later. Call it ONLY when the user explicitly asks to remember or save a "
                "procedure, and write name, description and body in the same language the "
                "user is using."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": (
                            "Skill name: lowercase latin letters, digits and single "
                            "hyphens, at most 64 characters."
                        ),
                    },
                    "description": {
                        "type": "string",
                        "description": (
                            "What the skill does and when to apply it, at most 1024 "
                            "characters. Shown in the skills catalog."
                        ),
                    },
                    "body": {
                        "type": "string",
                        "description": "Full markdown instructions the agent will follow.",
                    },
                    "license": {
                        "type": "string",
                        "description": "Optional license name or note.",
                    },
                    "compatibility": {
                        "type": "string",
                        "description": "Optional compatibility notes.",
                    },
                    "metadata": {
                        "type": "object",
                        "additionalProperties": {"type": "string"},
                        "description": "Optional flat string-to-string metadata.",
                    },
                    "allowed_tools": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of tool names the skill may use (stored).",
                    },
                },
                "required": ["name", "description", "body"],
            },
            func=save_skill,
        )

    def save(
        self,
        name: str,
        description: str,
        body: str,
        license: str | None = None,
        compatibility: str | None = None,
        metadata: dict[str, str] | None = None,
        allowed_tools: list[str] | None = None,
    ) -> str:
        """Создать или обновить скилл в первом каталоге списка.

        Строгая валидация по стандарту: ``name`` (charset/длина/без ``--``),
        ``description`` (непустое, ≤1024), непустое тело. Опциональные поля
        спеки пишутся в frontmatter, если заданы, — так задаётся полный
        контракт скилла (enforcement ``allowed-tools`` — issue #34).

        Args:
            name: Имя скилла (совпадает с именем каталога).
            description: Что делает скилл и когда применять.
            body: Markdown-инструкции скилла.
            license: Опциональная лицензия.
            compatibility: Опциональные заметки о совместимости.
            metadata: Опциональные плоские метаданные.
            allowed_tools: Опциональный список разрешённых инструментов.

        Returns:
            Человекочитаемый результат для модели: создан или обновлён,
            и не перекрывает ли он одноимённый скилл из каталога ниже.

        Raises:
            ValueError: Если данные не проходят валидацию либо путь скилла
                выходит за пределы целевого каталога.
        """
        skill_name = validate_skill_name(_require_str(name, "name"))
        skill_description = validate_description(_require_str(description, "description"))
        skill_body = _require_str(body, "body").strip()

        frontmatter: dict[str, Any] = {
            "name": skill_name,
            "description": skill_description,
        }
        optional_fields: list[tuple[str, Any]] = [
            ("license", _clean_optional_str(license, "license")),
            ("compatibility", _clean_optional_str(compatibility, "compatibility")),
            ("metadata", _clean_metadata(metadata)),
            ("allowed-tools", _clean_allowed_tools(allowed_tools)),
        ]
        for key, value in optional_fields:
            if value is not None:
                frontmatter[key] = value

        root = self.directories[0]
        skill_dir = root / skill_name
        resolved_root = root.resolve()
        resolved_dir = skill_dir.resolve()
        if resolved_root not in resolved_dir.parents:
            raise ValueError(f"Путь скилла выходит за пределы каталога {root}: {skill_dir}")
        skill_dir.mkdir(parents=True, exist_ok=True)
        skill_md = skill_dir / "SKILL.md"
        existed_here = skill_md.exists()

        dumped = yaml.safe_dump(frontmatter, allow_unicode=True, sort_keys=False)
        content = f"---\n{dumped}---\n\n{skill_body}\n"
        tmp_path = skill_md.with_name(skill_md.name + ".tmp")
        tmp_path.write_text(content, encoding="utf-8")
        os.replace(tmp_path, skill_md)

        if existed_here:
            return f"Скилл {skill_name!r} обновлён: {skill_md}"
        shadowed = self._shadowed_path(skill_name)
        if shadowed is not None:
            return (
                f"Скилл {skill_name!r} создан: {skill_md}. "
                f"Он перекрывает одноимённый скилл из {shadowed}."
            )
        return f"Скилл {skill_name!r} создан: {skill_md}"

    def _shadowed_path(self, name: str) -> Path | None:
        """Путь одноимённого скилла из каталогов ниже приоритетом (если есть)."""
        for directory in self.directories[1:]:
            candidate = directory / name / "SKILL.md"
            if candidate.is_file():
                return candidate
        return None
