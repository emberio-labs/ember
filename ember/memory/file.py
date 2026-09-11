"""Файловая реализация ``Memory``: один JSONL-файл на сессию.

Каждая сессия хранится в ``<directory>/<session_id>.json``; каждая строка
файла — JSON-представление одного сообщения (формат JSON Lines, расширение
``.json``). Запись атомарная: сначала во временный файл, затем
переименование — падение в середине записи не оставит битый файл сессии.

``session_id`` проверяется (``validate_session_id``) и совпадает с именем файла
без расширения. Поиск
(``search``) — bag-of-words overlap по user/assistant-сообщениям всех сессий:
без LLM и эмбеддингов, достаточно для небольших локальных архивов.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ember.memory.base import Memory, is_valid_session_id, validate_session_id
from ember.types import Message, ToolCall

#: Слова короче этой длины — только шум для поиска (предлоги, союзы, артикли).
_MIN_WORD_LENGTH = 3
#: Частотные слова, не несущие смысла для поиска.
_STOP_WORDS: frozenset[str] = frozenset(
    {
        "для",
        "что",
        "как",
        "когда",
        "чтобы",
        "при",
        "или",
        "если",
        "уже",
        "еще",
        "ещё",
        "будет",
        "можно",
        "надо",
        "нужно",
        "очень",
        "просто",
        "какой",
        "какая",
        "какие",
        "который",
        "and",
        "for",
        "are",
        "but",
        "not",
        "you",
        "all",
        "any",
        "can",
        "was",
        "one",
        "our",
        "out",
        "the",
    }
)

_WORD_RE = re.compile(r"[a-zа-яё0-9]+")


def _tokenize(text: str) -> set[str]:
    """Разбить текст на значимые слова: lowercase, без пунктуации и стоп-слов."""
    words = _WORD_RE.findall(text.lower())
    return {word for word in words if len(word) >= _MIN_WORD_LENGTH and word not in _STOP_WORDS}


def _tool_call_to_dict(call: ToolCall) -> dict[str, Any]:
    return {"id": call.id, "name": call.name, "arguments": call.arguments}


def _tool_call_from_dict(data: dict[str, Any]) -> ToolCall:
    return ToolCall(
        id=data["id"],
        name=data["name"],
        arguments=data.get("arguments", "{}"),
    )


def _message_to_dict(message: Message) -> dict[str, Any]:
    """Компактное JSON-представление сообщения: None-поля опускаются."""
    data: dict[str, Any] = {"role": message.role, "content": message.content}
    if message.name is not None:
        data["name"] = message.name
    if message.tool_calls:
        data["tool_calls"] = [_tool_call_to_dict(call) for call in message.tool_calls]
    if message.tool_call_id is not None:
        data["tool_call_id"] = message.tool_call_id
    return data


def _message_from_dict(data: dict[str, Any]) -> Message:
    raw_calls = data.get("tool_calls")
    tool_calls = [_tool_call_from_dict(call) for call in raw_calls] if raw_calls else None
    return Message(
        role=data["role"],
        content=data.get("content", ""),
        name=data.get("name"),
        tool_calls=tool_calls,
        tool_call_id=data.get("tool_call_id"),
    )


class FileMemory(Memory):
    """Файловая реализация ``Memory``: один JSONL-файл на сессию.

    Каждая сессия хранится в ``<directory>/<session_id>.json``; каждая строка
    файла — JSON-представление одного сообщения (формат JSON Lines, расширение
    ``.json``). Запись атомарная: сначала во временный файл, затем
    переименование — падение в середине записи не оставит битый файл сессии.

    ``session_id`` проверяется (``validate_session_id``); непригодный id —
    ``InvalidSessionIdError``.
    """

    def __init__(self, directory: str | Path) -> None:
        """Создать хранилище в ``directory`` (создаётся при необходимости).

        Args:
            directory: Директория с файлами сессий. Путь явный, без «магических»
                дефолтов в рабочей директории.
        """
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path_for(self, session_id: str) -> Path:
        """Путь к файлу сессии: ``session_id`` — это имя файла без расширения."""
        return self.directory / f"{validate_session_id(session_id)}.json"

    def load_session(self, session_id: str) -> list[Message]:
        path = self._path_for(session_id)
        if not path.exists():
            return []
        messages: list[Message] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            messages.append(_message_from_dict(json.loads(line)))
        return messages

    def save_session(self, session_id: str, messages: Sequence[Message]) -> None:
        path = self._path_for(session_id)
        self.directory.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(path.name + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            for message in messages:
                handle.write(json.dumps(_message_to_dict(message), ensure_ascii=False) + "\n")
        os.replace(tmp_path, path)

    def delete_session(self, session_id: str) -> None:
        self._path_for(session_id).unlink(missing_ok=True)

    def search(
        self,
        query: str,
        *,
        exclude_session_id: str | None = None,
        limit: int = 5,
    ) -> list[Message]:
        """Найти сообщения по прошлым сессиям (контракт — ``Memory.search``)."""
        if exclude_session_id is not None:
            validate_session_id(exclude_session_id)
        query_words = _tokenize(query)
        if not query_words or limit <= 0:
            return []
        scored: list[tuple[int, Message]] = []
        for session_id in self._session_ids():
            if session_id == exclude_session_id:
                continue
            for message in self.load_session(session_id):
                if message.role not in ("user", "assistant"):
                    continue
                overlap = len(query_words & _tokenize(message.content))
                if overlap:
                    scored.append((overlap, message))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [message for _, message in scored[:limit]]

    def _session_ids(self) -> list[str]:
        """Id сессий в директории.

        Имя файла без расширения и есть ``session_id``. Посторонние ``*.json``
        пропускаются: адресовать их через публичный API нельзя.
        """
        names = [path.stem for path in self.directory.glob("*.json")]
        return sorted(session_id for session_id in names if is_valid_session_id(session_id))
