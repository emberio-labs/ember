"""Персистентная память агента: интерфейс ``Memory`` и реализации.

Публичный API пакета — ``Memory`` (абстрактный интерфейс хранилища диалогов)
и ``FileMemory``, файловая реализация по умолчанию. Пользователь может
подставить свою реализацию (Redis, Postgres, SQLite...) — агенту важен только
интерфейс ``Memory``.

Контракт хранилища: оно работает с «разговорными» сообщениями
(user/assistant/tool) **без system** — системный промпт считается конфигурацией
агента: при загрузке агент сам ставит свой system первым, иначе при
пересоздании с тем же промптом в истории накопились бы дубликаты
system-сообщений.

``session_id`` — переносимый идентификатор (см. ``validate_session_id``).
``FileMemory`` использует его как имя файла сессии, поэтому непригодный id —
это ``InvalidSessionIdError``, а не молчаливая подстановка символов.
"""

from ember.memory.base import (
    InvalidSessionIdError,
    Memory,
    is_valid_session_id,
    validate_session_id,
)
from ember.memory.file import FileMemory

__all__ = [
    "FileMemory",
    "InvalidSessionIdError",
    "Memory",
    "is_valid_session_id",
    "validate_session_id",
]
