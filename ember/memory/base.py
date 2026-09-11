"""Интерфейс хранилища диалогов агента (``Memory``).

Контракт: сессия — это диалог с одним ``session_id`` из «разговорных»
сообщений (user/assistant/tool) без system-сообщений — системный промпт
задаётся агенту отдельно и при загрузке ставится первым. ``search`` ищет
по прошлым сессиям и используется агентом для кросс-сессионного recall.

``session_id`` — переносимый идентификатор: латинские буквы, цифры, «_», «-»
и «.» (точка — не первым и не последним символом), не длиннее
``MAX_SESSION_ID_LENGTH`` символов. Реализации, для которых id становится
именем файла или ключом с ограниченным набором символов, обязаны проверять
его (``validate_session_id``) и бросать ``InvalidSessionIdError``:
«исправлять» id подстановкой символов нельзя — id адресует сессию.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Sequence

from ember.types import Message

#: Предел длины ``session_id``. Файловая реализация добавляет к id расширение
#: и суффикс временного файла — 128 символов с запасом укладываются в лимит
#: файловой системы (255 байт).
MAX_SESSION_ID_LENGTH = 128

#: Допустимый ``session_id``: один разрешённый символ либо последовательность,
#: которая начинается и заканчивается символом, отличным от точки.
_SESSION_ID_RE = re.compile(r"[A-Za-z0-9_-](?:[A-Za-z0-9_.-]*[A-Za-z0-9_-])?")


class InvalidSessionIdError(ValueError):
    """``session_id`` не является допустимым идентификатором сессии.

    Наследник ``ValueError``: код, который уже ловил ``ValueError``
    (например, на пустой id), продолжает работать.
    """


def validate_session_id(session_id: str) -> str:
    """Проверить ``session_id`` и вернуть его без изменений.

    Проверка не «починяет» id: непригодный идентификатор — ошибка вызывающего,
    а не повод молча подставить символы и записать сессию под чужим именем.

    Args:
        session_id: Идентификатор сессии.

    Returns:
        Тот же ``session_id``, если он допустим.

    Raises:
        InvalidSessionIdError: Если id пуст, длиннее ``MAX_SESSION_ID_LENGTH``
            или содержит символы вне ``[A-Za-z0-9_.-]``, а также если он
            начинается или заканчивается точкой.
    """
    if not session_id:
        raise InvalidSessionIdError("session_id не может быть пустым")
    if len(session_id) > MAX_SESSION_ID_LENGTH:
        raise InvalidSessionIdError(
            f"session_id длиной {len(session_id)} символов превышает допустимые "
            f"{MAX_SESSION_ID_LENGTH}"
        )
    if _SESSION_ID_RE.fullmatch(session_id) is None:
        raise InvalidSessionIdError(
            f"session_id {session_id!r} содержит недопустимые символы: разрешены "
            "латинские буквы, цифры, '_', '-' и '.' (точка — не первым и не последним "
            "символом)"
        )
    return session_id


def is_valid_session_id(session_id: str) -> bool:
    """Допустим ли ``session_id``: та же проверка, но без исключения."""
    try:
        validate_session_id(session_id)
    except InvalidSessionIdError:
        return False
    return True


class Memory(ABC):
    """Хранилище диалогов агента между сессиями.

    Сессия — это диалог с одним ``session_id``. Контракт: хранилище работает
    с «разговорными» сообщениями (user/assistant/tool) без system-сообщений —
    системный промпт задаётся агенту отдельно и при загрузке ставится первым.
    ``search`` ищет по прошлым сессиям и используется агентом для
    кросс-сессионного recall.

    Пользователь может подставить свою реализацию (Redis, Postgres, SQLite...):
    для агента важен только этот интерфейс. Реализации, где ``session_id``
    становится именем файла или ключом с ограниченным набором символов,
    обязаны проверять его через ``validate_session_id`` и бросать
    ``InvalidSessionIdError``.
    """

    @abstractmethod
    def load_session(self, session_id: str) -> list[Message]:
        """Вернуть сообщения диалога; пустой список, если сессии ещё нет.

        Raises:
            InvalidSessionIdError: Если ``session_id`` недопустим (см. реализацию).
        """
        raise NotImplementedError

    @abstractmethod
    def save_session(self, session_id: str, messages: Sequence[Message]) -> None:
        """Сохранить диалог целиком (перезапись сессии).

        Raises:
            InvalidSessionIdError: Если ``session_id`` недопустим (см. реализацию).
        """
        raise NotImplementedError

    @abstractmethod
    def search(
        self,
        query: str,
        *,
        exclude_session_id: str | None = None,
        limit: int = 5,
    ) -> list[Message]:
        """Найти релевантные сообщения в сессиях.

        Args:
            query: Текст, по которому ищем (например, вопрос пользователя).
            exclude_session_id: Сессия, которую нужно пропустить (обычно —
                текущая: её история и так целиком в контексте агента).
            limit: Максимум сообщений в результате.

        Returns:
            Сообщения из прошлых сессий, отсортированные по релевантности.

        Raises:
            InvalidSessionIdError: Если ``exclude_session_id`` недопустим.
        """
        raise NotImplementedError

    @abstractmethod
    def delete_session(self, session_id: str) -> None:
        """Удалить диалог. Отсутствующая сессия — не ошибка.

        Raises:
            InvalidSessionIdError: Если ``session_id`` недопустим (см. реализацию).
        """
        raise NotImplementedError
