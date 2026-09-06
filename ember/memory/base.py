"""Интерфейс хранилища диалогов агента (``Memory``).

Контракт: сессия — это диалог с одним ``session_id`` из «разговорных»
сообщений (user/assistant/tool) без system-сообщений — системный промпт
задаётся агенту отдельно и при загрузке ставится первым. ``search`` ищет
по прошлым сессиям и используется агентом для кросс-сессионного recall.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from ember.types import Message


class Memory(ABC):
    """Хранилище диалогов агента между сессиями.

    Сессия — это диалог с одним ``session_id``. Контракт: хранилище работает
    с «разговорными» сообщениями (user/assistant/tool) без system-сообщений —
    системный промпт задаётся агенту отдельно и при загрузке ставится первым.
    ``search`` ищет по прошлым сессиям и используется агентом для
    кросс-сессионного recall.

    Пользователь может подставить свою реализацию (Redis, Postgres, SQLite...):
    для агента важен только этот интерфейс.
    """

    @abstractmethod
    def load_session(self, session_id: str) -> list[Message]:
        """Вернуть сообщения диалога; пустой список, если сессии ещё нет."""
        raise NotImplementedError

    @abstractmethod
    def save_session(self, session_id: str, messages: Sequence[Message]) -> None:
        """Сохранить диалог целиком (перезапись сессии)."""
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
        """
        raise NotImplementedError

    @abstractmethod
    def delete_session(self, session_id: str) -> None:
        """Удалить диалог. Отсутствующая сессия — не ошибка."""
        raise NotImplementedError
