"""Провайдеры ember: интерфейс, реестр и встроенные реализации."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ember.providers.base import Provider
from ember.providers.mock import MockProvider

__all__ = ["MockProvider", "Provider", "get_provider", "register_provider"]

_REGISTRY: dict[str, type[Provider]] = {"mock": MockProvider}


def register_provider(name: str) -> Callable[[type[Provider]], type[Provider]]:
    """Декоратор: зарегистрировать класс провайдера в реестре по имени.

    Args:
        name: Имя, по которому провайдер будет доступен через get_provider().

    Returns:
        Декоратор, который добавляет класс в реестр и возвращает его без изменений.
    """

    def decorator(cls: type[Provider]) -> type[Provider]:
        _REGISTRY[name] = cls
        return cls

    return decorator


def get_provider(name: str, **kwargs: Any) -> Provider:
    """Создать экземпляр провайдера по имени из реестра.

    Args:
        name: Имя зарегистрированного провайдера (например, "mock").
        **kwargs: Аргументы, передаваемые конструктору провайдера.

    Returns:
        Новый экземпляр провайдера.

    Raises:
        ValueError: Если провайдер с таким именем не зарегистрирован.
    """
    try:
        provider_cls = _REGISTRY[name]
    except KeyError:
        available = ", ".join(sorted(_REGISTRY))
        raise ValueError(f"Неизвестный провайдер: {name!r}. Доступны: {available}") from None
    return provider_cls(**kwargs)
