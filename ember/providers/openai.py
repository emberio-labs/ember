"""Адаптер OpenAI — поверх официального SDK ``openai``.

Модуль импортирует SDK лениво (в конструкторе), чтобы ядро ember оставалось
лёгким: без установки ``openai`` остальной функционал библиотеки работает.
Установите пакет: ``pip install "ember[openai]"``.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, cast

from ember.providers.base import Provider, ProviderError
from ember.types import ChatRequest, ChatResponse, Message, StreamChunk, Usage

if TYPE_CHECKING:
    from openai import OpenAI, OpenAIError
    from openai.types.chat import ChatCompletion, ChatCompletionChunk, ChatCompletionMessageParam

_OPENAI_IMPORT_HINT = (
    "OpenAIProvider требует пакет 'openai'. Установите его: pip install 'ember[openai]'"
)


class OpenAIProvider(Provider):
    """Провайдер поверх официального OpenAI SDK.

    API-ключ передаётся явно в конструкторе; провайдер сам не читает
    окружение и файлы ``.env`` — управление ключом полностью на стороне
    вызывающего кода.

    Атрибуты:
        model: Модель по умолчанию, используемая, когда запрос её не задаёт.

    Args:
        api_key: API-ключ OpenAI (обязательный).
        model: Модель по умолчанию (например, "gpt-4o-mini").

    Raises:
        ImportError: Если пакет ``openai`` не установлен.
    """

    _client: OpenAI
    _openai_error: type[OpenAIError]

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
    ) -> None:
        try:
            from openai import OpenAI, OpenAIError
        except ImportError as exc:
            raise ImportError(_OPENAI_IMPORT_HINT) from exc

        self.model = model
        self._client = OpenAI(api_key=api_key)
        self._openai_error = OpenAIError

    def complete(self, request: ChatRequest) -> ChatResponse:
        """Получить полный ответ модели (не потоковый).

        Args:
            request: Запрос к модели.

        Returns:
            Ответ модели с текстом, моделью и счётчиками токенов.

        Raises:
            ProviderError: Если OpenAI API вернул ошибку.
        """
        try:
            completion = cast(
                "ChatCompletion",
                self._client.chat.completions.create(**self._build_params(request, stream=False)),
            )
        except self._openai_error as exc:
            raise ProviderError(f"Ошибка OpenAI API: {exc}") from exc

        if not completion.choices:
            raise ProviderError("OpenAI вернул пустой список choices")

        choice = completion.choices[0]
        content = choice.message.content or ""
        usage = completion.usage
        return ChatResponse(
            message=Message(role="assistant", content=content),
            model=completion.model,
            usage=(
                Usage(
                    prompt_tokens=usage.prompt_tokens,
                    completion_tokens=usage.completion_tokens,
                    total_tokens=usage.total_tokens,
                )
                if usage is not None
                else None
            ),
        )

    def stream(self, request: ChatRequest) -> Iterator[StreamChunk]:
        """Получить ответ модели потоком — по одному фрагменту за раз.

        Args:
            request: Запрос к модели.

        Yields:
            Фрагменты ответа модели.

        Raises:
            ProviderError: Если OpenAI API вернул ошибку.
        """
        try:
            chunks = cast(
                "Iterator[ChatCompletionChunk]",
                self._client.chat.completions.create(**self._build_params(request, stream=True)),
            )
        except self._openai_error as exc:
            raise ProviderError(f"Ошибка OpenAI API: {exc}") from exc

        for chunk in chunks:
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            yield StreamChunk(
                delta=choice.delta.content or "",
                model=chunk.model,
                finish_reason=choice.finish_reason,
            )

    def _build_params(self, request: ChatRequest, *, stream: bool) -> dict[str, Any]:
        """Собрать параметры для вызова chat.completions.create."""
        params: dict[str, Any] = {
            "model": request.model or self.model,
            "messages": [self._to_openai_message(message) for message in request.messages],
            "temperature": request.temperature,
            "stream": stream,
        }
        if request.max_tokens is not None:
            params["max_tokens"] = request.max_tokens
        return params

    @staticmethod
    def _to_openai_message(message: Message) -> ChatCompletionMessageParam:
        """Сконвертировать Message ядра в формат сообщения OpenAI."""
        params: dict[str, Any] = {"role": message.role, "content": message.content}
        if message.name is not None:
            params["name"] = message.name
        return cast("ChatCompletionMessageParam", params)
