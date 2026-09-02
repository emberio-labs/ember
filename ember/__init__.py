"""ember — библиотека для простой интеграции с LLM-провайдерами и создания агентов."""

from ember.providers import (
    MockProvider,
    OpenAIProvider,
    Provider,
    ProviderError,
    get_provider,
    register_provider,
)
from ember.types import ChatRequest, ChatResponse, Message, StreamChunk, Usage

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "ChatRequest",
    "ChatResponse",
    "Message",
    "MockProvider",
    "OpenAIProvider",
    "Provider",
    "ProviderError",
    "StreamChunk",
    "Usage",
    "get_provider",
    "register_provider",
]
