"""ember — библиотека для простой интеграции с LLM-провайдерами и создания агентов."""

from ember.agent import Agent, ToolCallLimitError
from ember.mcp import MCPClient, MCPError
from ember.providers import (
    MockProvider,
    OpenAIProvider,
    Provider,
    ProviderError,
    get_provider,
    register_provider,
)
from ember.types import (
    ChatRequest,
    ChatResponse,
    FunctionTool,
    Message,
    StreamChunk,
    Tool,
    ToolCall,
    Usage,
)

__version__ = "0.2.0"

__all__ = [
    "__version__",
    "Agent",
    "ChatRequest",
    "ChatResponse",
    "FunctionTool",
    "MCPClient",
    "MCPError",
    "Message",
    "MockProvider",
    "OpenAIProvider",
    "Provider",
    "ProviderError",
    "StreamChunk",
    "Tool",
    "ToolCall",
    "ToolCallLimitError",
    "Usage",
    "get_provider",
    "register_provider",
]
