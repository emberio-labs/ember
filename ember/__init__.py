"""ember — библиотека для простой интеграции с LLM-провайдерами и создания агентов."""

from ember.agent import Agent, ToolCallLimitError
from ember.mcp import MCPClient, MCPError
from ember.memory import FileMemory, Memory
from ember.providers import (
    MockProvider,
    OpenAIProvider,
    Provider,
    ProviderError,
    get_provider,
    register_provider,
)
from ember.skills import Skill
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

__version__ = "0.4.0"

__all__ = [
    "__version__",
    "Agent",
    "ChatRequest",
    "ChatResponse",
    "FileMemory",
    "FunctionTool",
    "MCPClient",
    "MCPError",
    "Memory",
    "Message",
    "MockProvider",
    "OpenAIProvider",
    "Provider",
    "ProviderError",
    "Skill",
    "StreamChunk",
    "Tool",
    "ToolCall",
    "ToolCallLimitError",
    "Usage",
    "get_provider",
    "register_provider",
]
