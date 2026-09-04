"""Тесты OpenAIProvider — мокают SDK openai, без реальных запросов и ключей."""

from types import SimpleNamespace

import openai
import pytest

from ember.providers import OpenAIProvider, ProviderError, get_provider
from ember.types import ChatRequest, ChatResponse, Message, Tool, ToolCall, Usage


class FakeCompletions:
    """Заглушка для client.chat.completions: записывает вызовы, отдаёт результат."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self._result: object = None

    def set_result(self, result: object) -> None:
        self._result = result

    def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        if callable(self._result):
            return self._result(**kwargs)
        return self._result


class FakeChat:
    def __init__(self) -> None:
        self.completions = FakeCompletions()


class FakeClient:
    """Заглушка для класса openai.OpenAI."""

    def __init__(self, **kwargs: object) -> None:
        self.api_key = kwargs.get("api_key")
        self.base_url = kwargs.get("base_url")
        self.chat = FakeChat()


@pytest.fixture
def fake_client(monkeypatch: pytest.MonkeyPatch) -> FakeClient:
    client = FakeClient()

    def factory(**kwargs: object) -> FakeClient:
        client.api_key = kwargs.get("api_key")
        client.base_url = kwargs.get("base_url")
        return client

    monkeypatch.setattr("openai.OpenAI", factory)
    return client


def _request(**overrides: object) -> ChatRequest:
    params: dict[str, object] = {
        "messages": [Message(role="user", content="Привет")],
        "model": "gpt-4o-mini",
    }
    params.update(overrides)
    return ChatRequest(**params)


def _choice(
    content: str = "Hello",
    finish_reason: str = "stop",
    tool_calls: list[SimpleNamespace] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        message=SimpleNamespace(role="assistant", content=content, tool_calls=tool_calls),
        finish_reason=finish_reason,
    )


def _tool_call(
    id: str = "call_1",
    name: str = "get_weather",
    arguments: str = '{"city": "Moscow"}',
) -> SimpleNamespace:
    return SimpleNamespace(
        id=id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _completion(
    *,
    content: str = "Hello",
    model: str = "gpt-4o-mini",
    usage: SimpleNamespace | None = None,
    tool_calls: list[SimpleNamespace] | None = None,
) -> SimpleNamespace:
    choice = _choice(content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[choice], model=model, usage=usage)


def _usage(prompt: int = 10, completion: int = 5) -> SimpleNamespace:
    return SimpleNamespace(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=prompt + completion,
    )


def _chunk(
    delta: str = "",
    finish_reason: str | None = None,
    model: str = "gpt-4o-mini",
    with_choices: bool = True,
) -> SimpleNamespace:
    choices = (
        [SimpleNamespace(delta=SimpleNamespace(content=delta), finish_reason=finish_reason)]
        if with_choices
        else []
    )
    return SimpleNamespace(choices=choices, model=model)


def test_openai_complete_maps_request(fake_client: FakeClient) -> None:
    provider = OpenAIProvider(api_key="test-key")
    fake_client.chat.completions.set_result(_completion())

    request = ChatRequest(
        messages=[
            Message(role="system", content="Ты помощник"),
            Message(role="user", content="Привет"),
        ],
        model="gpt-4o-mini",
        temperature=0.3,
    )
    provider.complete(request)

    kwargs = fake_client.chat.completions.calls[-1]
    assert kwargs["model"] == "gpt-4o-mini"
    assert kwargs["temperature"] == 0.3
    assert kwargs["stream"] is False
    assert kwargs["messages"] == [
        {"role": "system", "content": "Ты помощник"},
        {"role": "user", "content": "Привет"},
    ]
    assert "max_tokens" not in kwargs


def test_openai_complete_with_max_tokens(fake_client: FakeClient) -> None:
    provider = OpenAIProvider(api_key="test-key")
    fake_client.chat.completions.set_result(_completion())

    provider.complete(_request(max_tokens=100))

    assert fake_client.chat.completions.calls[-1]["max_tokens"] == 100


def test_openai_complete_with_name(fake_client: FakeClient) -> None:
    provider = OpenAIProvider(api_key="test-key")
    fake_client.chat.completions.set_result(_completion())

    request = ChatRequest(
        messages=[Message(role="user", content="Привет", name="ivan")],
        model="gpt-4o-mini",
    )
    provider.complete(request)

    assert fake_client.chat.completions.calls[-1]["messages"] == [
        {"role": "user", "content": "Привет", "name": "ivan"}
    ]


def test_openai_complete_maps_response(fake_client: FakeClient) -> None:
    provider = OpenAIProvider(api_key="test-key")
    fake_client.chat.completions.set_result(
        _completion(content="Привет, мир!", model="gpt-4o-mini", usage=_usage(10, 5))
    )

    response = provider.complete(_request())

    assert isinstance(response, ChatResponse)
    assert response.message.role == "assistant"
    assert response.message.content == "Привет, мир!"
    assert response.model == "gpt-4o-mini"
    assert response.usage == Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15)


def test_openai_complete_without_usage(fake_client: FakeClient) -> None:
    provider = OpenAIProvider(api_key="test-key")
    fake_client.chat.completions.set_result(_completion(usage=None))

    response = provider.complete(_request())

    assert response.usage is None


def test_openai_default_model_from_constructor(fake_client: FakeClient) -> None:
    provider = OpenAIProvider(api_key="test-key", model="gpt-4o")
    fake_client.chat.completions.set_result(_completion())

    provider.complete(_request(model=""))

    assert fake_client.chat.completions.calls[-1]["model"] == "gpt-4o"


def test_openai_passes_api_key_to_client(fake_client: FakeClient) -> None:
    OpenAIProvider(api_key="test-key")

    assert fake_client.api_key == "test-key"


def test_openai_passes_base_url_to_client(fake_client: FakeClient) -> None:
    OpenAIProvider(api_key="test-key", base_url="http://localhost:1234/v1")

    assert fake_client.base_url == "http://localhost:1234/v1"


def test_openai_without_base_url_omits_client_arg(fake_client: FakeClient) -> None:
    OpenAIProvider(api_key="test-key")

    assert fake_client.base_url is None


def test_openai_complete_with_base_url(fake_client: FakeClient) -> None:
    provider = OpenAIProvider(
        api_key="test-key", base_url="https://api.groq.com/openai/v1", model="llama-3.1-8b"
    )
    fake_client.chat.completions.set_result(_completion(model="llama-3.1-8b"))

    response = provider.complete(_request(model="llama-3.1-8b"))

    assert fake_client.base_url == "https://api.groq.com/openai/v1"
    assert response.message.content == "Hello"
    assert fake_client.chat.completions.calls[-1]["model"] == "llama-3.1-8b"


def test_openai_stream_params(fake_client: FakeClient) -> None:
    provider = OpenAIProvider(api_key="test-key")
    fake_client.chat.completions.set_result([])

    list(provider.stream(_request()))

    assert fake_client.chat.completions.calls[-1]["stream"] is True


def test_openai_stream_chunks(fake_client: FakeClient) -> None:
    provider = OpenAIProvider(api_key="test-key")
    fake_client.chat.completions.set_result(
        [
            _chunk(delta="При", finish_reason=None),
            _chunk(delta="вет", finish_reason="stop"),
        ]
    )

    chunks = list(provider.stream(_request()))

    assert "".join(c.delta for c in chunks) == "Привет"
    assert chunks[0].finish_reason is None
    assert chunks[-1].finish_reason == "stop"
    assert all(c.model == "gpt-4o-mini" for c in chunks)


def test_openai_stream_skips_empty_choices(fake_client: FakeClient) -> None:
    provider = OpenAIProvider(api_key="test-key")
    fake_client.chat.completions.set_result(
        [
            _chunk(with_choices=False),
            _chunk(delta="ok", finish_reason="stop"),
        ]
    )

    chunks = list(provider.stream(_request()))

    assert len(chunks) == 1
    assert chunks[0].delta == "ok"


def test_openai_error_wrapped(fake_client: FakeClient) -> None:
    provider = OpenAIProvider(api_key="test-key")

    def boom(**kwargs: object) -> object:
        raise openai.OpenAIError("invalid api key")

    fake_client.chat.completions.set_result(boom)

    with pytest.raises(ProviderError, match="invalid api key"):
        provider.complete(_request())


def test_openai_complete_sends_tools(fake_client: FakeClient) -> None:
    provider = OpenAIProvider(api_key="test-key")
    fake_client.chat.completions.set_result(_completion())

    parameters = {"type": "object", "properties": {"city": {"type": "string"}}}
    request = _request(
        tools=[
            Tool(
                name="get_weather",
                description="Погода в городе",
                parameters=parameters,
            )
        ]
    )
    provider.complete(request)

    assert fake_client.chat.completions.calls[-1]["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Погода в городе",
                "parameters": parameters,
            },
        }
    ]


def test_openai_complete_without_tools_omits_key(fake_client: FakeClient) -> None:
    provider = OpenAIProvider(api_key="test-key")
    fake_client.chat.completions.set_result(_completion())

    provider.complete(_request())

    assert "tools" not in fake_client.chat.completions.calls[-1]


def test_openai_complete_parses_tool_calls(fake_client: FakeClient) -> None:
    provider = OpenAIProvider(api_key="test-key")
    fake_client.chat.completions.set_result(
        _completion(
            content=None,
            tool_calls=[
                _tool_call(id="call_1", name="get_weather", arguments='{"city": "Moscow"}'),
                _tool_call(id="call_2", name="get_time", arguments="{}"),
            ],
        )
    )

    response = provider.complete(_request())

    assert response.message.content == ""
    assert response.message.tool_calls == [
        ToolCall(id="call_1", name="get_weather", arguments='{"city": "Moscow"}'),
        ToolCall(id="call_2", name="get_time", arguments="{}"),
    ]


def test_openai_history_with_tool_calls_and_results(fake_client: FakeClient) -> None:
    provider = OpenAIProvider(api_key="test-key")
    fake_client.chat.completions.set_result(_completion())

    request = ChatRequest(
        messages=[
            Message(
                role="assistant",
                content="",
                tool_calls=[
                    ToolCall(id="call_1", name="get_weather", arguments='{"city": "Moscow"}')
                ],
            ),
            Message(role="tool", content="+15C", tool_call_id="call_1"),
            Message(role="user", content="Спасибо"),
        ],
        model="gpt-4o-mini",
    )
    provider.complete(request)

    assert fake_client.chat.completions.calls[-1]["messages"] == [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city": "Moscow"}'},
                }
            ],
        },
        {"role": "tool", "content": "+15C", "tool_call_id": "call_1"},
        {"role": "user", "content": "Спасибо"},
    ]


def test_registry_get_openai(fake_client: FakeClient) -> None:
    provider = get_provider("openai", api_key="test-key")

    assert isinstance(provider, OpenAIProvider)
    assert provider.model == "gpt-4o-mini"
    assert fake_client.api_key == "test-key"


def test_registry_get_openai_with_base_url(fake_client: FakeClient) -> None:
    provider = get_provider("openai", api_key="test-key", base_url="http://localhost:1234/v1")

    assert isinstance(provider, OpenAIProvider)
    assert fake_client.api_key == "test-key"
    assert fake_client.base_url == "http://localhost:1234/v1"
