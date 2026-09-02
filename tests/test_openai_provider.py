"""Тесты OpenAIProvider — мокают SDK openai, без реальных запросов и ключей."""

from types import SimpleNamespace

import openai
import pytest

from ember.providers import OpenAIProvider, ProviderError, get_provider
from ember.types import ChatRequest, ChatResponse, Message, Usage


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
        self.chat = FakeChat()


@pytest.fixture
def fake_client(monkeypatch: pytest.MonkeyPatch) -> FakeClient:
    client = FakeClient()

    def factory(**kwargs: object) -> FakeClient:
        client.api_key = kwargs.get("api_key")
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


def _choice(content: str = "Hello", finish_reason: str = "stop") -> SimpleNamespace:
    return SimpleNamespace(
        message=SimpleNamespace(role="assistant", content=content),
        finish_reason=finish_reason,
    )


def _completion(
    *,
    content: str = "Hello",
    model: str = "gpt-4o-mini",
    usage: SimpleNamespace | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(choices=[_choice(content)], model=model, usage=usage)


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


def test_openai_missing_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        OpenAIProvider()


def test_openai_api_key_from_env(monkeypatch: pytest.MonkeyPatch, fake_client: FakeClient) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")

    OpenAIProvider()

    assert fake_client.api_key == "env-key"


def test_registry_get_openai(fake_client: FakeClient) -> None:
    provider = get_provider("openai", api_key="test-key")

    assert isinstance(provider, OpenAIProvider)
    assert provider.model == "gpt-4o-mini"
