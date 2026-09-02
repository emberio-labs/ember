"""Тесты Agent — на MockProvider и провайдере-шпионе."""

from collections.abc import Iterator

from ember.agent import Agent
from ember.providers.base import Provider
from ember.types import ChatRequest, ChatResponse, Message, StreamChunk, Usage


class RecordingProvider(Provider):
    """Провайдер-шпион: записывает все запросы и возвращает фиксированный ответ."""

    def __init__(self, response_text: str = "ответ агента") -> None:
        self.response_text = response_text
        self.requests: list[ChatRequest] = []

    def complete(self, request: ChatRequest) -> ChatResponse:
        self.requests.append(request)
        return ChatResponse(
            message=Message(role="assistant", content=self.response_text),
            model="rec-1",
            usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )

    def stream(self, request: ChatRequest) -> Iterator[StreamChunk]:
        self.requests.append(request)
        words = self.response_text.split()
        for index, word in enumerate(words):
            last = index == len(words) - 1
            yield StreamChunk(
                delta=word if last else word + " ",
                model="rec-1",
                finish_reason="stop" if last else None,
            )


def test_agent_returns_response_text() -> None:
    provider = RecordingProvider(response_text="Привет, мир")
    agent = Agent(provider)

    assert agent.run("Как дела?") == "Привет, мир"


def test_agent_system_prompt_first_message() -> None:
    provider = RecordingProvider()
    agent = Agent(provider, system_prompt="Ты полезный помощник")

    agent.run("Привет")

    first = provider.requests[0].messages[0]
    assert first.role == "system"
    assert first.content == "Ты полезный помощник"


def test_agent_accumulates_history() -> None:
    provider = RecordingProvider(response_text="ответ")
    agent = Agent(provider)

    agent.run("первый вопрос")
    agent.run("второй вопрос")

    assert [m.content for m in agent.messages] == [
        "первый вопрос",
        "ответ",
        "второй вопрос",
        "ответ",
    ]
    second_request = provider.requests[1]
    assert [m.content for m in second_request.messages] == [
        "первый вопрос",
        "ответ",
        "второй вопрос",
    ]


def test_agent_model_from_constructor() -> None:
    provider = RecordingProvider()
    agent = Agent(provider, model="gpt-4o-mini")

    agent.run("Привет")

    assert provider.requests[0].model == "gpt-4o-mini"


def test_agent_model_defaults_to_provider() -> None:
    provider = RecordingProvider()
    agent = Agent(provider)

    agent.run("Привет")

    assert provider.requests[0].model == ""


def test_agent_reset_clears_history() -> None:
    provider = RecordingProvider()
    agent = Agent(provider, system_prompt="Ты помощник")

    agent.run("Привет")
    agent.run("Пока")
    agent.reset()

    assert [m.role for m in agent.messages] == ["system"]

    agent.run("Снова")

    assert len(agent.messages) == 3  # system, user, assistant


def test_agent_stream_returns_chunks() -> None:
    provider = RecordingProvider(response_text="Привет мир")
    agent = Agent(provider)

    chunks = list(agent.stream_run("Привет"))

    assert "".join(chunks) == "Привет мир"


def test_agent_stream_updates_history() -> None:
    provider = RecordingProvider(response_text="Привет мир")
    agent = Agent(provider)

    list(agent.stream_run("Привет"))

    assert [m.content for m in agent.messages] == ["Привет", "Привет мир"]
