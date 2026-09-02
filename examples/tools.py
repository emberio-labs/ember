"""Tool calling в агенте — исполняемый пример без API-ключей.

Демонстрирует цикл: модель запрашивает инструмент ``get_weather``, агент
исполняет его локальной функцией, результат возвращается модели, после
чего модель даёт финальный текстовый ответ.
"""

from __future__ import annotations

from collections.abc import Iterator

from ember import Agent, FunctionTool
from ember.providers.base import Provider
from ember.types import ChatRequest, ChatResponse, Message, StreamChunk, ToolCall


def get_weather(city: str) -> dict[str, object]:
    """Заглушка сервиса погоды: в реальном приложении здесь был бы HTTP-запрос."""
    return {"city": city, "temperature": 18, "condition": "ясно"}


class WeatherToolMock(Provider):
    """Сценарный мок: первый ответ просит инструмент, второй — финальный текст.

    Имитирует модель: после исполнения инструмента она «видит» результат
    в истории (role=tool) и формулирует ответ пользователю.
    """

    def __init__(self) -> None:
        self._step = 0

    def complete(self, request: ChatRequest) -> ChatResponse:
        self._step += 1
        if self._step == 1:
            return ChatResponse(
                message=Message(
                    role="assistant",
                    content="",
                    tool_calls=[
                        ToolCall(id="call_1", name="get_weather", arguments='{"city": "Париж"}')
                    ],
                ),
                model="mock-tools",
            )
        return ChatResponse(
            message=Message(
                role="assistant",
                content="В Париже сейчас +18 °C, ясно.",
            ),
            model="mock-tools",
        )

    def stream(self, request: ChatRequest) -> Iterator[StreamChunk]:
        raise NotImplementedError("Потоковый режим в примере не используется")


def main() -> None:
    """Запустить пример: агент с инструментом get_weather на мок-провайдере."""
    agent = Agent(
        provider=WeatherToolMock(),
        system_prompt="Ты — погодный помощник.",
        tools=[
            FunctionTool(
                name="get_weather",
                description="Узнать текущую погоду в городе.",
                parameters={
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
                func=get_weather,
            ),
        ],
    )
    print(agent.run("Какая погода в Париже?"))


if __name__ == "__main__":
    main()
