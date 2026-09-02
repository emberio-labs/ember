"""Быстрый старт ember — исполняемый пример, используемый в README."""

from __future__ import annotations

import os

from ember import Agent, MockProvider, OpenAIProvider


def main() -> None:
    """Запустить примеры: без ключей (MockProvider) и с OpenAI (если есть ключ)."""
    agent = Agent(provider=MockProvider(response_text="Привет! Я мок-провайдер."))
    print(agent.run("Привет!"))

    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        openai_agent = Agent(provider=OpenAIProvider(api_key=api_key), model="gpt-4o-mini")
        print(openai_agent.run("Расскажи о себе в одном предложении."))
    else:
        print("OPENAI_API_KEY не задан — пример с OpenAIProvider пропущен.")


if __name__ == "__main__":
    main()
