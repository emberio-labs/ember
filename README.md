# ember

Библиотека для простой интеграции с LLM-провайдерами и создания собственных агентов.
Предоставляет простой и единообразный интерфейс для работы с языковыми моделями,
чтобы вы могли сосредоточиться на логике своих агентов, а не на деталях API.

## Возможности

- Единый интерфейс для различных LLM-провайдеров
- Простой способ создавать и конфигурировать собственных агентов
- Инструменты/функции для модели (tool calling)
- Минимальное количество кода для старта

> ⚠️ Проект на ранней стадии разработки. API активно меняется.

## Установка

Проект управляется через [Poetry](https://python-poetry.org/).

```bash
poetry install
```

Для работы с OpenAI дополнительно нужен пакет `openai` (устанавливается
вместе с extra `ember[openai]`):

```bash
pip install "ember[openai]"
```

## Быстрый старт

### Без ключей: `MockProvider`

Работает без сети и API-ключей — удобно для экспериментов:

```python
from ember import Agent, MockProvider

agent = Agent(provider=MockProvider(response_text="Привет! Я мок-провайдер."))
print(agent.run("Привет!"))
# Привет! Я мок-провайдер.
```

### С OpenAI: `OpenAIProvider`

API-ключ передаётся явно (провайдер сам не читает окружение):

```python
import os

from ember import Agent, OpenAIProvider

agent = Agent(
    provider=OpenAIProvider(api_key=os.environ["OPENAI_API_KEY"]),
    model="gpt-4o-mini",
)
print(agent.run("Расскажи о себе в одном предложении."))
```

Полный исполняемый пример — в [`examples/quickstart.py`](examples/quickstart.py).

### История диалога

`Agent` сам накапливает историю: сообщения пользователя и ответы модели
добавляются в `agent.messages`. Сбросить диалог можно через `agent.reset()`.
Системный промпт задаётся в конструкторе:

```python
agent = Agent(
    provider=MockProvider(),
    system_prompt="Ты краткий и полезный помощник.",
)
```

### Инструменты (tool calling)

Модель можно научить вызывать функции. Опишите инструмент через `Tool`
и передайте список в запрос:

```python
from ember import ChatRequest, Message, Tool

tools = [
    Tool(
        name="get_weather",
        description="Погода в городе",
        parameters={
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    )
]

# provider — любой Provider, например OpenAIProvider(api_key=...)
response = provider.complete(
    ChatRequest(
        messages=[Message(role="user", content="Какая погода в Москве?")],
        model="gpt-4o-mini",
        tools=tools,
    )
)
# Если модель решила вызвать инструмент, вызовы будут в response.message.tool_calls
if response.message.tool_calls:
    for call in response.message.tool_calls:
        print(call.name, call.arguments)
```

Результат выполнения возвращается модели сообщением с ролью `tool`
и идентификатором вызова:

```python
Message(role="tool", content="+15C", tool_call_id=call.id)
```

## Разработка

```bash
# Установка зависимостей (включая dev)
poetry install --with dev

# Запуск тестов
poetry run pytest

# Линтинг
poetry run ruff check .
poetry run ruff format --check .

# Проверка типов
poetry run mypy ember
```

CI (GitHub Actions) автоматически прогоняет линтинг, проверку типов и тесты
на Python 3.10–3.12 для каждого pull request.

## Релиз

Публикация новой версии на PyPI автоматизирована через GitHub Actions
(workflow `.github/workflows/publish.yml`):

1. Поднимите версию в `pyproject.toml` (`version = "0.1.0"`) и закоммитьте
   изменение, например: `chore: bump version to 0.1.0`
2. Создайте и запушьте git-тег, совпадающий с версией:

   ```bash
   git tag v0.1.0
   git push origin v0.1.0
   ```

3. Workflow соберёт wheel и sdist (`poetry build`) и опубликует их на PyPI.
   Ветка `main` при этом не нужна — достаточно тега.

Публикация использует Trusted Publishing (OIDC): секреты в GitHub не хранятся.
Для этого владельцу нужно один раз настроить publisher на PyPI:

- **PyPI:** https://pypi.org/manage/account/publishing/ —
  `emberio-labs/ember`, workflow name `publish.yml`
- **TestPyPI** (опционально, для проверки перед боевым релизом):
  https://test.pypi.org/manage/account/publishing/ — тот же publisher

После настройки проверить публикацию на TestPyPI можно вручную:
GitHub → Actions → Publish → Run workflow. На боевой PyPI пакет уходит
только по git-тегу `v*`.

## Лицензия

Проект распространяется под лицензией [MIT](LICENSE).

© 2026 Emberio Labs
