# ember

Библиотека для простой интеграции с LLM-провайдерами и создания собственных агентов.
Предоставляет простой и единообразный интерфейс для работы с языковыми моделями,
чтобы вы могли сосредоточиться на логике своих агентов, а не на деталях API.

## Возможности

- Единый интерфейс для различных LLM-провайдеров
- Простой способ создавать и конфигурировать собственных агентов
- Инструменты/функции для модели (tool calling)
- Подключение внешних инструментов по MCP (Model Context Protocol)
- Скиллы агента по стандарту Agent Skills: каталог в промпте, загрузка тела по требованию
- Персистентная память агента: диалоги между сессиями и кросс-сессионный recall
- Минимальное количество кода для старта

> ⚠️ Проект на ранней стадии разработки. API активно меняется.

## Установка

На PyPI пакет публикуется под именем `emberio-labs-ember` (импорт в коде — `ember`):

```bash
pip install "emberio-labs-ember[openai]"   # с поддержкой OpenAI
pip install "emberio-labs-ember[mcp]"      # с поддержкой MCP-клиента
pip install emberio-labs-ember             # ядро (без провайдеров)
```

Для разработки (из репозитория) проект управляется через
[Poetry](https://python-poetry.org/):

```bash
poetry install
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

### OpenAI-совместимые API (`base_url`)

Chat Completions — де-факто индустриальный стандарт: его поддерживают
облачные провайдеры (OpenRouter, Groq, DeepSeek, Mistral, Perplexity)
и локальные/self-hosted серверы (LM Studio, vLLM, LocalAI). Тот же
`OpenAIProvider` подключается к любому из них через `base_url`:

```python
import os

from ember import Agent, OpenAIProvider

# облачный провайдер (пример: Groq)
agent = Agent(
    provider=OpenAIProvider(
        api_key=os.environ["GROQ_API_KEY"],
        base_url="https://api.groq.com/openai/v1",
        model="llama-3.1-8b-instant",
    ),
)

# локальный сервер (пример: LM Studio) — ключ не требуется, передайте заглушку
provider = OpenAIProvider(
    api_key="sk-local",
    base_url="http://localhost:1234/v1",
    model="local-model",
)
```

`base_url` — это полный URL до версии API: SDK сам добавит
`/chat/completions`, а `/v1` дописывать за вас никто не будет. Ключ
остаётся обязательным параметром конструктора; локальные серверы обычно
игнорируют его значение.

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

### Память агента: персистентные сессии и recall

По умолчанию история диалога живёт в памяти агента и исчезает вместе с ним.
Подключив хранилище — интерфейс `Memory` с реализацией по умолчанию
`FileMemory` — вы получите диалоги, переживающие перезапуск приложения,
и кросс-сессионный recall: агент «вспоминает» релевантное из прошлых разговоров.

```python
from ember import Agent, FileMemory, MockProvider

memory = FileMemory("/tmp/ember-memory")  # директория создаётся сама

agent = Agent(
    provider=MockProvider(response_text="Привет, Алекс!"),
    memory=memory,
    session_id="user-42",  # новая беседа = новый session_id
)
agent.run("Привет, меня зовут Алекс")

# «Перезапуск приложения»: тот же session_id — диалог продолжается
restored = Agent(
    provider=MockProvider(response_text="Привет, Алекс!"),
    memory=memory,
    session_id="user-42",
)
print([m.content for m in restored.messages])
# ['Привет, меня зовут Алекс', 'Привет, Алекс!']
```

Как это работает:

- `memory` и `session_id` задаются **вместе**: без `session_id` диалог некуда
  сохранять. `FileMemory` хранит каждую сессию в отдельном JSONL-файле
  `<directory>/<session_id>.json`.
- При создании агент загружает историю сессии и продолжает диалог с неё.
  В хранилище пишется только «разговорная» часть (user/assistant/tool) —
  system-промпт остаётся конфигурацией агента и ставится первым при создании.
- Сохранение происходит после каждого `run()`/`stream_run()`, в том числе при
  завершении с исключением: накопленная часть диалога не теряется.
- `reset()` начинает текущую сессию заново (и очищает её в хранилище);
  сессии с другими `session_id` не трогаются.
- **Recall**: в `run()`/`stream_run()` агент ищет по тексту запроса релевантные
  сообщения в *других* сессиях (bag-of-words overlap, без LLM) и добавляет их
  в запрос отдельным system-сообщением «Из прошлых сессий: ...». История
  и хранилище при этом не изменяются.

Своё хранилище (Redis, Postgres, SQLite...) подключить просто: реализуйте
`Memory` — `load_session`/`save_session`/`search`/`delete_session` — и передайте
в `Agent`. Полный исполняемый пример — [`examples/memory.py`](examples/memory.py).

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

### MCP: внешние инструменты (Model Context Protocol)

[MCP](https://modelcontextprotocol.io) — открытый стандарт интеграции
инструментов с LLM-агентами. `ember` умеет подключаться к MCP-серверам
(клиентская часть) и использовать их инструменты наряду с локальными
`FunctionTool`. Сервер при этом может быть как вашим процессом, так
и сторонним сервисом.

Подключение к серверу — через `MCPClient`:

```python
from ember import Agent, MCPClient

# stdio: сервер запускается как дочерний процесс
with MCPClient.stdio(command="python", args=["server.py"]) as mcp:
    agent = Agent(provider=provider, tools=mcp.list_tools())
    print(agent.run("Проверь доступность сервиса."))
```

`MCPClient.list_tools()` выполняет `tools/list` и возвращает обычные
`FunctionTool`: JSON Schema параметров сохраняется, а `func` проксирует
вызов на сервер (`tools/call`). Для агента такие инструменты ничем не
отличаются от локальных — они исполняются в цикле `agent.run()`,
результаты возвращаются модели сообщениями `role=tool`, а в одном агенте
можно смешивать MCP-инструменты и локальные функции.

Поддерживается и streamable HTTP транспорт — передайте URL endpoint:

```python
from ember import Agent, MCPClient

with MCPClient.http("http://127.0.0.1:8000/mcp") as mcp:
    agent = Agent(provider=provider, tools=mcp.list_tools())
    print(agent.run("Проверь доступность сервиса."))
```

Если сервер требует аутентификацию или другие HTTP-заголовки, передайте
их словарём в `headers=` — заголовки уходят с каждым запросом:

```python
with MCPClient.http(
    "https://mcp.example.com/mcp",
    headers={"Authorization": "Bearer YOUR_TOKEN"},
) as mcp:
    tools = mcp.list_tools()
```

Ошибки транспорта и сервера (падение процесса, таймаут, сбой `tools/call`,
отказ HTTP-сервера) оборачиваются в `MCPError` — понятное исключение
по образцу `ProviderError`. Требуется пакет `mcp` (extra `ember[mcp]`).

Полный исполняемый пример — [`examples/mcp_client.py`](examples/mcp_client.py):
он запускает MCP-сервер с инструментом `ping` и исполняет его агентом
без API-ключей:

```bash
python examples/mcp_client.py
```

### Скиллы агента (Agent Skills)

Скилл — переносимая инструкция в формате
[Agent Skills](https://agentskills.io): каталог `<name>/SKILL.md`
с YAML-frontmatter (`name`, `description`) и markdown-телом инструкции.
Работают скиллы по принципу **progressive disclosure**: модель всегда видит
только каталог «имя + описание», а полное тело подгружает, когда описание
подходит к задаче. Формат — стандартный, поэтому скиллы переносимы между
клиентами.

```python
from ember import Agent, MockProvider

agent = Agent(
    provider=MockProvider(),
    skills_dirs=[".ember/skills", "~/.config/ember/skills"],  # порядок = приоритет
    system_prompt="Ты — ассистент разработчика.",
)
```

Как это работает:

- Каталоги сканируются по шаблону `<каталог>/<name>/SKILL.md` (глубже одного
  уровня не смотрим); каталоги перебираются по порядку, и при коллизии имён
  побеждает скилл из более раннего.
- В промпт добавляется tier-1 каталог скиллов (`name` + `description`)
  отдельным system-сообщением — история агента при этом не меняется, а каталог
  перечитывается с диска на каждый запрос: скилл, сохранённый в этом же
  диалоге, виден уже в следующем запросе.
- Модели становятся доступны два инструмента: `read_skill(name)` — подгрузить
  тело скилла и `save_skill(...)` — создать или обновить скилл.
- `skills_dirs=None` (по умолчанию) — скиллы выключены: ни инструментов,
  ни секции в промпте. Дефолтных путей библиотека не подставляет: каталоги
  задаёт потребитель. Инструменты и скиллы вместе работают в `run()`;
  `stream_run()` их не поддерживает.

Запись — **только по явной просьбе пользователя** и только в **первый**
каталог списка. `save_skill` строго валидирует данные по стандарту: имя
(строчные латинские буквы, цифры и одиночные дефисы, ≤64 символов), описание
(≤1024 символов) и непустое тело; писать за пределы целевого каталога
инструмент отказывается. Чтение чужих скиллов — лояльное: битый YAML
разбирается грубо, скилл без `name`/`description` пропускается, расхождение
имени с каталогом даёт предупреждение. Опциональные поля стандарта (`license`,
`compatibility`, `metadata`, `allowed-tools`) сохраняются и читаются, но
в v1 не исполняются.

> ⚠️ Скиллы из недоверенных источников могут подмешивать свои инструкции
> в контекст модели. Подключайте каталоги, которым доверяете; trust-гейт —
> в планах.

Полный исполняемый пример — [`examples/skills.py`](examples/skills.py): он
сохраняет скилл, показывает каталог, который получает модель, и подгружает
тело скилла:

```bash
python examples/skills.py
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
Для этого владельцу нужно один раз настроить publisher на PyPI
(и, опционально, на TestPyPI для проверок):

- **PyPI:** https://pypi.org/manage/account/publishing/
- **TestPyPI:** https://test.pypi.org/manage/account/publishing/

Поля формы одинаковы для PyPI и TestPyPI:

| Поле | Значение |
|---|---|
| Project name | `emberio-labs-ember` |
| GitHub owner | `emberio-labs` |
| GitHub repository | `ember` |
| Workflow name | `publish.yml` |
| Environment | *(пусто)* |

После настройки публикацию можно проверить вручную на TestPyPI:
GitHub → Actions → Publish → Run workflow. На боевой PyPI пакет уходит
только по git-тегу `v*`.

## Лицензия

Проект распространяется под лицензией [MIT](LICENSE).

© 2026 Emberio Labs
