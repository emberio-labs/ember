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
