# StruktX Documentation

StruktX is a lean, typed framework for building Natural Language → Action applications. It provides swappable components for LLMs, classifiers, handlers, and optional memory, along with middleware hooks and LangChain helpers.

## Getting Started

### Quickstart

Minimal example with the default components:

```python
from strukt import create, StruktConfig, HandlersConfig

app = create(StruktConfig(
    handlers=HandlersConfig(default_route="general")
))

print(app.invoke("Hello, StruktX!").response)
```

With LangChain and memory augmentation:

```python
import os
from strukt import create, StruktConfig, HandlersConfig, LLMClientConfig, ClassifierConfig, MemoryConfig

os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

config = StruktConfig(
  llm=LLMClientConfig("langchain_openai:ChatOpenAI", dict(model="gpt-4o-mini")),
  classifier=ClassifierConfig(factory=None),
  handlers=HandlersConfig(default_route="general"),
  memory=MemoryConfig(factory=None, use_store=False, augment_llm=True),
)

app = create(config)
res = app.invoke("I live in Beirut, what's the time?", context={"user_id": "u1"})
print(res.response)
```

### Architecture

Core flow: text → classify → group parts → route to handlers → combine responses.
Components are swappable via factories and follow interfaces for type safety.

- **LLM Client**: Any model client implementing invoke/structured.
- **Classifier**: Maps input to one or more query types and parts.
- **Handlers**: Process grouped parts per query type.
- **Memory**: Optional, supports scoped retrieval and injection.
- **Middleware**: Before/after classify and handle hooks.

### Configuration

Factory-based config supports callables, classes, instances, or import strings like "module:attr". Dicts are coerced into dataclasses.

```python
from strukt import (
  create, StruktConfig, LLMClientConfig, ClassifierConfig,
  HandlersConfig, MemoryConfig, MiddlewareConfig
  )

config = StruktConfig(
  llm=LLMClientConfig(factory="langchain_openai:ChatOpenAI", params=dict(model="gpt-4o-mini")),
  classifier=ClassifierConfig(factory=None),
  handlers=HandlersConfig(
    registry={
      # "time_service": "your_pkg.handlers:TimeHandler",
    },
    default_route="general",
  ),
  memory=MemoryConfig(factory=None, params={}, use_store=False, augment_llm=True),
  middleware=[MiddlewareConfig(factory="strukt.logging:LoggingMiddleware", params=dict(verbose=True))],
)

app = create(config)
```

Swap LLMs and pass custom parameters (including OpenAI-compatible providers):

```python
from strukt import StruktConfig, LLMClientConfig

# OpenAI-compatible provider
cfg = StruktConfig(
  llm=LLMClientConfig(
    factory="langchain_openai:ChatOpenAI",
    params=dict(
      model="gpt-4o-mini",
      api_key="{OPENAI_API_KEY}",
      base_url="https://api.openai.com/v1"  # or your compatible endpoint
    ),
  )
)
```

## Core Components

### Providers (OpenAI-compatible)

OpenRouter, Groq, and Cerebras expose OpenAI-style APIs and work via LangChain's `ChatOpenAI` or direct OpenAI clients using a custom base URL.

#### OpenRouter

```python
import os
from strukt import create, StruktConfig, LLMClientConfig

os.environ["OPENROUTER_API_KEY"] = "..."

cfg = StruktConfig(
  llm=LLMClientConfig(
    factory="langchain_openai:ChatOpenAI",
    params=dict(
      model="openrouter/auto",
      api_key=os.environ["OPENROUTER_API_KEY"],
      base_url="https://openrouter.ai/api/v1",
    ),
  )
)
app = create(cfg)
```

#### Groq

```python
import os
from strukt import create, StruktConfig, LLMClientConfig

os.environ["GROQ_API_KEY"] = "..."

cfg = StruktConfig(
  llm=LLMClientConfig(
    factory="langchain_openai:ChatOpenAI",
    params=dict(
      model="llama3-70b-8192",
      api_key=os.environ["GROQ_API_KEY"],
      base_url="https://api.groq.com/openai/v1",
    ),
  )
)
app = create(cfg)
```

#### Cerebras

```python
import os
from strukt import create, StruktConfig, LLMClientConfig

os.environ["CEREBRAS_API_KEY"] = "..."

cfg = StruktConfig(
  llm=LLMClientConfig(
    factory="langchain_openai:ChatOpenAI",
    params=dict(
      model="llama3.1-70b",
      api_key=os.environ["CEREBRAS_API_KEY"],
      base_url="https://api.cerebras.ai/v1",
    ),
  )
)
app = create(cfg)
```

Alternatively, if you use the OpenAI SDK directly, set `OPENAI_BASE_URL` env and pass your key. StruktX will auto-adapt LangChain runnables to `LLMClient`.

### LLM Clients

Provide your own `LLMClient` or adapt LangChain with `LangChainLLMClient`.

```python
from strukt.interfaces import LLMClient

class MyLLM(LLMClient):
    def invoke(self, prompt: str, **kwargs):
        return type("Resp", (), {"content": prompt.upper()})

    def structured(self, prompt: str, output_model, **kwargs):
        return output_model()
```

### Classifier

Return query types, confidences, and parts.

```python
from strukt.interfaces import Classifier
from strukt.types import InvocationState, QueryClassification

class MyClassifier(Classifier):
    def classify(self, state: InvocationState) -> QueryClassification:
        return QueryClassification(query_types=["general"], confidences=[1.0], parts=[state.text])
```

### Handlers

Handle grouped parts for a given query type.

```python
from strukt.interfaces import Handler, LLMClient
from strukt.types import InvocationState, HandlerResult

class EchoHandler(Handler):
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def handle(self, state: InvocationState, parts: list[str]) -> HandlerResult:
        return HandlerResult(response=" | ".join(parts), status="general")
```

### Middleware

Hooks: before_classify, after_classify, before_handle, after_handle.

```python
from strukt.middleware import Middleware
from strukt.types import InvocationState, HandlerResult, QueryClassification

class Metrics(Middleware):
    def before_classify(self, state: InvocationState):
        state.context["t0"] = 0
        return state

    def after_handle(self, state: InvocationState, query_type: str, result: HandlerResult):
        return result
```

Memory extraction middleware (packaged):

```python
from strukt import StruktConfig, MemoryConfig, MiddlewareConfig

config = StruktConfig(
  memory=MemoryConfig(
    factory="strukt.memory:UpstashVectorMemoryEngine",
    params={"index_url": "...", "index_token": "...", "namespace": "app1"},
    use_store=True,
    augment_llm=True,
  ),
  middleware=[
    MiddlewareConfig("strukt.memory.middleware:MemoryExtractionMiddleware"),
  ],
)
```

### Memory

Enable scoped memory and automatic prompt augmentation.

```python
from strukt import StruktConfig, MemoryConfig

cfg = StruktConfig(
  memory=MemoryConfig(factory=None, use_store=False, augment_llm=True)
)
```

## Ecosystem

### LangChain Helpers

Use `LangChainLLMClient` and `create_structured_chain` to generate typed outputs.

```python
from pydantic import BaseModel
from strukt.langchain_helpers import create_structured_chain

class Foo(BaseModel):
    query_types: list[str] = []
    confidences: list[float] = []
    parts: list[str] = []

# chain = create_structured_chain(llm_client=your_langchain_client, prompt_template="...", output_model=Foo)
```

### Logging

Use `get_logger` and `LoggingMiddleware`.
Optional logging variables: `STRUKTX_LOG_LEVEL`, `STRUKTX_LOG_MAXLEN`, `STRUKTX_RICH_TRACEBACK`, `STRUKTX_DEBUG`.

```python
from strukt import get_logger

log = get_logger("struktx")
log.info("Hello logs")
```

Augmented memory injections appear under the `memory` logger with the provided `augment_source` label.

### Memory Extraction Middleware

Automatically extracts durable facts from conversations and stores them in your memory engine (e.g., Upstash Vector). On subsequent requests, `MemoryAugmentedLLMClient` retrieves relevant items and prepends them to prompts.

- **Extraction**: After handler or classification, extracts facts (e.g., preferences, locations).
- **Storage**: Writes to memory with scope from `context.user_id` and optionally `context.unit_id`.
- **Retrieval**: On the next request, scoped items are used to enrich prompts.

```python
from strukt import create, StruktConfig, MemoryConfig
from strukt import HandlersConfig, LLMClientConfig, ClassifierConfig, MiddlewareConfig

cfg = StruktConfig(
  llm=LLMClientConfig("langchain_openai:ChatOpenAI", dict(model="gpt-4o-mini")),
  handlers=HandlersConfig(default_route="general"),
  memory=MemoryConfig(
    factory="strukt.memory:UpstashVectorMemoryEngine",
    params={"index_url": "...", "index_token": "...", "namespace": "app1"},
    use_store=True,
    augment_llm=True,
  ),
  middleware=[
    MiddlewareConfig("strukt.memory.middleware:MemoryExtractionMiddleware", params={"max_items": 5}),
  ],
)
app = create(cfg)
```

**Note**: `MemoryConfig.use_store=True` is required for `MemoryExtractionMiddleware` to work.
It currently only reads context of `user_id` and `unit_id`, if you wish to change this behavior, you can implement a custom `MemoryAugmentedLLMClient` and `MemoryExtractionMiddleware`.

Tips:
- Always scope memories per user/session.
- Keep extracted items concise and factual. Avoid storing ephemeral content.
- Control verbosity with env and middleware params. Logs show when memory is injected.

### Extensions

Build reusable packages exposing factories for handlers, middleware, and memory engines.

```python
# your_extension/__init__.py
from .handlers import DeviceHandler
from .middleware import DeviceAuthMiddleware
from .models import DeviceCommand

__all__ = ["DeviceHandler", "DeviceAuthMiddleware", "DeviceCommand"]
```

### Devices Extension

Example of building a devices extension that receives natural language and triggers device actions.

#### Models

```python
from pydantic import BaseModel, Field

class DeviceCommand(BaseModel):
    device_id: str = Field(..., min_length=1)
    action: str = Field(..., min_length=1)  # e.g., "turn_on", "set_temp"
    value: str | int | float | None = None
```

#### Handler

```python
from strukt.interfaces import Handler, LLMClient
from strukt.types import InvocationState, HandlerResult
from .models import DeviceCommand

class DeviceHandler(Handler):
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def handle(self, state: InvocationState, parts: list[str]) -> HandlerResult:
        # Use structured output to extract a DeviceCommand from text
        from pydantic import BaseModel
        class Cmd(DeviceCommand):
            pass
        try:
            cmd = self.llm.structured(
                prompt=f"Extract a devices command from: {state.text}",
                output_model=Cmd,
                query_hint="device_command",
                augment_source="devices",
                context=state.context,
            )
            # TODO: execute the command using your device client
            return HandlerResult(response=f"Executed {cmd.action} on {cmd.device_id}", status="devices")
        except Exception:
            return HandlerResult(response="Could not understand device command.", status="error")
```

#### Middleware (optional)

```python
from strukt.middleware import Middleware
from strukt.types import InvocationState

class DeviceAuthMiddleware(Middleware):
    def before_handle(self, state: InvocationState, query_type: str, parts: list[str]):
        if query_type == "devices":
            if not state.context.get("auth_token"):
                # deny by changing parts to an error sentinel or add a flag
                parts = ["UNAUTHORIZED"]
        return state, parts
```

#### Register

```python
from strukt import create, StruktConfig, HandlersConfig, MiddlewareConfig
from your_extension.handlers import DeviceHandler
from your_extension.middleware import DeviceAuthMiddleware

cfg = StruktConfig(
  handlers=HandlersConfig(
    registry={"devices": DeviceHandler},
    default_route="general",
  ),
  middleware=[MiddlewareConfig(DeviceAuthMiddleware)],
)
app = create(cfg)
```

Best practices:
- Validate structured outputs with Pydantic models that reflect your device API.
- Keep device-side effects idempotent; return clear status strings for telemetry.
- Use `query_hint` and `augment_source` to improve logging and memory quality.

## Advanced

### Query Hints

Pass `query_hint` to help memory retrieval and logging. It is provided when calling an LLM automatically or can be modified when invoking any LLM inheriting from the StruktX LLM classes.

```python
resp = app.invoke("recommend lunch", context={"user_id": "u1"})
# or: llm.invoke(prompt, query_hint="recommendation")
```

### augment_source

Provide `augment_source` when calling an LLM client to label memory injection source in logs.

```python
# Inside a handler
llm.invoke(prompt, context=state.context, query_hint=state.text, augment_source="recommendations")
```

### Context & Scoping

Use `user_id` and optionally `unit_id` in context for scoped memory retrieval. If a `KnowledgeStore` is enabled, StruktX may use it to further scope memory.

```python
# Handlers receive the full invocation state
def handle(self, state: InvocationState, parts: list[str]) -> HandlerResult:
    user_id = state.context.get("user_id")
    # enrich prompts or enforce auth
    ...
```

### Step-by-Step: Build Your First App

This guided tutorial creates two handlers **Time Handler** and **Weather Handler** adds a custom **Rate Limit Middleware** and enables **Memory Extraction**

#### Step 0: Setup

Create a virtual environment and install dependencies.

```bash
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
uv pip install struktX
```

#### Step 1: Weather Model

Typed outputs reduce parsing errors and improve reliability when extracting fields from natural language.

```python
from pydantic import BaseModel, Field

class WeatherQuery(BaseModel):
    city: str = Field(..., min_length=1)
    unit: str | None = Field(default="celsius", description="celsius or fahrenheit")
```

#### Step 2: Time Handler

Handlers encapsulate business logic per query type and receive the full `InvocationState`

```python
from strukt.interfaces import Handler, LLMClient
from strukt.types import InvocationState, HandlerResult
from datetime import datetime

class TimeHandler(Handler):
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def handle(self, state: InvocationState, parts: list[str]) -> HandlerResult:
        now = datetime.utcnow().strftime("%H:%M UTC")
        return HandlerResult(response=f"Current time: {now}", status="time")
```

#### Step 3: Weather Handler

Use `llm.structured` to reliably extract fields into `WeatherQuery`, then call a weather client.

```python
from strukt.interfaces import Handler, LLMClient
from strukt.types import InvocationState, HandlerResult
from step_01_models import WeatherQuery

def get_weather(city: str, unit: str | None = "celsius") -> str:
    unit_symbol = "°C" if (unit or "celsius").lower().startswith("c") else "°F"
    return f"22{unit_symbol} and clear in {city}"

class WeatherHandler(Handler):
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def handle(self, state: InvocationState, parts: list[str]) -> HandlerResult:
        q = self.llm.structured(
            prompt=f"Extract weather query fields from: {state.text}",
            output_model=WeatherQuery,
            query_hint="weather_query",
            augment_source="weather",
            context=state.context,
        )
        return HandlerResult(response=get_weather(q.city, q.unit), status="weather")
```

#### Step 4: Rate Limit Middleware

Centralize rate limiting so handlers remain clean. This example counts calls by `user_id` and returns an error sentinel when exceeded.

```python
from strukt.middleware import Middleware
from strukt.types import InvocationState

class RateLimitMiddleware(Middleware):
    def __init__(self, max_calls: int = 5):
        self.max_calls = max_calls
        self._counts: dict[str, int] = {}

    def before_handle(self, state: InvocationState, query_type: str, parts: list[str]):
        user_id = state.context.get("user_id", "anon")
        self._counts[user_id] = self._counts.get(user_id, 0) + 1
        if self._counts[user_id] > self.max_calls:
            return state, ["RATE_LIMITED"]
        return state, parts
```

#### Step 5: Enable Memory

Enable `MemoryExtractionMiddleware` and a vector store to persist durable facts. StruktX will auto-inject them via `MemoryAugmentedLLMClient` during LLM calls.

Use `MemoryConfig.use_store=True` and `MemoryConfig.augment_llm=True` to enable a `KnowledgeStore` to further scope memory.

**Note**: `MemoryConfig.use_store=True` is required for `MemoryExtractionMiddleware` to work.

```python
from strukt import MemoryConfig, MiddlewareConfig

memory = MemoryConfig(
  factory="strukt.memory:UpstashVectorMemoryEngine",
  params={"index_url": "...", "index_token": "...", "namespace": "demo"},
  use_store=True,
  augment_llm=True,
)

memory_mw = MiddlewareConfig("strukt.memory.middleware:MemoryExtractionMiddleware", params={"max_items": 5})
```

#### Step 6: Wire It Up

Register handlers under query types and compose middleware. Set the default route for unclassified requests.

```python
import os
from strukt import create, StruktConfig, HandlersConfig, LLMClientConfig, MiddlewareConfig
from step_02_time_handler import TimeHandler
from step_03_weather_handler import WeatherHandler
from step_04_rate_limit_middleware import RateLimitMiddleware
from step_05_memory import memory, memory_mw

os.environ.setdefault("OPENAI_API_KEY", "...")

cfg = StruktConfig(
  llm=LLMClientConfig("langchain_openai:ChatOpenAI", dict(model="gpt-4o-mini")),
  handlers=HandlersConfig(
    registry={
      "time": TimeHandler,
      "weather": WeatherHandler,
    },
    default_route="time",
  ),
  memory=memory,
  middleware=[MiddlewareConfig(RateLimitMiddleware, params=dict(max_calls=3)), memory_mw],
)

app = create(cfg)
```

#### Step 7: Run

Invoke with `user_id` so rate-limiting and memory are scoped per user.

```python
print(app.invoke("what's the time now?", context={"user_id": "u1"}).response)
print(app.invoke("weather in Paris in celsius", context={"user_id": "u1"}).response)

for _ in range(5):
    r = app.invoke("time please", context={"user_id": "u1"})
    print(r.response)
```

#### Why this structure?

- **Handlers**: isolate use cases; adding a new capability is additive.
- **Middleware**: keeps cross-cutting concerns reusable and testable.
- **Memory**: improves continuity and reduces repeated user input.
- **Typed Extraction**: increases determinism for downstream consumers.

### Reference (Overview)

- `strukt.create(config)`: builds the app with LLM, classifier, handlers, memory, middleware.
- `Strukt.invoke(text, context)` / `Strukt.ainvoke`: run requests.
- `StruktConfig`: top-level config dataclass; subconfigs: `LLMClientConfig`, `ClassifierConfig`, `HandlersConfig`, `MemoryConfig`, `MiddlewareConfig`.
- `interfaces.LLMClient`: `invoke`, `structured`.
- `interfaces.Classifier`: `classify` → `QueryClassification`.
- `interfaces.Handler`: `handle` → `HandlerResult`.
- `interfaces.MemoryEngine`: `add`, `get`, `get_scoped`, `remove`, `cleanup`.
- `defaults.MemoryAugmentedLLMClient`: auto-injects relevant memory into prompts; supports `augment_source` and `query_hint`.
- `logging.get_logger`, `LoggingMiddleware`: structured, Rich-powered console logging.
- `langchain_helpers.LangChainLLMClient`, `adapt_to_llm_client`, `create_structured_chain`.
- `utils.load_factory`, `utils.coerce_factory`: resolve factories from strings/callables/classes/instances.
- `types`: `InvocationState`, `QueryClassification`, `HandlerResult`, `StruktResponse`, `StruktQueryEnum`.

### Best Practices

- Prefer typed handlers with clear responsibilities.
- Keep middleware small and composable.
- Scope memory using user-specific / traceable context.

### FAQ

**Can I use non-LangChain LLMs?** Yes—implement `LLMClient` or provide an adapter.

**How do I add a new query type?** Implement a handler and register it in `HandlersConfig.registry` and include it in the classifier config.

**How is memory injected?** If `MemoryConfig.augment_llm=True`, `MemoryAugmentedLLMClient` retrieves relevant docs and prepends them to prompts.

## Extras

### MCP Server

#### Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "struktx": {
      "command": "npx",
      "args": ["-y", "mcp-remote", "https://struktx.vercel.app/api/mcp"]
    }
  }
}
```

#### Cursor

Add to `~/.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "struktx": {
      "url": "https://struktx.vercel.app/api/mcp"
    }
  }
}
```

#### Windsurf

Add to `~/.codeium/windsurf/mcp_config.json`:

```json
{
  "mcpServers": {
    "struktx": {
      "url": "https://struktx.vercel.app/api/mcp"
    }
  }
}
```