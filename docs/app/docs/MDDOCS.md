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
  classifier=ClassifierConfig("strukt.classifiers:LLMClassifier"),
  handlers=HandlersConfig(default_route="general"),
  memory=MemoryConfig(
    factory="strukt.memory:UpstashVectorMemoryEngine",
    params={"index_url": "...", "index_token": "...", "namespace": "app1"},
    use_store=True,
    augment_llm=True,
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
  classifier=ClassifierConfig(factory="strukt.classifiers:LLMClassifier"),
  handlers=HandlersConfig(
    registry={
      # "time_service": "your_pkg.handlers:TimeHandler",
    },
    default_route="general",
  ),
  memory=MemoryConfig(
    factory="strukt.memory:UpstashVectorMemoryEngine",
    params={"index_url": "...", "index_token": "...", "namespace": "app1"},
    use_store=True,
    augment_llm=True
  ),
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

### Available Middleware Types

StruktX provides several built-in middleware types for common use cases:

#### Background Task Middleware
Execute handlers in background threads for improved user experience:

```python
from strukt.middleware import BackgroundTaskMiddleware

middleware=[
    MiddlewareConfig(BackgroundTaskMiddleware, dict(
        max_workers=4,
        enable_background_for={"device_control"},
        action_based_background={
            "maintenance_or_helpdesk": {"create"},
        },
        custom_messages={
            "device_control": "Device control successful",
        },
    )),
]
```

#### Logging Middleware
Structured logging with Rich-powered console output:

```python
from strukt.middleware import LoggingMiddleware

middleware=[
    MiddlewareConfig(LoggingMiddleware, dict(verbose=True)),
]
```

#### Response Cleaner Middleware
Clean and format handler responses:

```python
from strukt.middleware import ResponseCleanerMiddleware

middleware=[
    MiddlewareConfig(ResponseCleanerMiddleware),
]
```

#### Memory Extraction Middleware
Automatically extract and store conversation facts:

```python
from strukt.memory.middleware import MemoryExtractionMiddleware

middleware=[
    MiddlewareConfig(MemoryExtractionMiddleware),
]
```

### Memory

Enable scoped memory and automatic prompt augmentation.

```python
from strukt import StruktConfig, MemoryConfig

cfg = StruktConfig(
  memory=MemoryConfig(
    factory="strukt.memory:UpstashVectorMemoryEngine",
    params={"index_url": "...", "index_token": "...", "namespace": "app1"},
    use_store=True,
    augment_llm=True,)
)
```

## Ecosystem

### MCP Server

Expose StruktX handlers as MCP tools over HTTP.

#### Configuration

```python
from strukt import StruktConfig

config = StruktConfig(
  mcp=dict(
    enabled=True,
    server_name="struktmcp",
    include_handlers=[
      "device_control","amenity_service","maintenance_or_helpdesk",
      "bill_service","event_service","weather_service","schedule_future_event"
    ],
    default_consent_policy="ask-once",
    tools={
      "device_control": [
        dict(
          name="device_list",
          description="List devices for a user/unit",
          method_name="mcp_list",
          parameters_schema={
            "type":"object",
            "properties":{"user_id":{"type":"string"},"unit_id":{"type":"string"}},
            "required":["user_id","unit_id"]
          },
        ),
        dict(
          name="device_execute",
          description="Execute device commands",
          method_name="mcp_execute",
          usage_prompt="Call device_list first. Use attributes.identifier as deviceId; follow provider rules.",
          parameters_schema={
            "type":"object",
            "properties":{
              "commands":{"type":"array"},
              "user_id":{"type":"string"},
              "unit_id":{"type":"string"}
            },
            "required":["commands","user_id","unit_id"]
          },
        ),
      ],
      # ... other handlers
    }
  )
)
```

- `usage_prompt` appends extra guidance to tool descriptions.
- Consent policy defaults to ask-once and persists via MemoryEngine.

#### Serve via FastAPI

```python
from fastapi import FastAPI
from strukt import create, build_fastapi_app

app = create(config)

# Create a new FastAPI app exposing /mcp
fastapi_app = build_fastapi_app(app, config)

# Or mount onto an existing FastAPI app under a prefix
existing = FastAPI()
build_fastapi_app(app, config, app=existing, prefix="/mcp")
```

#### Endpoints

```bash
# List tools (GET)
curl -H "x-api-key: dev-key" http://localhost:8000/mcp

# Call tool (implicit op=call_tool)
curl -X POST -H "Content-Type: application/json" -H "x-api-key: dev-key" \
  -d '{"tool_name":"amenity_check","args":{"user_id":"u1","unit_id":"UNIT","facility_name_query":"gym"}}' \
  http://localhost:8000/mcp

# Explicit list via POST
curl -X POST -H "Content-Type: application/json" -H "x-api-key: dev-key" \
  -d '{"op":"list_tools"}' http://localhost:8000/mcp
```

#### Extending Handlers for MCP

Add `mcp_*` methods to handlers for precise tool entrypoints, then reference them via `method_name` in the MCP config.

```python
from strukt.interfaces import Handler, LLMClient
from strukt.types import InvocationState, HandlerResult

class MyHandler(Handler):
    def __init__(self, llm: LLMClient, toolkit):
        self.llm = llm
        self.toolkit = toolkit

    # Strukt entrypoint
    def handle(self, state: InvocationState, parts: list[str]) -> HandlerResult:
        ...

    # MCP tool entrypoints (keyword-only args)
    def mcp_list(self, *, user_id: str, unit_id: str):
        return self.toolkit.list(user_id, unit_id)

    def mcp_create(self, *, user_id: str, unit_id: str, payload: dict):
        return self.toolkit.create(user_id=user_id, unit_id=unit_id, payload=payload)
```

Map the methods in the MCP config:

```python
mcp=dict(
  tools={
    "my_service": [
      dict(
        name="my_list",
        description="List items",
        method_name="mcp_list",
        parameters_schema={
          "type": "object",
          "properties": {"user_id": {"type":"string"}, "unit_id": {"type":"string"}},
          "required": ["user_id","unit_id"]
        },
      ),
      dict(
        name="my_create",
        description="Create item",
        method_name="mcp_create",
        parameters_schema={
          "type":"object",
          "properties": {"user_id": {"type":"string"}, "unit_id": {"type":"string"}, "payload": {"type":"object"}},
          "required": ["user_id","unit_id","payload"]
        },
      )
    ]
  }
)
```

You can also point `method_name` at toolkit methods using dotted paths (e.g., `"toolkit.book_amenity_data"`). Dotted resolution supports `toolkit.*` and `_toolkit.*`.

#### MCP Config Reference

- **enabled**: Enable the MCP integration.
- **server_name**: Identifier for the server (e.g. `"struktmcp"`).
- **include_handlers**: Handler keys to expose as tools.
- **auth_api_key.header_name**: Request header for API key (default `x-api-key`).
- **auth_api_key.env_var**: Env var that holds the API key (default `STRUKTX_MCP_API_KEY`).
- **default_consent_policy**: `always-ask` | `ask-once` | `always-allow` | `never-allow`.
- **tools**: Map of handler name → list of tool configs (MCPToolConfig):

```python
dict(
  name="tool_name",                    # required
  description="what it does",          # required
  parameters_schema={...},              # JSON Schema for args
  method_name="mcp_list",              # dotted path to callable on handler
  required_scopes=["scope:read"],      # optional OAuth scopes metadata
  consent_policy="ask-once",           # optional per-tool consent override
  usage_prompt="LLM guidance...",      # optional extra prompt appended to description
)
```

Consent decisions are persisted through your configured MemoryEngine when available.

### LangChain Helpers

Use `LangChainLLMClient` and `create_structured_chain` to generate typed outputs.

```python
from pydantic import BaseModel
from strukt.langchain_helpers import create_structured_chain

class Foo(BaseModel):
    query_types: list[str] = []
    confidences: list[float] = []
    parts: list[str] = []

# chain = create_structured_chain(llm_client=your_langchain_client, prompt template, output_model=Foo)
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
  classifier=ClassifierConfig("strukt.classifiers:LLMClassifier"),
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

### Augment Source

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

## Background Task Middleware

The Background Task Middleware provides advanced task management capabilities, allowing you to execute handlers in background threads and return immediate responses to users while processing continues asynchronously.

### Features

- **Immediate Response**: Return custom messages instantly while tasks run in background
- **Action-Based Execution**: Configure which specific actions within handlers should run in background
- **Task Tracking**: Monitor task progress, status, and results
- **Parallel Execution**: Run multiple handlers concurrently
- **Task Management**: Cancel, cleanup, and query background tasks

### Configuration

```python
from strukt import StruktConfig, MiddlewareConfig
from strukt.middleware import BackgroundTaskMiddleware

config = StruktConfig(
    # ... other config
    middleware=[
        MiddlewareConfig(BackgroundTaskMiddleware, dict(
            max_workers=6,  # Number of concurrent background tasks
            default_message="Your request is being processed.",
            
            # Always run these handlers in background
            enable_background_for={"device_control"},
            
            # Run specific actions in background for specific handlers
            action_based_background={
                "maintenance_or_helpdesk": {"create"},  # Only "create" action
                "some_handler": {"create", "update"},   # Multiple actions
            },
            
            # Custom messages for different handlers
            custom_messages={
                "device_control": "Device control successful",
                "maintenance_or_helpdesk": "I've created your helpdesk ticket. Someone will be in touch shortly.",
            },
            
            # Custom return query types for different handlers
            return_query_types={
                "device_control": "DEVICE_CONTROL_SUCCESS",
                "maintenance_or_helpdesk": "HELPDESK_TICKET_CREATED",
            },
        )),
    ],
)
```

### Action-Based Background Execution

The middleware can intelligently determine when to run tasks in background based on the specific action being performed:

### Custom Return Query Types

The middleware supports custom return query types, allowing handlers to specify what query type should be returned in the response instead of the generic "background_task_created:..." format. This is useful for maintaining consistent API responses and providing meaningful status information to clients.

#### Configuration

```python
# Configure custom return query types
return_query_types={
    "device_control": "DEVICE_CONTROL_SUCCESS",
    "maintenance_or_helpdesk": "HELPDESK_TICKET_CREATED",
}
```

#### Handler Integration

Handlers can also specify return query types dynamically by setting them in the context. Since multiple handlers may run simultaneously, use a dictionary format where the key is the handler name and the value is the return query type:

```python
class MyHandler(Handler):
    def handle(self, state: InvocationState, parts: List[str]) -> HandlerResult:
        # Set custom return query type for this specific request
        # Use dictionary format to support multiple handlers
        state.context['return_query_types'] = {
            'my_handler_name': "CUSTOM_SUCCESS_STATUS"
        }
        
        # ... rest of handler logic
```

**Note**: For backward compatibility, the old single `return_query_type` format is still supported, but the dictionary format is recommended when multiple handlers are involved.

#### Multiple Handler Example

When multiple handlers are involved in a single request, each can specify its own return query type:

```python
class DeviceHandler(Handler):
    def handle(self, state: InvocationState, parts: List[str]) -> HandlerResult:
        # Set return query type for this handler
        if 'return_query_types' not in state.context:
            state.context['return_query_types'] = {}
        state.context['return_query_types']['device_control'] = "DEVICE_CONTROL_SUCCESS"
        
        # ... handler logic
        return HandlerResult(response="Device control initiated", status="device_control")

class NotificationHandler(Handler):
    def handle(self, state: InvocationState, parts: List[str]) -> HandlerResult:
        # Set return query type for this handler
        if 'return_query_types' not in state.context:
            state.context['return_query_types'] = {}
        state.context['return_query_types']['notification'] = "NOTIFICATION_SENT"
        
        # ... handler logic
        return HandlerResult(response="Notification sent", status="notification")
```

This approach ensures that each handler can specify its own meaningful return query type while maintaining backward compatibility.

#### Response Format

With custom return query types, the response will look like:

```json
{
    "response": "Device control successful",
    "background_tasks": [],
    "query_type": "DEVICE_CONTROL_SUCCESS",
    "query_types": [
        "DEVICE_CONTROL_SUCCESS"
    ],
    "transcript_parts": [
        "Turn on the kitchen AC",
        "Set temperature to 25 degrees"
    ]
}
```

Instead of the generic format:

```json
{
    "response": "Device control successful",
    "background_tasks": [],
    "query_type": "background_task_created:9541dc3f-cf4b-4257-a3e6-9d08ca77f702",
    "query_types": [
        "background_task_created:9541dc3f-cf4b-4257-a3e6-9d08ca77f702"
    ],
    "transcript_parts": [
        "Turn on the kitchen AC",
        "Set temperature to 25 degrees"
    ]
}
```

```python
# Configuration
action_based_background={
    "maintenance_or_helpdesk": {"create"},  # Only run "create" in background
}

# Behavior Examples:
"create a ticket for broken AC" → Runs in background, returns immediately
"check my ticket status" → Runs normally, waits for completion
```

### Handler Intents System

The background task middleware uses a handler intents system to determine which actions should run in background. Handlers extract intents and store them in the context for the middleware to use.

#### How Handler Intents Work

1. **Intent Extraction**: Handlers extract the user's intent/action from the input
2. **Context Storage**: Store the intent in `state.context['handler_intents'][handler_name] = action`
3. **Middleware Decision**: Background task middleware checks if the action matches configured background rules
4. **Background Execution**: If matched, the task runs in background with immediate response

#### Handler Integration

To enable action-based background execution, handlers should set the extracted action in the context:

```python
class MyHandler(Handler):
    def handle(self, state: InvocationState, parts: List[str]) -> HandlerResult:
        # Extract intent/action from user input
        intent = self._extract_intent(parts)
        
        # Set the action in context for middleware to use
        # The handler_intents dict is automatically initialized
        state.context['handler_intents']['my_handler_name'] = intent.action
        
        # Handle based on action
        if intent.action == "create":
            return self._handle_create()
        elif intent.action == "status":
            return self._handle_status()
```

#### Example: Helpdesk Handler

```python
class HelpdeskHandler(Handler):
    def handle(self, state: InvocationState, parts: List[str]) -> HandlerResult:
        # Extract intent from user input using LLM
        intent = self._extract_intent(full_request, user_id, unit_id)
        
        # Set the extracted action in context for middleware to use
        state.context['handler_intents']['maintenance_or_helpdesk'] = intent.action
        
        # Handle based on action
        if intent.action == "create":
            return self._handle_create_ticket(intent, user_id, unit_id)
        elif intent.action == "status":
            return self._handle_status_check(intent, user_id, unit_id)
```

#### Configuration Mapping

The middleware configuration maps to handler intents:

```python
# Configuration
action_based_background={
    "maintenance_or_helpdesk": {"create", "update"},  # These actions run in background
    "device_control": {"turn_on", "turn_off"},        # These actions run in background
}

# Handler sets intent
state.context['handler_intents']['maintenance_or_helpdesk'] = "create"  # ✅ Runs in background
state.context['handler_intents']['maintenance_or_helpdesk'] = "status"  # ❌ Runs in foreground
```

### Task Management API

The middleware provides comprehensive task management through the main `Strukt` instance:

```python
# Get all background tasks
all_tasks = app.get_all_background_tasks()

# Get tasks by status
running_tasks = app.get_running_background_tasks()
completed_tasks = app.get_completed_background_tasks()
failed_tasks = app.get_failed_background_tasks()

# Get specific task info
task_info = app.get_background_task_info("task-id-123")

# Get tasks filtered by status
tasks = app.get_background_tasks_by_status("running")
```

### Task Information Structure

Each task provides detailed information:

```python
{
    'task_id': 'uuid-string',
    'handler_name': 'device_control',
    'handler_id': 'device_control',
    'status': 'running',  # pending, running, completed, failed, cancelled
    'progress': 0.75,     # 0.0 to 1.0
    'created_at': '2024-01-01T12:00:00',
    'started_at': '2024-01-01T12:00:01',
    'completed_at': None,  # Set when task completes
    'result': {...},       # HandlerResult when completed
    'error': None,         # Error message if failed
    'metadata': {...}      # Additional task metadata
}
```

### Complete Example

```python
from strukt import create, StruktConfig, MiddlewareConfig
from strukt.middleware import BackgroundTaskMiddleware

# Configure with action-based background execution
config = StruktConfig(
    # ... other config
    middleware=[
        MiddlewareConfig(BackgroundTaskMiddleware, dict(
            max_workers=4,
            default_message="Processing your request...",
            enable_background_for={"device_control"},
            action_based_background={
                "maintenance_or_helpdesk": {"create"},
            },
            custom_messages={
                "device_control": "Device control successful",
                "maintenance_or_helpdesk": "Ticket created successfully. You'll receive a confirmation shortly.",
            },
        )),
    ],
)

app = create(config)

# Execute requests
result = app.invoke("turn on the bedroom lights", context={"user_id": "user1"})
print(result.response)  # "Device control successful" (immediate)

# Check background tasks
running = app.get_running_background_tasks()
for task in running:
    print(f"Task {task['task_id'][:8]}... is {task['progress']*100:.1f}% complete")
```

### Best Practices

1. **Use Action-Based Execution**: Configure specific actions rather than entire handlers when possible
2. **Provide Clear Messages**: Give users immediate feedback about what's happening
3. **Monitor Task Progress**: Use the task management API to track long-running operations
4. **Handle Failures**: Check for failed tasks and provide appropriate error handling
5. **Clean Up**: The middleware automatically cleans up old tasks, but you can customize the retention period

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
- `types`: `InvocationState`, `QueryClassification`, `HandlerResult`, `StruktResponse`, `StruktQueryEnum`, `BackgroundTaskInfo`.
- `middleware.BackgroundTaskMiddleware`: Background task management with action-based execution.
- `Strukt.get_all_background_tasks()`, `Strukt.get_running_background_tasks()`, `Strukt.get_background_task_info()`: Task management methods.

### Best Practices

- Prefer typed handlers with clear responsibilities.
- Keep middleware small and composable.
- Scope memory using user-specific / traceable context.

### FAQ

**Can I use non-LangChain LLMs?** Yes—implement `LLMClient` or provide an adapter.

**How do I add a new query type?** Implement a handler and register it in `HandlersConfig.registry` and include it in the classifier config.

**How is memory injected?** If `MemoryConfig.augment_llm=True`, `MemoryAugmentedLLMClient` retrieves relevant docs and prepends them to prompts.

**When should I use background tasks?** Use background tasks for operations that take time (device control, ticket creation) while keeping quick operations (status checks, queries) synchronous for immediate responses.

**How do I configure action-based background execution?** Use the `action_based_background` parameter to specify which actions within handlers should run in background, and ensure your handler sets `state.context['handler_intents'][handler_name] = action`.

**Can I monitor background task progress?** Yes, use `app.get_all_background_tasks()`, `app.get_running_background_tasks()`, and other task management methods to monitor progress and status.

**How do handler intents work?** Handlers extract intents and store them in `state.context['handler_intents'][handler_name] = action`. The background task middleware uses these intents to determine if a task should run in background based on the configured `action_based_background` rules.

## Extras

### MCP Server

#### Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "struktx": {
      "command": "npx",
      "args": ["-y", "mcp-remote", "https://struktx.snowheap.ai/api/mcp"]
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
      "url": "https://struktx.snowheap.ai/api/mcp"
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
      "url": "https://struktx.snowheap.ai/api/mcp"
    }
  }
}
```