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

#### Auto-discovery (recommended)

Handlers that define `mcp_*` methods are automatically exposed as MCP tools. Tool names are generated as `<handler_key>_<method_suffix>`, descriptions default to the method docstring, and the input schema is inferred from type hints (including `Optional`, `list`, `dict`).

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
    # Optional overlays (no method_name): tweak descriptions/prompts for auto tools
    tools={
      "device_control": [
        dict(
          name="device_control_execute",  # auto from handler "device_control" + mcp_execute
          description="Execute device commands",
          usage_prompt="Call device_control_list first. Use attributes.identifier as deviceId; see provider rules.",
        )
      ]
    }
  )
)
```

> Tip: Prefer docstrings on your `mcp_*` methods to provide LLM-facing descriptions. Use overlay entries (only `name`, `description`, `usage_prompt`) to override text while keeping schema/dispatch auto-generated.

#### Explicit tool mapping (advanced)

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

Add `mcp_*` methods to handlers for precise tool entrypoints. Descriptions default to the method docstring; input schemas are inferred from type hints.

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
        """List items for a given user and unit."""
        return self.toolkit.list(user_id, unit_id)

    def mcp_create(self, *, user_id: str, unit_id: str, payload: dict):
        """Create an item with the provided payload."""
        return self.toolkit.create(user_id=user_id, unit_id=unit_id, payload=payload)
```

Map methods explicitly if needed, or use overlays to override only text:

```python
mcp=dict(
  tools={
    "my_service": [
      dict(
        name="my_service_list",             # auto from handler "my_service" + mcp_list
        description="Custom list description",  # override docstring only
        # no method_name → overlay only
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
  description="what it does",          # optional for overlays; defaults to docstring
  parameters_schema={...},              # optional; inferred from type hints when omitted
  method_name="mcp_list",              # optional; dotted path for explicit mapping
  required_scopes=["scope:read"],      # optional OAuth scopes metadata
  consent_policy="ask-once",           # optional per-tool consent override
  usage_prompt="LLM guidance...",      # optional extra prompt appended to description
)
```

Consent decisions are persisted through your configured MemoryEngine when available.

### FastMCP v2 Integration

StruktX now includes a modern FastMCP v2 integration that provides improved performance, better compliance with the MCP specification, and enhanced maintainability. This is the recommended approach for new projects.

#### Key Features

- **FastMCP v2 Framework**: Built on the latest FastMCP library for better performance and compliance
- **Convention-based Discovery**: Automatically discovers `mcp_*` methods on handlers
- **Multiple Transports**: Support for stdio, SSE, and HTTP transports
- **Enhanced Authentication**: Improved API key authentication with configurable headers
- **Middleware Integration**: Seamless integration with StruktX middleware system
- **Async Support**: Full async/await support for better performance

#### Quick Start

Enable FastMCP v2 in your configuration:

```python
from strukt import StruktConfig, MCPv2Config, MCPAuthAPIKeyConfig

config = StruktConfig(
    # ... other config ...
    mcp_v2=MCPv2Config(
        enabled=True,
        server_name="my-struktx-mcp",
        transport="http",  # stdio, sse, or http
        include_handlers=[
            "time_query",
            "weather_query", 
            "web_search_query",
            "device_control_query"
        ],
        auth_api_key=MCPAuthAPIKeyConfig(
            enabled=True,
            header_name="x-api-key",
            env_var="STRUKTX_MCP_API_KEY"
        ),
        # HTTP transport settings
        http_host="localhost",
        http_port=8000
    )
)
```

#### Handler Integration

Handlers with `mcp_*` methods are automatically discovered and exposed as MCP tools:

```python
from strukt.interfaces import Handler
from typing import Dict, Any

class WeatherHandler(Handler):
    def __init__(self, toolkit, llm_client):
        self.toolkit = toolkit
        self.llm_client = llm_client

    # Regular StruktX handler method
    def handle(self, state, parts):
        # ... handler logic ...
        pass

    # MCP tool methods (automatically discovered)
    async def mcp_current(self, *, location: str) -> Dict[str, Any]:
        """Get current weather data for a location."""
        return await self.toolkit.get_current_weather_data(location)

    async def mcp_forecast(self, *, location: str, days: int = 5) -> Dict[str, Any]:
        """Get weather forecast for a location."""
        return await self.toolkit.get_forecast_data(location, days=days)
```

#### Transport Options

**HTTP Transport (Recommended for Production)**

```python
config = StruktConfig(
    mcp_v2=MCPv2Config(
        enabled=True,
        transport="http",
        http_host="0.0.0.0",
        http_port=8000,
        auth_api_key=MCPAuthAPIKeyConfig(enabled=True)
    )
)

# Start the server
from strukt.mcp_v2 import run_http
run_http(config)
```

**SSE Transport (Server-Sent Events)**

```python
config = StruktConfig(
    mcp_v2=MCPv2Config(
        enabled=True,
        transport="sse",
        sse_host="localhost",
        sse_port=8000
    )
)

from strukt.mcp_v2 import run_sse
run_sse(config)
```

**Stdio Transport (CLI Integration)**

```python
config = StruktConfig(
    mcp_v2=MCPv2Config(
        enabled=True,
        transport="stdio"
    )
)

from strukt.mcp_v2 import run_stdio
run_stdio(config)
```

#### FastAPI Integration

Mount FastMCP v2 on an existing FastAPI application:

```python
from fastapi import FastAPI
from strukt import create, StruktConfig
from strukt.mcp_v2 import build_fastapi_app

# Create your StruktX app
config = StruktConfig(
    mcp_v2=MCPv2Config(
        enabled=True,
        transport="http",
        auth_api_key=MCPAuthAPIKeyConfig(enabled=True)
    )
)
app = create(config)

# Create FastAPI app with MCP endpoints
fastapi_app = build_fastapi_app(app, config)

# Or mount on existing FastAPI app
existing_app = FastAPI()
build_fastapi_app(app, config, app=existing_app, prefix="/mcp")
```

#### API Endpoints

**List Available Tools (GET)**
```bash
curl -H "x-api-key: your-api-key" http://localhost:8000/v1/mcp
```

**Execute Tool (POST)**
```bash
curl -X POST "http://localhost:8000/v1/mcp" \
  -H "Content-Type: application/json" \
  -H "x-api-key: your-api-key" \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/call",
    "params": {
      "name": "weather_query_current",
      "arguments": {
        "location": "New York"
      }
    }
  }'
```

#### Configuration Reference

**MCPv2Config**

- `enabled`: Enable FastMCP v2 integration (default: `False`)
- `server_name`: Server identifier (default: `None`)
- `include_handlers`: List of handler keys to expose (default: `[]` - all handlers)
- `transport`: Transport type - `"stdio"`, `"sse"`, or `"http"` (default: `"stdio"`)
- `sse_host`: Host for SSE transport (default: `"localhost"`)
- `sse_port`: Port for SSE transport (default: `8000`)
- `http_host`: Host for HTTP transport (default: `"localhost"`)
- `http_port`: Port for HTTP transport (default: `8000`)
- `auth_api_key`: API key authentication configuration

**MCPAuthAPIKeyConfig**

- `enabled`: Enable API key authentication (default: `False`)
- `header_name`: Request header name for API key (default: `"x-api-key"`)
- `env_var`: Environment variable containing the API key (default: `"STRUKTX_MCP_API_KEY"`)

#### Migration from MCP v1

To migrate from the legacy MCP v1 integration:

1. **Update Configuration**: Replace `mcp` config with `mcp_v2` config
2. **Update Imports**: Change imports from `strukt.mcp` to `strukt.mcp_v2`
3. **Handler Methods**: Ensure `mcp_*` methods are `async` if they perform async operations
4. **API Endpoints**: Update endpoint paths from `/mcp` to `/v1/mcp`
5. **Authentication**: Update header names if using custom authentication

**Before (MCP v1):**
```python
config = StruktConfig(
    mcp=dict(
        enabled=True,
        server_name="my-mcp",
        include_handlers=["weather", "time"]
    )
)
```

**After (FastMCP v2):**
```python
config = StruktConfig(
    mcp_v2=MCPv2Config(
        enabled=True,
        server_name="my-mcp",
        transport="http",
        include_handlers=["weather", "time"],
        auth_api_key=MCPAuthAPIKeyConfig(enabled=True)
    )
)
```

#### AWS Fargate Deployment

For production deployment on AWS Fargate:

```python
import os

config = StruktConfig(
    mcp_v2=MCPv2Config(
        enabled=True,
        server_name="struktx-mcp",
        transport="http",
        include_handlers=[
            "time_query",
            "weather_query",
            "web_search_query",
            "device_control_query"
        ],
        auth_api_key=MCPAuthAPIKeyConfig(
            enabled=True,
            header_name="x-api-key",
            env_var="STRUKTX_MCP_API_KEY"
        ),
        # AWS Fargate configuration
        http_host=os.getenv("MCP_BASE_URL", "https://your-domain.com").replace("https://", "").replace("http://", ""),
        http_port=443  # HTTPS port
    )
)
```

Environment variables for AWS Fargate:
- `MCP_BASE_URL`: Your Fargate service URL (e.g., `https://ai-stage-ae.roomi-services.com`)
- `MCP_TRANSPORT`: Transport type (e.g., `http`)
- `STRUKTX_MCP_API_KEY`: Your API key for authentication

#### Best Practices

1. **Use HTTP Transport**: For production deployments, use HTTP transport for better reliability
2. **Enable Authentication**: Always enable API key authentication in production
3. **Async Methods**: Make `mcp_*` methods `async` if they perform async operations
4. **Error Handling**: Implement proper error handling in your `mcp_*` methods
5. **Documentation**: Use descriptive docstrings for your `mcp_*` methods as they become tool descriptions
6. **Testing**: Test your MCP tools thoroughly before deploying to production

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

### Async Performance Optimizations

StruktX includes comprehensive async/await optimizations that make it the fastest NLP → action workflow framework. These optimizations provide true concurrency, automatic sync/async handler compatibility, and advanced performance monitoring.

#### Key Features

- **True Async Concurrency**: Up to 15 handlers can run in parallel using `asyncio.gather`
- **Automatic Handler Compatibility**: Sync and async handlers work seamlessly together
- **Performance Monitoring**: Real-time metrics for execution times, success rates, and P95 latency
- **Rate Limiting**: Configurable concurrency control with `asyncio.Semaphore`
- **LLM Optimizations**: Streaming, batching, and caching for improved throughput
- **Circuit Breaker Pattern**: Fault tolerance and cascading failure prevention

#### Configuration

Enable async optimizations in your StruktX configuration:

```python
from strukt import StruktConfig, EngineOptimizationsConfig

config = StruktConfig(
    # ... other config
    optimizations=EngineOptimizationsConfig(
        enable_performance_monitoring=True,
        max_concurrent_handlers=15,
        enable_llm_streaming=False,  # Disabled for LangChain compatibility
        enable_llm_batching=True,
        enable_llm_caching=True,
        llm_batch_size=10,
        llm_cache_size=1000,
        llm_cache_ttl=3600,
    )
)
```

#### Async Invocation

Use the async API for maximum performance:

```python
# Async invocation with full optimizations
result = await app.ainvoke(
    "turn on the kitchen AC and tell me the weather in Tokyo",
    context={"user_id": "user1", "unit_id": "unit1"}
)

# Access performance metrics
metrics = app._engine._performance_monitor.get_metrics()
print(f"Average handler time: {metrics['handler_time_query'].average_duration_ms}ms")
```

#### Handler Compatibility

Handlers can implement either sync or async methods - StruktX automatically bridges them:

```python
# Sync handler (automatically runs in thread pool when called via ainvoke)
class TimeHandler(Handler):
    def handle(self, state: InvocationState, parts: list[str]) -> HandlerResult:
        return HandlerResult(response=f"Current time: {datetime.now()}", status="time")

# Async handler (runs natively with true concurrency)
class WeatherHandler(Handler):
    async def ahandle(self, state: InvocationState, parts: list[str]) -> HandlerResult:
        weather_data = await weather_api.get_weather(parts[0])
        return HandlerResult(response=weather_data, status="weather")

# Hybrid handler (implements both for optimal performance)
class DeviceHandler(Handler):
    def handle(self, state: InvocationState, parts: list[str]) -> HandlerResult:
        # Sync implementation for quick operations
        return self._process_sync(state, parts)
    
    async def ahandle(self, state: InvocationState, parts: list[str]) -> HandlerResult:
        # Async implementation for I/O operations
        return await self._process_async(state, parts)
```

#### Performance Monitoring

Access real-time performance metrics:

```python
# Get performance metrics endpoint (if using FastAPI)
@app.get("/metrics")
async def get_performance_metrics():
    monitor = app._engine._performance_monitor
    return {
        "metrics": {
            operation: {
                "count": metrics.count,
                "average_duration_ms": metrics.average_duration * 1000,
                "p95_latency_ms": metrics.p95_latency * 1000,
                "success_rate": metrics.success_rate,
            }
            for operation, metrics in monitor._metrics.items()
        },
        "rate_limiter": {
            "active_requests": app._engine._rate_limiter._active_count,
            "max_concurrent": app._engine._rate_limiter.max_concurrent,
        }
    }
```

#### Pydantic Response Preservation

StruktX intelligently preserves structured Pydantic responses when multiple handlers execute together:

```python
# Single handler - returns full Pydantic object
result = await app.ainvoke("show me available facilities")
# result.response = {"status": "success", "available_facilities": [...], "current_date": "..."}

# Multiple handlers - preserves all structured data
result = await app.ainvoke("show facilities and weather in Tokyo")
# result.response = [
#   {"status": "success", "available_facilities": [...], "current_date": "..."},
#   {"status": "success", "temperature": 22.8, "conditions": "broken clouds", "location": "Tokyo"}
# ]

# Mixed responses - falls back to string concatenation
result = await app.ainvoke("turn on AC and tell me the time")
# result.response = "Device control successful. Current time is 2:39 PM"
```

#### Migration Guide

**For existing applications:**

1. **No code changes required** - existing sync handlers work automatically
2. **Optional async migration** - gradually convert handlers to async for better performance
3. **Enable optimizations** - add `EngineOptimizationsConfig` to your config
4. **Use async API** - replace `app.invoke()` with `await app.ainvoke()`

**Performance improvements:**
- **3x faster** concurrent execution for async handlers
- **Automatic compatibility** between sync and async handlers
- **Real-time monitoring** of performance metrics
- **Intelligent response handling** preserves structured data

### LLM Retry Mechanism

StruktX includes built-in retry functionality for LLM calls to handle transient failures and improve reliability. The retry mechanism supports both synchronous and asynchronous LLM operations with configurable backoff strategies.

#### Configuration

Enable retry functionality in your LLM client configuration:

```python
from strukt import StruktConfig, LLMClientConfig

config = StruktConfig(
    llm=LLMClientConfig(
        factory="langchain_openai:ChatOpenAI",
        params=dict(
            model="gpt-4o-mini",
            api_key="your-api-key",
            base_url="https://api.openai.com/v1"
        ),
        retry={
            "max_retries": 3,
            "base_delay": 1.0,
            "max_delay": 30.0,
            "exponential_base": 2.0,
            "jitter": True,
            "retryable_exceptions": (Exception,),  # Retry on any exception
        }
    )
)
```

#### Retry Parameters

- **max_retries**: Maximum number of retry attempts (default: 3)
- **base_delay**: Initial delay between retries in seconds (default: 1.0)
- **max_delay**: Maximum delay between retries in seconds (default: 30.0)
- **exponential_base**: Base for exponential backoff (default: 2.0)
- **jitter**: Add random jitter to prevent thundering herd (default: True)
- **retryable_exceptions**: Tuple of exception types to retry on (default: (Exception,))

#### Supported Operations

The retry mechanism automatically applies to all LLM operations:

- `invoke()` - Text generation
- `structured()` - Structured output generation
- `ainvoke()` - Async text generation
- `astructured()` - Async structured output generation

#### Example Usage

```python
from strukt import create, StruktConfig, LLMClientConfig

# Configure with retry
config = StruktConfig(
    llm=LLMClientConfig(
        factory="langchain_openai:ChatOpenAI",
        params=dict(model="gpt-4o-mini"),
        retry={
            "max_retries": 2,
            "base_delay": 0.5,
            "max_delay": 10.0,
            "jitter": True
        }
    )
)

app = create(config)

# All LLM calls will automatically retry on failure
result = app.invoke("Hello, world!")
```

### Intent Caching

StruktX provides intelligent caching for handler results based on semantic similarity of user queries. This reduces redundant processing and improves response times for similar requests.

#### Features

- **Semantic Matching**: Cache entries are matched based on meaning, not exact text
- **Fast Track Caching**: Immediate in-memory caching for exact matches
- **Configurable TTL**: Time-to-live settings per handler type
- **Scoped Caching**: Cache entries can be scoped by user, unit, or globally
- **Pretty Logging**: Rich console output for cache hits, misses, and stores

#### Configuration

Enable intent caching in your StruktX configuration:

```python
from strukt import StruktConfig, HandlersConfig
from strukt.memory import InMemoryIntentCacheEngine, IntentCacheConfig, HandlerCacheConfig, CacheStrategy, CacheScope

# Create intent cache configuration
intent_cache_config = IntentCacheConfig(
    enabled=True,
    default_strategy=CacheStrategy.SEMANTIC,
    default_ttl_seconds=3600,
    similarity_threshold=0.7,
    max_entries_per_handler=1000,
    handler_configs={
        "WeatherHandler": HandlerCacheConfig(
            handler_name="WeatherHandler",
            strategy=CacheStrategy.SEMANTIC,
            scope=CacheScope.USER,
            ttl_seconds=1800,
            max_entries=500,
            similarity_threshold=0.6,
            enable_fast_track=True
        ),
        "DeviceHandler": HandlerCacheConfig(
            handler_name="DeviceHandler",
            strategy=CacheStrategy.SEMANTIC,
            scope=CacheScope.USER,
            ttl_seconds=1800,
            max_entries=500,
            similarity_threshold=0.8,
            enable_fast_track=True
        )
    }
)

# Create cache engine
intent_cache_engine = InMemoryIntentCacheEngine(intent_cache_config)

# Configure handlers with caching
config = StruktConfig(
    handlers=HandlersConfig(
        registry={
            "weather_service": CachedWeatherHandler,
            "device_control": CachedDeviceHandler,
        },
        handler_params={
            "weather_service": dict(intent_cache_engine=intent_cache_engine),
            "device_control": dict(intent_cache_engine=intent_cache_engine),
        }
    )
)
```

#### Cache Strategies

- **EXACT**: Match exact text queries
- **SEMANTIC**: Match based on semantic similarity
- **FUZZY**: Match with fuzzy string matching
- **HYBRID**: Combine multiple strategies

#### Cache Scopes

- **GLOBAL**: Cache entries visible to all users
- **USER**: Cache entries scoped to specific users
- **UNIT**: Cache entries scoped to specific units
- **SESSION**: Cache entries scoped to specific sessions

#### Cache Management

```python
# Get cache statistics
stats = intent_cache_engine.get_stats()
print(f"Hit rate: {stats.hit_rate:.2%}")

# Clean up expired entries
cleanup_stats = intent_cache_engine.cleanup()
print(f"Removed {cleanup_stats.expired_entries} expired entries")

# Clear all cache entries
intent_cache_engine.clear()
```

#### Cache Management API Endpoints

When using StruktX with FastAPI, you can expose cache management endpoints:

```python
from fastapi import FastAPI
from strukt import build_fastapi_app

app = FastAPI()
strukt_app = create(config)
build_fastapi_app(strukt_app, config, app=app)

# Cache management endpoints are automatically available:
# GET /cache/stats - Get cache statistics
# POST /cache/cleanup - Clean up expired entries
# DELETE /cache/clear - Clear all cache entries
```

Example usage with curl:

```bash
# Get cache statistics
curl -H "x-api-key: dev-key" http://localhost:8000/cache/stats

# Clean up expired entries
curl -X POST -H "x-api-key: dev-key" http://localhost:8000/cache/cleanup

# Clear all cache entries
curl -X DELETE -H "x-api-key: dev-key" http://localhost:8000/cache/clear
```

#### Pretty Logging

The caching system provides rich console output for cache operations:

```
╭───────────────── 📦 JSON: Cache Hit Result ─────────────────╮
│ {                                                           │
│   "cache_hit": true,                                        │
│   "handler_name": "CachedWeatherHandler",                   │
│   "similarity": 0.95,                                       │
│   "match_type": "semantic",                                 │
│   "key": "weather:dubai:what is the weat..."                │
│ }                                                           │
╰─────────────────────────────────────────────────────────────╯
```

#### Multi-Request Handling

The caching system properly handles multi-request transcripts by caching each individual request component separately:

```python
# This multi-request will cache each component individually
result = app.invoke(
    "turn off the kitchen AC and tell me the weather in Dubai",
    context={"user_id": "user1", "unit_id": "unit1"}
)

# Each component (device control, weather) is cached separately
# Subsequent similar requests will hit the cache for individual components
```

#### Creating Cached Handlers

To create a cached version of your handler:

```python
from strukt.memory import CacheAwareHandler
from strukt.types import InvocationState, HandlerResult

class CachedMyHandler(CacheAwareHandler, MyHandler):
    """My handler with intent caching support."""
    
    def __init__(self, *args, intent_cache_engine=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.intent_cache_engine = intent_cache_engine
        self._cache_config = None
        self._cache_data_type = MyCacheData
    
    def get_cache_config(self) -> Optional[HandlerCacheConfig]:
        return self._cache_config
    
    def should_cache(self, state: InvocationState) -> bool:
        return True  # Cache all requests
    
    def build_cache_key(self, state: InvocationState) -> str:
        return f"my_handler:{state.text}:{state.context.get('user_id', '')}"
    
    def extract_cache_data(self, result: HandlerResult) -> Dict[str, Any]:
        return {"response": result.response, "status": result.status}
    
    def apply_cached_data(self, cached_data: Dict[str, Any]) -> HandlerResult:
        return HandlerResult(
            response=cached_data["response"],
            status=cached_data["status"]
        )
```

### Weave Logging Integration

StruktX includes comprehensive Weave and OpenTelemetry integration for detailed operation tracking, performance monitoring, and debugging. The system creates a unified trace tree where all operations, including background tasks and parallel execution, are nested under a single root trace for complete visibility.

#### Key Features

- **Unified Trace Tree**: All operations appear as a single, deeply nested trace with no separate top-level entries
- **Custom Trace Naming**: Configurable trace names in format `userID-unitID-threadID-timestamp`
- **Background Task Nesting**: Background tasks execute within the original trace context, even across threads
- **OpenTelemetry Export**: Exclusively exports to Weave's OTLP endpoint for unified observability
- **Auto-Instrumentation**: Automatically instruments OpenAI SDK calls and StruktX components
- **Component Label Customization**: Configure display names (e.g., change "Engine.run" to "StruktX.run")

#### Configuration

Enable Weave logging in your StruktX configuration:

```python
from strukt import StruktConfig, WeaveConfig, TracingConfig, OpenTelemetryConfig

config = StruktConfig(
    # ... other config
    weave=WeaveConfig(
        enabled=True,
        project_name="my-ai-app",  # Or use PROJECT_NAME env var
        environment="development",  # Or use CURRENT_ENV env var
        api_key_env="WANDB_API_KEY"  # Environment variable for Weave API key
    ),
    tracing=TracingConfig(
        component_label="StruktX",  # Customize component name (default: "Engine")
        collapse_status_ops=True,   # Collapse status operations into attributes
        enable_middleware_tracing=False  # Optional middleware tracing
    ),
    opentelemetry=OpenTelemetryConfig(
        enabled=True,
        project_id="my-project",
        api_key_env="WANDB_API_KEY",
        use_openai_instrumentation=True  # Auto-instrument OpenAI SDK calls
    )
)
```

#### Environment Variables

Set these environment variables for Weave integration:

```bash
export WANDB_API_KEY="your-weave-api-key"
export PROJECT_NAME="my-project"        # Optional, defaults to "struktx"
export CURRENT_ENV="production"         # Optional, defaults to "development"
```

#### Unified Trace Architecture

The system creates a single root trace session that contains all operations:

```
StruktX.run(user123) [userID-unitID-threadID-timestamp]
├── StruktX.Engine.classify
├── StruktX.Engine._execute_grouped_handlers
│   ├── StruktX.Handler.handle (parallel)
│   └── BackgroundTask.device_control
│       └── StruktX.Handler.handle (background thread)
├── StruktX.LLMClient.invoke
└── StruktX.Engine.log_post_run_evaluation
```

#### Automatic Operation Tracking

When enabled, Weave automatically tracks:

- **Root Session**: Custom-named trace containing all operations (e.g., `user123-unit456-thread789-1234567890`)
- **Engine Operations**: `Engine.run`, `classify`, `execute_grouped_handlers`, `execute_handlers_parallel`
- **LLM Operations**: All OpenAI SDK calls via auto-instrumentation, plus StruktX LLM client calls
- **Handler Operations**: Individual handler executions with inputs/outputs captured
- **Background Tasks**: `BackgroundTask.execute` nested within the original trace, including results
- **Memory Operations**: Memory retrieval and injection with scoped context
- **Performance Metrics**: Execution times, success/failure rates, parallel execution timing
- **Error Tracking**: Comprehensive error context with trace correlation

#### Custom Trace Naming

StruktX automatically generates meaningful trace names using context information:

```python
# Context provided in invoke call
response = ai.invoke(
    "Turn on the living room lights", 
    context={
        "user_id": "user123",
        "unit_id": "apartment456", 
        "thread_id": "session_789"  # Optional, UUID generated if missing
    }
)
# Creates trace: user123-apartment456-session_789-1234567890
```

#### Component Label Customization

Customize the component label shown in traces:

```python
config = StruktConfig(
    tracing=TracingConfig(
        component_label="StruktX"  # Changes "Engine.run(user123)" to "StruktX.run(user123)"
    )
)
```

#### Background Task Integration

Background tasks are automatically nested within the original trace:

```python
# Main request creates root trace
response = ai.invoke("Control bedroom AC", context={"user_id": "user123"})

# Background task appears nested in Weave:
# StruktX.run(user123)
# └── BackgroundTask.device_control
#     ├── Input: {task_id, query_type, parts, user_id}
#     └── Output: {status: "completed", result: {...}}
```

#### OpenTelemetry Integration

StruktX exports all traces to Weave via OpenTelemetry Protocol (OTLP):

```python
config = StruktConfig(
    opentelemetry=OpenTelemetryConfig(
        enabled=True,
        project_id="my-project",
        api_key_env="WANDB_API_KEY",
        export_endpoint="https://trace.wandb.ai/otel/v1/traces",  # Optional, auto-detected
        use_openai_instrumentation=True  # Auto-instrument OpenAI SDK calls
    )
)
```

#### Advanced Configuration

```python
config = StruktConfig(
    tracing=TracingConfig(
        component_label="MyApp",           # Custom component name
        collapse_status_ops=True,         # Collapse status events into attributes  
        enable_middleware_tracing=False   # Optional middleware operation tracing
    )
)
```

#### Manual User Context Tracking

For advanced use cases, you can still use the `weave_context` method:

**Explicit values:**
```python
# Track all operations within a user context
with ai.weave_context(
    user_id="user123",
    unit_id="apartment456",
    unit_name="Sunset Apartments"
):
    # All operations within this context will have user context
    response = ai.invoke("What's the weather like today?")
    response2 = ai.invoke("Can you help me schedule maintenance?")
```

**From context dictionary:**
```python
# Extract user context from a dictionary
user_context = {
    "user_id": "user456",
    "unit_id": "apartment789",
    "unit_name": "Downtown Loft"
}

with ai.weave_context(context=user_context):
    response = ai.invoke("Can you help me schedule maintenance?")
```

**Mixed explicit and context values:**
```python
# Explicit values take precedence over context dictionary
with ai.weave_context(
    user_id="explicit_override",  # This overrides context["user_id"]
    context=user_context
):
    response = ai.invoke("Turn on the living room lights")
```

**From InvocationState (for handlers):**
```python
# Automatically extract context from InvocationState
with ai.weave_context_from_state(state):
    # All operations will have user context from state.context
    response = ai.invoke("What's my current temperature setting?")
```

#### Custom Operation Tracking

Decorate functions with Weave tracking:

```python
@ai.create_weave_op(name="process_user_request", call_display_name="Process Request")
def process_user_request(user_input: str, user_context: dict) -> str:
    """This function will be automatically tracked by Weave."""
    # Function logic here
    return f"Processed: {user_input}"

# Call the decorated function
result = process_user_request("Hello", {"user_id": "user123"})
```

#### Weave Dashboard Information

In your Weave dashboard, you'll see comprehensive tracking:

1. **Engine Operations**: Complete request lifecycle from start to completion
2. **LLM Operations**: Input prompts, outputs, timing, and performance metrics
3. **Handler Operations**: Input/output tracking, execution times, success rates
4. **Memory Operations**: Retrieval patterns, injection sources, context usage
5. **User Context**: All operations tagged with user_id, unit_id, unit_name
6. **Performance Metrics**: Execution times, throughput, latency, success/failure rates
7. **Error Tracking**: Error types, messages, context at time of error, stack traces

#### Advanced Usage

Access Weave functionality directly through the Strukt instance:

```python
# Check if Weave is available
if ai.is_weave_available():
    print("Weave logging is enabled")
    
    # Get project information
    project_name, environment = ai.get_weave_project_info()
    print(f"Project: {project_name}-{environment}")

# Create custom Weave operations
@ai.create_weave_op(name="custom_operation")
def my_custom_function():
    pass

# Use context managers for user tracking
with ai.weave_context(user_id="user1", unit_id="unit1"):
    # Operations tracked with user context
    pass
```

#### Installation

Install Weave as an optional dependency:

```bash
# Install with Weave support
pip install struktx[weave]

# Or install Weave separately
pip install weave
```

#### Best Practices

1. **Project Naming**: Use descriptive project names that reflect your application
2. **Environment Separation**: Use different environments for development, staging, and production
3. **User Context**: Always provide user context for better tracking and debugging
4. **Custom Operations**: Decorate important business logic functions for detailed tracking
5. **Error Handling**: Weave automatically tracks errors, but ensure proper exception handling
6. **Performance Monitoring**: Use the tracked metrics to identify bottlenecks and optimize performance

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

**How do handler intents work?** Handlers extract intents and store them in `state.context['handler_intents'][handler_name] = action`. The background task middleware uses these intents to determine if a task should run in background based on the configured `action_based_background` rules.

**How do async optimizations work?** StruktX automatically bridges sync and async handlers. Use `await app.ainvoke()` for maximum performance with up to 15 concurrent handlers. Existing sync handlers work without changes.

**What's the difference between sync and async handlers?** Sync handlers run in thread pools when called via `ainvoke()`. Async handlers run natively with true concurrency. Hybrid handlers can implement both for optimal performance.

**How are Pydantic responses preserved?** When multiple handlers return structured objects, StruktX preserves them as a list. Single responses return the full object. Mixed responses fall back to string concatenation.

**How do I monitor performance?** Enable `EngineOptimizationsConfig` and access metrics via `app._engine._performance_monitor` or the `/metrics` endpoint in FastAPI applications.

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