"use client"

import React, { useEffect, useRef } from 'react'
import { motion } from 'framer-motion'
import { CodeBlock } from '@/components/ui/code-block'
import { gsap } from 'gsap'

export default function DocsPage() {
  const pageRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (typeof window === 'undefined') return
    
    const ctx = gsap.context(() => {
      // Animate the main title
      gsap.fromTo('.docs-title',
        { opacity: 0, y: -30, scale: 0.95 },
        {
          opacity: 1,
          y: 0,
          scale: 1,
          duration: 1,
          ease: "power2.out",
          delay: 0.5
        }
      )

      // Animate code blocks with stagger
      gsap.fromTo('.code-block',
        { opacity: 0, y: 20, scale: 0.98 },
        {
          opacity: 1,
          y: 0,
          scale: 1,
          duration: 0.8,
          stagger: 0.2,
          ease: "power2.out",
          delay: 0.8
        }
      )
    }, pageRef)

    return () => ctx.revert()
  }, [])

  return (
    <div ref={pageRef} className="prose prose-invert max-w-none">
      <motion.section 
        id="introduction" 
        className="section"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8, delay: 0.3 }}
      >
        <h1 className="text-3xl font-bold docs-title">StruktX Documentation</h1>
        <p className="text-dark-700 dark:text-dark-300">
          StruktX is a lean, typed framework for building Natural Language → Action applications. It provides swappable components for LLMs, classifiers, handlers, and optional memory, along with middleware hooks and LangChain helpers.
        </p>
      </motion.section>

      <div className="section-header">
        <h2 className="text-2xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-blue-600 to-white dark:from-blue-400 dark:to-white">Getting Started</h2>
      </div>

      <section id="quickstart" className="section">
        <h2 className="text-2xl font-semibold">Quickstart</h2>
        <p>Minimal example with the default components:</p>
        <CodeBlock className="code-block" language="python" filename="quickstart.py" showExample={true} code={`from strukt import create, StruktConfig, HandlersConfig

app = create(StruktConfig(
    handlers=HandlersConfig(default_route="general")
))

print(app.invoke("Hello, StruktX!").response)
`} />

        <p>With LangChain and memory augmentation:</p>
        <CodeBlock className="code-block" language="python" filename="quickstart_memory.py" showExample={true} code={`import os
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
`} />
      </section>
      
      <section id="architecture" className="section">
        <h2 className="text-2xl font-semibold">Architecture</h2>
        <p>
          Core flow: text → classify → group parts → route to handlers → combine responses.
          Components are swappable via factories and follow interfaces for type safety.
        </p>
        <div className="concept-list">
          <div className="concept-list-item">
            <span className="concept-badge">LLM Client</span>
            <span>Any model client implementing invoke/structured.</span>
          </div>
          <div className="concept-list-item">
            <span className="concept-badge">Classifier</span>
            <span>Maps input to one or more query types and parts.</span>
          </div>
          <div className="concept-list-item">
            <span className="concept-badge">Handlers</span>
            <span>Process grouped parts per query type.</span>
          </div>
          <div className="concept-list-item">
            <span className="concept-badge">Memory</span>
            <span>Optional, supports scoped retrieval and injection.</span>
          </div>
          <div className="concept-list-item">
            <span className="concept-badge">Middleware</span>
            <span>Before/after classify and handle hooks.</span>
          </div>
        </div>
      </section>

      <section id="configuration" className="section">
        <h2 className="text-2xl font-semibold">Configuration</h2>
        <p>Factory-based config supports callables, classes, instances, or import strings like "module:attr". Dicts are coerced into dataclasses.</p>
        <CodeBlock className="code-block" language="python" filename="config.py" showExample={true} code={`from strukt import (
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
`} />

        <p>Swap LLMs and pass custom parameters (including OpenAI-compatible providers):</p>
        
        <CodeBlock className="code-block" language="python" filename="llm_swap.py" showExample={true} code={`from strukt import StruktConfig, LLMClientConfig

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
`} />
      </section>

      <div className="section-header">
        <h2 className="text-2xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-blue-600 to-white dark:from-blue-400 dark:to-white">Core Components</h2>
      </div>

      <section id="providers" className="section">
        <h2 className="text-2xl font-semibold">Providers (OpenAI-compatible)</h2>
        <p>OpenRouter, Groq, and Cerebras expose OpenAI-style APIs and work via LangChain's <code>ChatOpenAI</code> or direct OpenAI clients using a custom base URL.</p>
        <h4 className="text-lg font-semibold">OpenRouter</h4>
        
        <CodeBlock className="code-block" language="python" filename="openrouter.py" showExample={true} code={`import os
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
`} />
        <h4 className="text-lg font-semibold">Groq</h4>
        
        <CodeBlock className="code-block" language="python" filename="groq.py" showExample={true} code={`import os
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
`} />
        <h4 className="text-lg font-semibold">Cerebras</h4>
        
        <CodeBlock className="code-block" language="python" filename="cerebras.py" showExample={true} code={`import os
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
`} />
        <p>Alternatively, if you use the OpenAI SDK directly, set <code>OPENAI_BASE_URL</code> env and pass your key. StruktX will auto-adapt LangChain runnables to <code>LLMClient</code>.</p>
      </section>

      <section id="llm" className="section">
        <h3 className="text-xl font-semibold">LLM Clients</h3>
        <p>Provide your own <code>LLMClient</code> or adapt LangChain with <code>LangChainLLMClient</code>.</p>
        
        <CodeBlock className="code-block" language="python" filename="llm_client.py" showExample={true} code={`from strukt.interfaces import LLMClient

class MyLLM(LLMClient):
    def invoke(self, prompt: str, **kwargs):
        return type("Resp", (), {"content": prompt.upper()})

    def structured(self, prompt: str, output_model, **kwargs):
        return output_model()
`} />
      </section>

      <section id="classifier" className="section">
        <h3 className="text-xl font-semibold">Classifier</h3>
        <p>Return query types, confidences, and parts.</p>
        
        <CodeBlock className="code-block" language="python" filename="classifier.py" showExample={true} code={`from strukt.interfaces import Classifier
from strukt.types import InvocationState, QueryClassification

class MyClassifier(Classifier):
    def classify(self, state: InvocationState) -> QueryClassification:
        return QueryClassification(query_types=["general"], confidences=[1.0], parts=[state.text])
`} />
      </section>

      <section id="handlers" className="section">
        <h3 className="text-xl font-semibold">Handlers</h3>
        <p>Handle grouped parts for a given query type.</p>
        
        <CodeBlock className="code-block" language="python" filename="handler.py" showExample={true} code={`from strukt.interfaces import Handler, LLMClient
from strukt.types import InvocationState, HandlerResult

class EchoHandler(Handler):
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def handle(self, state: InvocationState, parts: list[str]) -> HandlerResult:
        return HandlerResult(response=" | ".join(parts), status="general")
`} />
      </section>

      <section id="middleware" className="section">
        <h3 className="text-xl font-semibold">Middleware</h3>
        <p>Hooks: before_classify, after_classify, before_handle, after_handle.</p>
        
        <CodeBlock className="code-block" language="python" filename="middleware.py" showExample={true} code={`from strukt.middleware import Middleware
from strukt.types import InvocationState, HandlerResult, QueryClassification

class Metrics(Middleware):
    def before_classify(self, state: InvocationState):
        state.context["t0"] = 0
        return state

    def after_handle(self, state: InvocationState, query_type: str, result: HandlerResult):
        return result
`} />

        <p> Memory extraction middleware (packaged):</p>
        <CodeBlock className="code-block" language="python" filename="memory_extraction.py" showExample={true} code={`from strukt import StruktConfig, MemoryConfig, MiddlewareConfig

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
`} />
      </section>

      <section id="memory" className="section">
        <h3 className="text-xl font-semibold">Memory</h3>
        <p>Enable scoped memory and automatic prompt augmentation.</p>
        
        <CodeBlock className="code-block" language="python" filename="memory.py" showExample={true} code={`from strukt import StruktConfig, MemoryConfig

cfg = StruktConfig(
  memory=MemoryConfig(factory=None, use_store=False, augment_llm=True)
)
`} />
      </section>

      <div className="section-header">
        <h2 className="text-2xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-blue-600 to-white dark:from-blue-400 dark:to-white">Ecosystem</h2>
      </div>

      <section id="langchain" className="section">
        <h3 className="text-xl font-semibold">LangChain Helpers</h3>
        <p>Use <code>LangChainLLMClient</code> and <code>create_structured_chain</code> to generate typed outputs.</p>
        
        <CodeBlock className="code-block" language="python" filename="langchain.py" showExample={true} code={`from pydantic import BaseModel
from strukt.langchain_helpers import create_structured_chain

class Foo(BaseModel):
    query_types: list[str] = []
    confidences: list[float] = []
    parts: list[str] = []

# chain = create_structured_chain(llm_client=your_langchain_client, prompt_template="...", output_model=Foo)
`} />
      </section>

      <section id="logging" className="section">
        <h3 className="text-xl font-semibold">Logging</h3>
        <p>Use <code>get_logger</code> and <code>LoggingMiddleware</code>. <br/>
        Optional logging variables: <code>STRUKTX_LOG_LEVEL</code>, <code>STRUKTX_LOG_MAXLEN</code>, <code>STRUKTX_RICH_TRACEBACK</code>, <code>STRUKTX_DEBUG</code>.</p>
        
        <CodeBlock className="code-block" language="python" filename="logging.py" showExample={true} code={`from strukt import get_logger

log = get_logger("struktx")
log.info("Hello logs")
`} />
        <p>Augmented memory injections appear under the <code>memory</code> logger with the provided <code>augment_source</code> label.</p>
      </section>

      <section id="memory-extraction" className="section">
        <h3 className="text-xl font-semibold">Memory Extraction Middleware</h3>
        <p>Automatically extracts durable facts from conversations and stores them in your memory engine (e.g., Upstash Vector). On subsequent requests, <code>MemoryAugmentedLLMClient</code> retrieves relevant items and prepends them to prompts.</p>
        
        <div className="concept-list">
          <div className="concept-list-item">
            <span className="concept-badge">Extraction</span>
            <span>After handler or classification, extracts facts (e.g., preferences, locations).</span>
          </div>
          <div className="concept-list-item">
            <span className="concept-badge">Storage</span>
            <span>Writes to memory with scope from <code>context.user_id</code> and optionally <code>context.unit_id</code>.</span>
          </div>
          <div className="concept-list-item">
            <span className="concept-badge">Retrieval</span>
            <span>On the next request, scoped items are used to enrich prompts.</span>
          </div>
        </div>
        <CodeBlock className="code-block" language="python" filename="memory_extraction_config.py" showExample={true} code={`from strukt import create, StruktConfig, MemoryConfig
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
`} />
        <div className="note-box">
          <span className="concept-badge">Note</span>
          <div>
            <p><code>MemoryConfig.use_store=True</code> is required for <code>MemoryExtractionMiddleware</code> to work.</p>
            <p>It currently only reads context of <code>user_id</code> and <code>unit_id</code>, if you wish to change this behavior, you can implement a custom <code>MemoryAugmentedLLMClient</code> and <code>MemoryExtractionMiddleware</code>.</p>
          </div>
        </div>
        <p>Tips:</p>
        <ul className="list-disc pl-6">
          <li>Always scope memories per user/session.</li>
          <li>Keep extracted items concise and factual. Avoid storing ephemeral content.</li>
          <li>Control verbosity with env and middleware params. Logs show when memory is injected.</li>
        </ul>
      </section>

      <section id="extensions" className="section">
        <h3 className="text-xl font-semibold">Extensions</h3>
        <p>Build reusable packages exposing factories for handlers, middleware, and memory engines.</p>
        
        <CodeBlock className="code-block" language="python" filename="extension_structure.py" showExample={true} code={`# your_extension/__init__.py
from .handlers import DeviceHandler
from .middleware import DeviceAuthMiddleware
from .models import DeviceCommand

__all__ = ["DeviceHandler", "DeviceAuthMiddleware", "DeviceCommand"]
`} />
      </section>

      <section id="devices-extension" className="section">
        <h3 className="text-xl font-semibold">Devices Extension</h3>
        <p>Example of building a devices extension that receives natural language and triggers device actions.</p>
        <h4 className="text-lg font-semibold">Models</h4>
        
        <CodeBlock className="code-block" language="python" filename="models.py" showExample={true} code={`from pydantic import BaseModel, Field

class DeviceCommand(BaseModel):
    device_id: str = Field(..., min_length=1)
    action: str = Field(..., min_length=1)  # e.g., "turn_on", "set_temp"
    value: str | int | float | None = None
`} />
        <h4 className="text-lg font-semibold">Handler</h4>
        
        <CodeBlock className="code-block" language="python" filename="handlers.py" showExample={true} code={`from strukt.interfaces import Handler, LLMClient
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
`} />
        <h4 className="text-lg font-semibold">Middleware (optional)</h4>
        
        <CodeBlock className="code-block" language="python" filename="middleware.py" showExample={true} code={`from strukt.middleware import Middleware
from strukt.types import InvocationState

class DeviceAuthMiddleware(Middleware):
    def before_handle(self, state: InvocationState, query_type: str, parts: list[str]):
        if query_type == "devices":
            if not state.context.get("auth_token"):
                # deny by changing parts to an error sentinel or add a flag
                parts = ["UNAUTHORIZED"]
        return state, parts
`} />
        <h4 className="text-lg font-semibold">Register</h4>
        
        <CodeBlock className="code-block" language="python" filename="main.py" showExample={true} code={`from strukt import create, StruktConfig, HandlersConfig, MiddlewareConfig
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
`} />
        <p>Best practices:</p>
        <ul className="list-disc pl-6">
          <li>Validate structured outputs with Pydantic models that reflect your device API.</li>
          <li>Keep device-side effects idempotent; return clear status strings for telemetry.</li>
          <li>Use <code>query_hint</code> and <code>augment_source</code> to improve logging and memory quality.</li>
        </ul>
      </section>

      <div className="section-header">
        <h2 className="text-2xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-blue-600 to-white dark:from-blue-400 dark:to-white">Advanced</h2>
      </div>

      <section id="query-hints" className="section">
        <h3 className="text-xl font-semibold">Query Hints</h3>
        <p>Pass <code>query_hint</code> to help memory retrieval and logging. It is provided when calling an LLM automatically or can be modified when invoking any LLM inhereting from the StruktX LLM classes.</p>
        
        <CodeBlock className="code-block" language="python" filename="query_hints.py" showExample={true} code={`resp = app.invoke("recommend lunch", context={"user_id": "u1"})
# or: llm.invoke(prompt, query_hint="recommendation")
`} />
      </section>

      <section id="augment-source" className="section">
        <h3 className="text-xl font-semibold">augment_source</h3>
        <p>Provide <code>augment_source</code> when calling an LLM client to label memory injection source in logs.</p>
        
        <CodeBlock className="code-block" language="python" filename="augment_source.py" showExample={true} code={`# Inside a handler
llm.invoke(prompt, context=state.context, query_hint=state.text, augment_source="recommendations")
`} />
      </section>

      <section id="context" className="section">
        <h3 className="text-xl font-semibold">Context & Scoping</h3>
        <p>Use <code>user_id</code> and optionally <code>unit_id</code> in context for scoped memory retrieval. If a <code>KnowledgeStore</code> is enabled, StruktX may use it to further scope memory.</p>
        
        <CodeBlock className="code-block" language="python" filename="context.py" showExample={true} code={`# Handlers receive the full invocation state
def handle(self, state: InvocationState, parts: list[str]) -> HandlerResult:
    user_id = state.context.get("user_id")
    # enrich prompts or enforce auth
    ...
`} />
      </section>

      <section id="step-by-step" className="section">
        <h3 className="text-xl font-semibold">Step-by-Step: Build Your First App</h3>
        <p>This guided tutorial creates two handlers <span className="concept-badge-inline">Time Handler</span> and <span className="concept-badge-inline">Weather Handler</span> adds a custom <span className="concept-badge-inline">Rate Limit Middleware</span> and enables <span className="concept-badge-inline">Memory Extraction</span></p>

        <div className="step-title">
          <span className="step-number">0</span>
          <h4 className="text-lg font-semibold m-0">Setup</h4>
        </div>
        <div className="step-content">
          <p>Create a virtual environment and install dependencies.</p>
        <CodeBlock className="code-block" language="bash" filename="step_00_setup.sh" showExample={true} code={`uv venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
uv pip install struktX
`} />
        </div>

        <div className="step-title">
          <span className="step-number">1</span>
          <h4 className="text-lg font-semibold m-0">Weather Model</h4>
        </div>
        <div className="step-content">
          <p>Typed outputs reduce parsing errors and improve reliability when extracting fields from natural language.</p>
        <CodeBlock className="code-block" language="python" filename="step_01_models.py" showExample={true} code={`from pydantic import BaseModel, Field

class WeatherQuery(BaseModel):
    city: str = Field(..., min_length=1)
    unit: str | None = Field(default="celsius", description="celsius or fahrenheit")
`} />
        </div>

        <div className="step-title">
          <span className="step-number">2</span>
          <h4 className="text-lg font-semibold m-0">Time Handler</h4>
        </div>
        <div className="step-content">
          <p>Handlers encapsulate business logic per query type and receive the full <code>InvocationState</code></p>
          <CodeBlock className="code-block" language="python" filename="step_02_time_handler.py" showExample={true} code={`from strukt.interfaces import Handler, LLMClient
from strukt.types import InvocationState, HandlerResult
from datetime import datetime

class TimeHandler(Handler):
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def handle(self, state: InvocationState, parts: list[str]) -> HandlerResult:
        now = datetime.utcnow().strftime("%H:%M UTC")
        return HandlerResult(response=f"Current time: {now}", status="time")
`} />
        </div>

        <div className="step-title">
          <span className="step-number">3</span>
          <h4 className="text-lg font-semibold m-0">Weather Handler</h4>
        </div>
        <div className="step-content">
          <p>Use <code>llm.structured</code> to reliably extract fields into <code>WeatherQuery</code>, then call a weather client.</p>
          <CodeBlock className="code-block" language="python" filename="step_03_weather_handler.py" showExample={true} code={`from strukt.interfaces import Handler, LLMClient
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
`} />
        </div>

        <div className="step-title">
          <span className="step-number">4</span>
          <h4 className="text-lg font-semibold m-0">Rate Limit Middleware</h4>
        </div>
        <div className="step-content">
          <p>Centralize rate limiting so handlers remain clean. This example counts calls by <code>user_id</code> and returns an error sentinel when exceeded.</p>
          <CodeBlock className="code-block" language="python" filename="step_04_rate_limit_middleware.py" showExample={true} code={`from strukt.middleware import Middleware
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
`} />
        </div>

        <div className="step-title">
          <span className="step-number">5</span>
          <h4 className="text-lg font-semibold m-0">Enable Memory</h4>
        </div>
        <div className="step-content">
          <p>Enable <code>MemoryExtractionMiddleware</code> and a vector store to persist durable facts. StruktX will auto-inject them via <code>MemoryAugmentedLLMClient</code> during LLM calls. <br/><br/>Use <code>MemoryConfig.use_store=True</code> and <code>MemoryConfig.augment_llm=True</code> to enable a <code>KnowledgeStore</code> to further scope memory.</p>
          <div className="note-box">
            <span className="concept-badge">Note</span>
            <div>
              <p><code>MemoryConfig.use_store=True</code> is required for <code>MemoryExtractionMiddleware</code> to work.</p>
            </div>
          </div>
          <CodeBlock className="code-block" language="python" filename="step_05_memory.py" showExample={true} code={`from strukt import MemoryConfig, MiddlewareConfig

memory = MemoryConfig(
  factory="strukt.memory:UpstashVectorMemoryEngine",
  params={"index_url": "...", "index_token": "...", "namespace": "demo"},
  use_store=True,
  augment_llm=True,
)

memory_mw = MiddlewareConfig("strukt.memory.middleware:MemoryExtractionMiddleware", params={"max_items": 5})
`} />
        </div>

        <div className="step-title">
          <span className="step-number">6</span>
          <h4 className="text-lg font-semibold m-0">Wire It Up</h4>
        </div>
        <div className="step-content">
          <p>Register handlers under query types and compose middleware. Set the default route for unclassified requests.</p>
          <CodeBlock className="code-block" language="python" filename="step_06_config.py" showExample={true} code={`import os
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
`} />
        </div>

        <div className="step-title">
          <span className="step-number">7</span>
          <h4 className="text-lg font-semibold m-0">Run</h4>
        </div>
        <div className="step-content">
          <p>Invoke with <code>user_id</code> so rate-limiting and memory are scoped per user.</p>
          <CodeBlock className="code-block" language="python" filename="step_07_run.py" showExample={true} code={`print(app.invoke("what's the time now?", context={"user_id": "u1"}).response)
print(app.invoke("weather in Paris in celsius", context={"user_id": "u1"}).response)

for _ in range(5):
    r = app.invoke("time please", context={"user_id": "u1"})
    print(r.response)
`} />
        </div>

        <h4 className="text-lg font-semibold">Why this structure?</h4>
        <div className="concept-list">
          <div className="concept-list-item">
            <span className="concept-badge">Handlers</span>
            <span>isolate use cases; adding a new capability is additive.</span>
          </div>
          <div className="concept-list-item">
            <span className="concept-badge">Middleware</span>
            <span>keeps cross-cutting concerns reusable and testable.</span>
          </div>
          <div className="concept-list-item">
            <span className="concept-badge">Memory</span>
            <span>improves continuity and reduces repeated user input.</span>
          </div>
          <div className="concept-list-item">
            <span className="concept-badge">Typed Extraction</span>
            <span>increases determinism for downstream consumers.</span>
          </div>
        </div>
      </section>

      <section id="api-overview" className="section">
        <h3 className="text-xl font-semibold">Reference (Overview)</h3>
        <ul className="list-disc pl-6">
          <li><code>strukt.create(config)</code>: builds the app with LLM, classifier, handlers, memory, middleware.</li>
          <li><code>Strukt.invoke(text, context)</code> / <code>Strukt.ainvoke</code>: run requests.</li>
          <li><code>StruktConfig</code>: top-level config dataclass; subconfigs: <code>LLMClientConfig</code>, <code>ClassifierConfig</code>, <code>HandlersConfig</code>, <code>MemoryConfig</code>, <code>MiddlewareConfig</code>.</li>
          <li><code>interfaces.LLMClient</code>: <code>invoke</code>, <code>structured</code>.</li>
          <li><code>interfaces.Classifier</code>: <code>classify</code> → <code>QueryClassification</code>.</li>
          <li><code>interfaces.Handler</code>: <code>handle</code> → <code>HandlerResult</code>.</li>
          <li><code>interfaces.MemoryEngine</code>: <code>add</code>, <code>get</code>, <code>get_scoped</code>, <code>remove</code>, <code>cleanup</code>.</li>
          <li><code>defaults.MemoryAugmentedLLMClient</code>: auto-injects relevant memory into prompts; supports <code>augment_source</code> and <code>query_hint</code>.</li>
          <li><code>logging.get_logger</code>, <code>LoggingMiddleware</code>: structured, Rich-powered console logging.</li>
          <li><code>langchain_helpers.LangChainLLMClient</code>, <code>adapt_to_llm_client</code>, <code>create_structured_chain</code>.</li>
          <li><code>utils.load_factory</code>, <code>utils.coerce_factory</code>: resolve factories from strings/callables/classes/instances.</li>
          <li><code>types</code>: <code>InvocationState</code>, <code>QueryClassification</code>, <code>HandlerResult</code>, <code>StruktResponse</code>, <code>StruktQueryEnum</code>.</li>
        </ul>
      </section>

      <section id="best-practices" className="section">
        <h3 className="text-xl font-semibold">Best Practices</h3>
        <ul className="list-disc pl-6">
          <li>Prefer typed handlers with clear responsibilities.</li>
          <li>Keep middleware small and composable.</li>
          <li>Scope memory using user-specific / traceable context.</li>
        </ul>
      </section>

      <section id="faq" className="section">
        <h3 className="text-xl font-semibold">FAQ</h3>
        <p><b>Can I use non-LangChain LLMs?</b> Yes—implement <code>LLMClient</code> or provide an adapter.</p>
        <p><b>How do I add a new query type?</b> Implement a handler and register it in <code>HandlersConfig.registry</code> and include it in the classifier config.</p>
        <p><b>How is memory injected?</b> If <code>MemoryConfig.augment_llm=True</code>, <code>MemoryAugmentedLLMClient</code> retrieves relevant docs and prepends them to prompts.</p>
      </section>

      <div className="section-header">
        <h2 className="text-2xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-blue-600 to-white dark:from-blue-400 dark:to-white">Extras</h2>
      </div>
      <section id="extras" className="section">
        <h3 className="text-xl font-semibold">MCP Server</h3>
        <h5 className="concept-badge-inline text-md font-semibold">Claude Desktop</h5>
        <CodeBlock className="code-block" language="json" filename="claude_desktop_config.json" code={`{
  "mcpServers": {
    "struktx": {
      "command": "npx",
      "args": ["-y", "mcp-remote", "https://struktx.vercel.app/api/mcp"]
    }
  }
}`} />

        <h5 className="concept-badge-inline text-md font-semibold">Cursor</h5>
        <CodeBlock className="code-block" language="json" filename=".cursor/mcp.json" code={`{
  "mcpServers": {
    "struktx": {
      "url": "https://struktx.vercel.app/api/mcp"
    }
  }
}`} />

        <h5 className="concept-badge-inline text-md font-semibold">Windsurf</h5>
        <CodeBlock className="code-block" language="json" filename="/.codeium/windsurf/mcp_config.json" code={`{
  "mcpServers": {
    "struktx": {
      "url": "https://struktx.vercel.app/api/mcp"
    }
  }
}`} />

      </section>
    </div>
  )
}


