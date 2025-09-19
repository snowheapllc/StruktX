from strukt import (
    create,
    StruktConfig,
    HandlersConfig,
    LLMClientConfig,
    ClassifierConfig,
    MemoryConfig,
    MiddlewareConfig,
    AwsSigV4Signer,
    AWSSecretsManager
)

from strukt.classifiers.llm_classifier import DefaultLLMClassifier
from strukt.mcp import build_fastapi_app
from strukt.langchain_helpers import parse_models_list
from strukt.logging import StruktLogger
from strukt.middleware import BackgroundTaskMiddleware, ResponseCleanerMiddleware
from strukt.memory import UpstashVectorMemoryEngine, MemoryExtractionMiddleware

from strukt.extensions.roomi.time.handler import TimeHandler
from strukt.extensions.roomi import (
    FutureEventHandler,
    HelpdeskHandler,
    EventHandler,
    BillHandler,
    NotificationHandler,
    AmenityHandler,
    WeatherHandler,
)
from strukt.extensions.roomi.web_search import WebSearchHandler
from strukt.extensions.roomi.future_event import FutureEventToolkit
from strukt.extensions.roomi.helpdesk import HelpdeskToolkit
from strukt.extensions.roomi.event import EventToolkit
from strukt.extensions.roomi.bill import BillToolkit
from strukt.extensions.roomi.notification import NotificationToolkit
from strukt.extensions.roomi.amenity import AmenityToolkit
from strukt.extensions.roomi.weather import WeatherToolkit
from strukt.extensions.roomi.time import TimeToolkit
from strukt.extensions.roomi.web_search import WebSearchToolkit
from strukt.extensions.roomi.future_event.transport import FutureEventTransport
from strukt.extensions.roomi.helpdesk.transport import HelpdeskTransport
from strukt.extensions.roomi.event.transport import EventTransport
from strukt.extensions.roomi.bill.transport import BillTransport
from strukt.extensions.roomi.notification.transport import NotificationTransport
from strukt.extensions.roomi.amenity.transport import AmenityTransport
from strukt.extensions.roomi.weather.transport import WeatherTransport
from strukt.extensions.roomi.time.transport import TimeTransport
from strukt.extensions.roomi.web_search.transport import WebSearchTransport
from strukt.extensions.roomi.devices.toolkit import DeviceToolkit
from strukt.extensions.roomi.devices.prompts import MCP_DEVICE_CONTROL_USAGE_PROMPT
from strukt.extensions.roomi.devices.handler import DeviceControlHandler
from strukt.extensions.roomi.cached_handlers import (
    CachedDeviceHandler,
    CachedWeatherHandler,
    CachedFutureEventHandler,
    CachedHelpdeskHandler,
    CachedEventHandler,
    CachedBillHandler,
    CachedNotificationHandler,
    CachedAmenityHandler,
)
from strukt.extensions.roomi.devices.transport import AWSSignedHttpTransport
from strukt.extensions.roomi.devices.models import MCPDeviceCacheInvalidateResponse


from strukt.extensions.roomi.types import (
    RoomiRequest,
    RoomiResponse,
    BackgroundTaskResponse,
    BackgroundTasksResponse,
    HealthResponse,
    ElevenLabsWebhookResponse,
    ElevenLabsKnowledgeResponse,
    MemoryStatsResponse,
    MemoryCleanupResponse,
    WebhookProcessingRequest,
)

from extras import _verify_elevenlabs_signature, DEFAULT_CLASSIFIER_TEMPLATE

from fastapi import FastAPI, HTTPException, Request, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader

import os
import json
import time
import re
import asyncio
from datetime import datetime, timezone
from uuid import uuid4

# Environment setup
os.environ["STRUKTX_DEBUG"] = "1"
os.environ["STRUKTX_LOG_LEVEL"] = "info"
os.environ["STRUKTX_RICH_TRACEBACK"] = "1"
os.environ["STRUKTX_LOG_MAXLEN"] = "512"
os.environ["AWS_REGION"] = "me-central-1"

AWS_REGION = "me-central-1"
AWS_SECRET_NAME = "stage/roomi-ai-crew"
ROOMI_TOKEN = APIKeyHeader(
    name="x-roomi-token", description="Private Roomi authentication token"
)

MEMORY_DISABLED = "Memory system disabled"

logger = StruktLogger("roomi-ai-struktx")

# Initialize AWS Secrets Manager
AWSSecretsManager(
    region_name=AWS_REGION,
    secret_name=AWS_SECRET_NAME,
).inject_secrets_into_env()

# Create intent cache engine with proper configuration
from strukt.memory import create_default_memory_config, InMemoryIntentCacheEngine, IntentCacheConfig, HandlerCacheConfig, CacheStrategy, CacheScope, DictData

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
            cache_data_type=DictData,
            strategy=CacheStrategy.SEMANTIC,
            scope=CacheScope.USER,
            ttl_seconds=1800,
            max_entries=500,
            similarity_threshold=0.6,
            enable_fast_track=True
        ),
        "DeviceHandler": HandlerCacheConfig(
            handler_name="DeviceHandler",
            cache_data_type=DictData,
            strategy=CacheStrategy.SEMANTIC,
            scope=CacheScope.USER,
            ttl_seconds=1800,
            max_entries=500,
            similarity_threshold=0.8,
            enable_fast_track=True
        ),
        "FutureEventHandler": HandlerCacheConfig(
            handler_name="FutureEventHandler",
            cache_data_type=DictData,
            strategy=CacheStrategy.SEMANTIC,
            scope=CacheScope.USER,
            ttl_seconds=3600,
            max_entries=300,
            similarity_threshold=0.8,
            enable_fast_track=True
        ),
        "HelpdeskHandler": HandlerCacheConfig(
            handler_name="HelpdeskHandler",
            cache_data_type=DictData,
            strategy=CacheStrategy.SEMANTIC,
            scope=CacheScope.USER,
            ttl_seconds=7200,
            max_entries=200,
            similarity_threshold=0.8,
            enable_fast_track=True
        ),
        "EventHandler": HandlerCacheConfig(
            handler_name="EventHandler",
            cache_data_type=DictData,
            strategy=CacheStrategy.SEMANTIC,
            scope=CacheScope.USER,
            ttl_seconds=1800,
            max_entries=300,
            similarity_threshold=0.8,
            enable_fast_track=True
        ),
        "BillHandler": HandlerCacheConfig(
            handler_name="BillHandler",
            cache_data_type=DictData,
            strategy=CacheStrategy.SEMANTIC,
            scope=CacheScope.USER,
            ttl_seconds=3600,
            max_entries=200,
            similarity_threshold=0.8,
            enable_fast_track=True
        ),
        "NotificationHandler": HandlerCacheConfig(
            handler_name="NotificationHandler",
            cache_data_type=DictData,
            strategy=CacheStrategy.SEMANTIC,
            scope=CacheScope.USER,
            ttl_seconds=1800,
            max_entries=300,
            similarity_threshold=0.8,
            enable_fast_track=True
        ),
        "AmenityHandler": HandlerCacheConfig(
            handler_name="AmenityHandler",
            cache_data_type=DictData,
            strategy=CacheStrategy.SEMANTIC,
            scope=CacheScope.USER,
            ttl_seconds=3600,
            max_entries=300,
            similarity_threshold=0.8,
            enable_fast_track=True
        )
    }
)

intent_cache_engine = InMemoryIntentCacheEngine(intent_cache_config)

# Build Strukt app config with intent caching
strukt_config = StruktConfig(
        weave={"enabled": True, "project_name": os.getenv("PROJECT_NAME"), "environment": os.getenv("CURRENT_ENV")},
        tracing={"component_label": "Roomi", "collapse_status_ops": True},
        evaluation={"enabled": False},
        llm=LLMClientConfig(
            "langchain_openai:ChatOpenAI",
            dict(
                model=os.getenv("OPENROUTER_BASE_MODEL"),
                api_key=os.getenv("OPENROUTER_API_KEY"),
                base_url=os.getenv("OPENROUTER_BASE_URL"),
                extra_body=dict(
                    fallback_models=parse_models_list(
                        os.getenv("OPENROUTER_FALLBACK_MODELS", "")
                    )
                ),
                temperature=0.2,
            ),
            retry={
                "max_retries": 1,
                "base_delay": 1.0,
                "max_delay": 5.0,
                "exponential_base": 2.0,
                "jitter": True,
                "retryable_exceptions": (Exception,),  # Retry on any exception
            }
        ),
        classifier=ClassifierConfig(
            DefaultLLMClassifier,
            dict(
                prompt_template=DEFAULT_CLASSIFIER_TEMPLATE,
                allowed_types=[
                    "time_service",
                    "device_control",
                    "schedule_future_event",
                    "maintenance_or_helpdesk",
                    "event_service",
                    "bill_service",
                    "notification_service",
                    "amenity_or_restaurant",
                    "weather_service",
                    "web_search",
                    "memory_extraction",
                ],
                max_parts=7,
            ),
        ),
        memory=MemoryConfig(
            factory=UpstashVectorMemoryEngine,
            params=dict(
                index_url=os.getenv("UPSTASH_VECTOR_REST_URL"),
                index_token=os.getenv("UPSTASH_VECTOR_REST_TOKEN"),
            ),
            use_store=False,
            augment_llm=False,
        ),
        handlers=HandlersConfig(
            {
                "time_service": TimeHandler, 
                "device_control": CachedDeviceHandler,
                "schedule_future_event": CachedFutureEventHandler,
                "maintenance_or_helpdesk": CachedHelpdeskHandler,
                "event_service": CachedEventHandler,
                "bill_service": CachedBillHandler,
                "notification_service": CachedNotificationHandler,
                "amenity_or_restaurant": CachedAmenityHandler,
                "weather_service": CachedWeatherHandler,
                "web_search": WebSearchHandler,
            },
            default_route="web_search",
            handler_params=dict(
                device_control=dict(
                    toolkit=DeviceToolkit(
                        transport=AWSSignedHttpTransport(
                            base_url=os.getenv("STATE_MANAGER_URL"),
                            log_devices_response=True,
                            payload_builder=lambda devices, user_id, unit_id: dict(
                                user_id=user_id,
                                unit_id=unit_id,
                                data=dict(devices=devices),
                            ),
                            user_header="x-user-id",
                            unit_header="x-unit-id",
                            signer=AwsSigV4Signer(
                                service="execute-api", region=os.getenv("AWS_REGION")
                            ),
                        )
                    ),
                    intent_cache_engine=intent_cache_engine
                ),
                schedule_future_event=dict(
                    toolkit=FutureEventToolkit(
                        transport=FutureEventTransport(
                            base_url=os.getenv("HANDLER_URI"),
                            user_header="x-user-id",
                            unit_header="x-unit-id",
                            signer=AwsSigV4Signer(
                                service="execute-api", region=os.getenv("AWS_REGION")
                            ),
                        )
                    ),
                    intent_cache_engine=intent_cache_engine
                ),
                maintenance_or_helpdesk=dict(
                    toolkit=HelpdeskToolkit(
                        transport=HelpdeskTransport(
                            base_url=os.getenv("SERVICES_URL"),
                            user_header="x-user-id",
                            unit_header="x-unit-id",
                            signer=AwsSigV4Signer(
                                service="execute-api", region=os.getenv("AWS_REGION")
                            ),
                        )
                    ),
                    intent_cache_engine=intent_cache_engine
                ),
                event_service=dict(
                    toolkit=EventToolkit(
                        transport=EventTransport(
                            base_url=os.getenv("SERVICES_URL"),
                            user_header="x-user-id",
                            unit_header="x-unit-id",
                            signer=AwsSigV4Signer(
                                service="execute-api", region=os.getenv("AWS_REGION")
                            ),
                        )
                    ),
                    intent_cache_engine=intent_cache_engine
                ),
                bill_service=dict(
                    toolkit=BillToolkit(
                        transport=BillTransport(
                            base_url=os.getenv("SERVICES_URL"),
                            user_header="x-user-id",
                            unit_header="x-unit-id",
                            signer=AwsSigV4Signer(
                                service="execute-api", region=os.getenv("AWS_REGION")
                            ),
                        )
                    ),
                    intent_cache_engine=intent_cache_engine
                ),
                notification_service=dict(
                    toolkit=NotificationToolkit(
                        transport=NotificationTransport(
                            novu_api_key=os.getenv("NOVU_SECRET_KEY"),
                            novu_notification_uri=os.getenv("NOVU_NOTIFICATION_URI"),
                            base_url=os.getenv("SERVICES_URL"),
                            user_header="x-user-id",
                            unit_header="x-unit-id",
                            signer=AwsSigV4Signer(
                                service="execute-api", region=os.getenv("AWS_REGION")
                            ),
                        )
                    ),
                    intent_cache_engine=intent_cache_engine
                ),
                amenity_or_restaurant=dict(
                    toolkit=AmenityToolkit(
                        transport=AmenityTransport(
                            base_url=os.getenv("SERVICES_URL"),
                            user_header="x-user-id",
                            unit_header="x-unit-id",
                            signer=AwsSigV4Signer(
                                service="execute-api", region=os.getenv("AWS_REGION")
                            ),
                        )
                    ),
                    intent_cache_engine=intent_cache_engine
                ),
                weather_service=dict(
                    toolkit=WeatherToolkit(
                        transport=WeatherTransport(
                            api_key=os.getenv("OPENWEATHER_API_KEY"),
                            units="metric"
                        )
                    ),
                    intent_cache_engine=intent_cache_engine
                ),
                time_service=dict(
                    toolkit=TimeToolkit(
                        transport=TimeTransport()
                    )
                ),
                web_search=dict(
                    toolkit=WebSearchToolkit(
                        transport=WebSearchTransport(
                            api_key=os.getenv("TAVILY_API_KEY")
                        )
                    )
                )
            ),
        ),
        middleware=[
            # MiddlewareConfig(MemoryExtractionMiddleware),
            MiddlewareConfig(ResponseCleanerMiddleware),
            MiddlewareConfig(BackgroundTaskMiddleware,
                             dict(
                max_workers=6,
                default_message="Your request is being processed.",
                enable_background_for={"device_control"},
                action_based_background={
                    "maintenance_or_helpdesk": {"create"},  # Only run in background for "create" action
                },
                custom_messages={
                    "device_control": "Device control successful",
                    "maintenance_or_helpdesk": "I've created your helpdesk ticket. Someone will be in touch shortly.",
                },
                return_query_types={
                    "device_control": "DEVICE_CONTROL_SUCCESS",
                    "maintenance_or_helpdesk": "HELPDESK_TICKET_CREATED",
                }
            )),
        ],
        mcp=dict(
            enabled=False,
            server_name="roomi-ai-mcp",
            include_handlers=[
                "time_service",
                "device_control",
                "schedule_future_event",
                "maintenance_or_helpdesk",
                "event_service",
                "bill_service",
                "notification_service",
                "amenity_or_restaurant",
                "weather_service",
                "web_search",
            ],
            default_consent_policy="ask-once",
            tools={
                "device_control": [
                    dict(
                        name="device_control_execute",
                        description="Use this to control devices using natural language. Pass a list of query parts that contain device control requests. For example: ['turn on the kitchen AC', 'set the kitchen AC to 24 degrees']",
                        usage_prompt=MCP_DEVICE_CONTROL_USAGE_PROMPT,
                    ),
                    dict(
                        name="device_control_invalidate_cache",
                        description="Invalidate cached device list and index for a specific user/unit",
                        usage_prompt="Call with user_id and unit_id to clear per-request device cache",
                    ),
                ],
            },
        )
    )



strukt_app = create(strukt_config)

# Create FastAPI app
app = FastAPI(
    title="Roomi AI StruktX Server",
    description="Roomi AI Assistant API with StruktX framework",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    root_path="/v1",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app = build_fastapi_app(strukt_app=strukt_app, cfg=strukt_config, app=app)

# API Routes
@app.get("/health", tags=["Health"], response_model=HealthResponse)
async def health():
    """Check if the API is running"""
    return HealthResponse(status="ok", message="API is running")


@app.post("/roomi", response_model=RoomiResponse, tags=["Roomi Assistant"])
async def process_message(request: RoomiRequest):
    """Process a message through Roomi using StruktX"""
    try:
        # Create context for StruktX
        context = {
            "user_id": request.user_id,
            "unit_id": request.unit_id,
            "unit_name": request.unit_name,
            "thread_id": request.thread_id,
        }

        # Parallel orchestrator (soft 3000ms) for multi-intent across handlers
        orchestrator_enabled = os.getenv("ORCHESTRATOR_ENABLED", "1").lower() in {"1", "true", "yes"}
        soft_timeout_s = float(os.getenv("GLOBAL_SOFT_TIMEOUT_MS", "3000")) / 1000.0
        transcript = request.transcript or ""

        if orchestrator_enabled and any(k in transcript.lower() for k in [" and ", ",", " then "]):
            # Split into coarse parts
            parts = [p for p in re.split(r"\s*(?:,|\band\b|\bthen\b)\s*", transcript, flags=re.IGNORECASE) if p]

            device_parts: list[str] = []
            time_parts: list[str] = []
            weather_parts: list[str] = []
            other_parts: list[str] = []

            # Simple detectors (aligned with handlers)
            def is_time(p: str) -> bool:
                pl = p.lower()
                kws = [
                    "what time", "time in", "timezone", "current time", "local time", "what's the time", "clock", "hour", "minute",
                ]
                return any(k in pl for k in kws)

            def is_weather(p: str) -> bool:
                pl = p.lower()
                kws = ["weather", "temperature", "forecast", "how hot", "how cold", "weather like"]
                return any(k in pl for k in kws)

            try:
                from roomi.devices.fast_intent_lark import LarkIntentEngine
                from roomi.devices.fast_intent import IntentType
                engine = LarkIntentEngine()
            except Exception:
                engine = None
                IntentType = None  # type: ignore

            for p in parts:
                if engine is not None:
                    try:
                        parsed_out = engine.parse(p)
                        plist = parsed_out if isinstance(parsed_out, list) else [parsed_out]
                        ok = any(getattr(pi, "type", None) == IntentType.ACTUATE and not getattr(pi, "is_scheduled", False) for pi in plist)
                        if ok:
                            device_parts.append(p)
                            continue
                    except Exception:
                        pass
                if is_time(p):
                    time_parts.append(p)
                elif is_weather(p):
                    weather_parts.append(p)
                else:
                    other_parts.append(p)

            tasks: list[asyncio.Task] = []
            payloads: list[tuple[str, list[str]]] = []
            if device_parts:
                payloads.append((" ".join(device_parts), device_parts))
            if time_parts:
                payloads.append((" ".join(time_parts), time_parts))
            if weather_parts:
                payloads.append((" ".join(weather_parts), weather_parts))
            if not payloads:
                # Fallback to normal invoke
                result = await strukt_app.ainvoke(transcript, context)
                return RoomiResponse(
                    response=result.response,
                    background_tasks=[],
                    query_type=result.query_type,
                    query_types=result.query_types,
                    transcript_parts=result.parts,
                )

            for subtext, _ in payloads:
                tasks.append(asyncio.create_task(strukt_app.ainvoke(subtext, context)))

            done, pending = await asyncio.wait(tasks, timeout=soft_timeout_s)
            results = []
            for t in done:
                try:
                    results.append(t.result())
                except Exception as e:
                    results.append(None)
            # We won't cancel pending; let them run in background but ignore here

            # Safely extract responses, handling both string and dict responses
            response_parts = []
            for r in results:
                if r is not None and getattr(r, "response", None):
                    response = r.response
                    if isinstance(response, dict):
                        # Extract message from dict response if available
                        if "message" in response:
                            response_parts.append(response["message"])
                        else:
                            response_parts.append(str(response))
                    else:
                        response_parts.append(str(response))
            
            combined_response = " ".join(response_parts) or "Done."
            qtypes = []
            for r in results:
                if r is not None and getattr(r, "query_types", None):
                    for qt in r.query_types:
                        if qt not in qtypes:
                            qtypes.append(qt)
            primary_qtype = results[0].query_type if results and results[0] is not None else "GENERAL"
            all_parts = []
            for _, p in payloads:
                all_parts.extend(p)

            return RoomiResponse(
                response=combined_response,
                background_tasks=[],
                query_type=primary_qtype,
                query_types=qtypes or [primary_qtype],
                transcript_parts=all_parts,
            )

        # Default single-path invoke
        result = await strukt_app.ainvoke(transcript, context)

        return RoomiResponse(
            response=result.response,
            background_tasks=[],  # StruktX handles background tasks internally
            query_type=result.query_type,
            query_types=result.query_types,
            transcript_parts=result.parts,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing request: {str(e)}"
        )


# Device cache invalidation endpoint
@app.post("/devices/cache/invalidate", tags=["Devices"], response_model=MCPDeviceCacheInvalidateResponse)
async def invalidate_device_cache(request: Request):
    try:
        # Access the toolkit directly from the config and invalidate cache
        toolkit = strukt_config.handlers.handler_params["device_control"]["toolkit"]
        body = await request.json()
        user_id = body.get("user_id")
        unit_id = body.get("unit_id")
        result = toolkit.invalidate_cache(user_id=user_id, unit_id=unit_id)
        return MCPDeviceCacheInvalidateResponse(
            success=result,
            user_id=user_id,
            unit_id=unit_id,
            message=(
                "Cache invalidated" if result else "No cache entries existed for this user/unit"
            ),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error invalidating cache: {str(e)}")

# ElevenLabs Webhook Endpoints
@app.post(
    "/elevenlabs/webhook",
    response_model=ElevenLabsWebhookResponse,
    tags=["ElevenLabs Webhook"],
)
async def elevenlabs_webhook(request: Request, token: str = Security(ROOMI_TOKEN)):
    """Handle ElevenLabs webhook with multi-intent orchestrator support"""
    try:
        # Verify auth token
        if not token or token != os.getenv("ROOMI_AUTH_TOKEN"):
            raise HTTPException(status_code=401, detail="Unauthorized")

        body = await request.json()
        user_id = body.get("user_id")
        transcript = body.get("transcript")
        unit_id = body.get("unit_id")
        unit_name = body.get("unit_name")

        if not user_id or not transcript or not unit_id or not unit_name:
            raise ValueError("Invalid Request")

        logger.info(
            f"Received ElevenLabs webhook request from {user_id} in {unit_id} ({unit_name}): {transcript}"
        )

        # Create context for StruktX
        context = {
            "user_id": user_id,
            "unit_id": unit_id,
            "unit_name": unit_name,
            "thread_id": str(uuid4()),
        }

        # Parallel orchestrator (soft 3000ms) for multi-intent across handlers
        orchestrator_enabled = os.getenv("ORCHESTRATOR_ENABLED", "1").lower() in {"1", "true", "yes"}
        soft_timeout_s = float(os.getenv("GLOBAL_SOFT_TIMEOUT_MS", "3000")) / 1000.0
        transcript = transcript or ""

        if orchestrator_enabled and any(k in transcript.lower() for k in [" and ", ",", " then "]):
            # Split into coarse parts
            parts = [p for p in re.split(r"\s*(?:,|\band\b|\bthen\b)\s*", transcript, flags=re.IGNORECASE) if p]

            device_parts: list[str] = []
            time_parts: list[str] = []
            weather_parts: list[str] = []
            other_parts: list[str] = []

            # Simple detectors (aligned with handlers)
            def is_time(p: str) -> bool:
                pl = p.lower()
                kws = [
                    "what time", "time in", "timezone", "current time", "local time", "what's the time", "clock", "hour", "minute",
                ]
                return any(k in pl for k in kws)

            def is_weather(p: str) -> bool:
                pl = p.lower()
                kws = ["weather", "temperature", "forecast", "how hot", "how cold", "weather like"]
                return any(k in pl for k in kws)

            try:
                from roomi.devices.fast_intent_lark import LarkIntentEngine
                from roomi.devices.fast_intent import IntentType
                engine = LarkIntentEngine()
            except Exception:
                engine = None
                IntentType = None  # type: ignore

            for p in parts:
                if engine is not None:
                    try:
                        parsed_out = engine.parse(p)
                        plist = parsed_out if isinstance(parsed_out, list) else [parsed_out]
                        ok = any(getattr(pi, "type", None) == IntentType.ACTUATE and not getattr(pi, "is_scheduled", False) for pi in plist)
                        if ok:
                            device_parts.append(p)
                            continue
                    except Exception:
                        pass
                if is_time(p):
                    time_parts.append(p)
                elif is_weather(p):
                    weather_parts.append(p)
                else:
                    other_parts.append(p)

            tasks: list[asyncio.Task] = []
            payloads: list[tuple[str, list[str]]] = []
            if device_parts:
                payloads.append((" ".join(device_parts), device_parts))
            if time_parts:
                payloads.append((" ".join(time_parts), time_parts))
            if weather_parts:
                payloads.append((" ".join(weather_parts), weather_parts))
            if not payloads:
                # Fallback to normal invoke
                result = await strukt_app.ainvoke(transcript, context)
                return ElevenLabsWebhookResponse(
                    success=True,
                    message="Webhook processed successfully",
                    response=result.response,
                )

            for subtext, _ in payloads:
                tasks.append(asyncio.create_task(strukt_app.ainvoke(subtext, context)))

            done, pending = await asyncio.wait(tasks, timeout=soft_timeout_s)
            results = []
            for t in done:
                try:
                    results.append(t.result())
                except Exception as e:
                    results.append(None)
            # We won't cancel pending; let them run in background but ignore here

            # Safely extract responses, handling both string and dict responses
            response_parts = []
            for r in results:
                if r is not None and getattr(r, "response", None):
                    response = r.response
                    if isinstance(response, dict):
                        # Extract message from dict response if available
                        if "message" in response:
                            response_parts.append(response["message"])
                        else:
                            response_parts.append(str(response))
                    else:
                        response_parts.append(str(response))
            
            combined_response = " ".join(response_parts) or "Done."
            
            return ElevenLabsWebhookResponse(
                success=True,
                message="Webhook processed successfully with multi-intent orchestration",
                response=combined_response,
            )

        # Default single-path invoke
        result = await strukt_app.ainvoke(transcript, context)

        return ElevenLabsWebhookResponse(
            success=True,
            message="Webhook processed successfully",
            response=result.response,
        )

    except Exception as e:
        logger.error(f"Error processing webhook: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error processing webhook: {str(e)}"
        )


@app.post(
    "/elevenlabs/knowledge",
    tags=["ElevenLabs Knowledge Webhook"],
    response_model=ElevenLabsKnowledgeResponse,
)
async def elevenlabs_knowledge_webhook(request: Request):
    """Process ElevenLabs webhook to extract knowledge from transcript with HMAC authentication"""
    logger.info("🔔 ElevenLabs Knowledge Webhook received")
    try:
        # Check if memory system is enabled
        use_memory = os.getenv("USE_MEMORY", "true").lower() == "true"
        if not use_memory:
            logger.info("❌ Memory system disabled, skipping knowledge extraction")
            return ElevenLabsKnowledgeResponse(
                success=True, message=MEMORY_DISABLED, nodes_created=0, edges_created=0
            )

        # Get the raw body for HMAC verification
        payload = await request.body()
        payload_str = payload.decode("utf-8")

        # Verify HMAC signature
        if not await _verify_elevenlabs_signature(request, payload_str):
            logger.error("❌ HMAC signature verification failed")
            return ElevenLabsKnowledgeResponse(
                success=True,
                message="Invalid HMAC signature (Incorrect environment most likely)",
                nodes_created=0,
                edges_created=0,
            )

        # Parse the webhook data
        webhook_data = json.loads(payload_str)
        webhook_type = webhook_data.get("type")

        # Validate webhook type
        if webhook_type != "post_call_transcription":
            logger.error(f"❌ Invalid webhook type: {webhook_type}")
            return ElevenLabsKnowledgeResponse(
                success=True,
                message="Invalid webhook type",
                nodes_created=0,
                edges_created=0,
            )

        # Extract data from the webhook structure
        data = webhook_data.get("data", {})
        dynamic_variables = data.get("conversation_initiation_client_data", {}).get(
            "dynamic_variables", {}
        )

        # Extract user_id and unit_id from dynamic_variables
        user_id = dynamic_variables.get("user_id")
        unit_id = dynamic_variables.get("unit_id")
        conversation_id = data.get("conversation_id")
        transcript_data = data.get("transcript", [])

        if not user_id or not unit_id or not conversation_id:
            logger.error(
                f"❌ Missing required fields - user_id: {user_id}, unit_id: {unit_id}, conversation_id: {conversation_id}"
            )
            raise HTTPException(
                status_code=400,
                detail="Missing required fields: user_id, unit_id, conversation_id",
            )

        # Extract user messages from transcript
        user_messages = []
        for turn in transcript_data:
            role = turn.get("role")
            if role == "user":
                message = turn.get("message", "")
                user_messages.append(message)

        # Combine user messages into a single transcript
        full_transcript = " ".join(user_messages)

        if not full_transcript.strip():
            logger.warn(
                f"⚠️ No user messages found in transcript for conversation {conversation_id}"
            )
            return ElevenLabsKnowledgeResponse(
                success=True,
                message="No user messages to process",
                nodes_created=0,
                edges_created=0,
            )

        logger.info(f"📄 Transcript: {full_transcript[:100]}")

        # Create webhook processing request
        event_timestamp = webhook_data.get("event_timestamp", time.time())
        webhook_request = WebhookProcessingRequest(
            user_id=user_id,
            unit_id=unit_id,
            transcript=full_transcript,
            conversation_id=conversation_id,
            timestamp=datetime.fromtimestamp(event_timestamp, tz=timezone.utc),
        )

        # Process webhook to extract knowledge using StruktX memory
        logger.info("🔍 Starting knowledge extraction process")
        memory = strukt_app.get_memory()
        if memory is None:
            logger.error("Memory engine is None, returning default response")
            return ElevenLabsKnowledgeResponse(
                success=True,
                message=MEMORY_DISABLED,
                nodes_created=0,
                edges_created=0,
                conversation_id=conversation_id,
                user_id=user_id,
            )

        # Add to memory using StruktX
        try:
            memory.add(
                full_transcript,
                {
                    "user_id": user_id,
                    "unit_id": unit_id,
                    "conversation_id": conversation_id,
                    "timestamp": event_timestamp,
                    "source": "elevenlabs_webhook",
                },
            )

            logger.info("🎉 Knowledge extraction successful")
            return ElevenLabsKnowledgeResponse(
                success=True,
                message="Knowledge extracted successfully",
                nodes_created=1,  # Simplified - actual count would depend on memory engine
                edges_created=0,
                conversation_id=conversation_id,
                user_id=user_id,
            )
        except Exception as e:
            logger.error(f"❌ Knowledge extraction failed: {str(e)}")
            return ElevenLabsKnowledgeResponse(
                success=False,
                message=f"Knowledge extraction failed: {str(e)}",
                nodes_created=0,
                edges_created=0,
            )

    except Exception as e:
        logger.error(f"💥 Unexpected error in knowledge webhook processing: {str(e)}")
        return ElevenLabsKnowledgeResponse(
            success=False,
            message=f"Error processing knowledge webhook: {str(e)}",
            nodes_created=0,
            edges_created=0,
        )


# Background Task Status Endpoints
@app.get(
    "/background-tasks",
    response_model=BackgroundTasksResponse,
    tags=["Background Tasks"],
)
async def get_all_background_tasks():
    """Get all background tasks"""
    try:
        tasks = strukt_app.get_all_background_tasks()
        return BackgroundTasksResponse(
            tasks=[BackgroundTaskResponse(**task) for task in tasks],
            total_count=len(tasks),
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error retrieving background tasks: {str(e)}"
        )


@app.get(
    "/background-tasks/running",
    response_model=BackgroundTasksResponse,
    tags=["Background Tasks"],
)
async def get_running_background_tasks():
    """Get all currently running background tasks"""
    try:
        tasks = strukt_app.get_running_background_tasks()
        return BackgroundTasksResponse(
            tasks=[BackgroundTaskResponse(**task) for task in tasks],
            total_count=len(tasks),
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving running background tasks: {str(e)}",
        )


@app.get(
    "/background-tasks/completed",
    response_model=BackgroundTasksResponse,
    tags=["Background Tasks"],
)
async def get_completed_background_tasks():
    """Get all completed background tasks"""
    try:
        tasks = strukt_app.get_completed_background_tasks()
        return BackgroundTasksResponse(
            tasks=[BackgroundTaskResponse(**task) for task in tasks],
            total_count=len(tasks),
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving completed background tasks: {str(e)}",
        )


@app.get(
    "/background-tasks/failed",
    response_model=BackgroundTasksResponse,
    tags=["Background Tasks"],
)
async def get_failed_background_tasks():
    """Get all failed background tasks"""
    try:
        tasks = strukt_app.get_failed_background_tasks()
        return BackgroundTasksResponse(
            tasks=[BackgroundTaskResponse(**task) for task in tasks],
            total_count=len(tasks),
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving failed background tasks: {str(e)}",
        )


@app.get(
    "/background-tasks/{task_id}",
    response_model=BackgroundTaskResponse,
    tags=["Background Tasks"],
)
async def get_background_task_info(task_id: str):
    """Get information about a specific background task"""
    try:
        task_info = strukt_app.get_background_task_info(task_id)
        if task_info is None:
            raise HTTPException(
                status_code=404, detail=f"Background task with ID {task_id} not found"
            )
        return BackgroundTaskResponse(**task_info)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error retrieving background task info: {str(e)}"
        )


@app.get(
    "/background-tasks/status/{status}",
    response_model=BackgroundTasksResponse,
    tags=["Background Tasks"],
)
async def get_background_tasks_by_status(status: str):
    """Get background tasks filtered by status"""
    try:
        tasks = strukt_app.get_background_tasks_by_status(status)
        return BackgroundTasksResponse(
            tasks=[BackgroundTaskResponse(**task) for task in tasks],
            total_count=len(tasks),
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving background tasks by status: {str(e)}",
        )


# Memory endpoints
@app.get("/memory/stats", tags=["Memory"], response_model=MemoryStatsResponse)
async def get_memory_stats():
    """Get memory system statistics"""
    try:
        # Check if memory system is enabled
        use_memory = os.getenv("USE_MEMORY", "true").lower() == "true"
        if not use_memory:
            return MemoryStatsResponse(
                success=True,
                message=MEMORY_DISABLED,
                stats={
                    "total_graphs": 0,
                    "total_nodes": 0,
                    "total_edges": 0,
                    "nodes_by_category": {},
                    "graphs_by_user": {},
                },
            )

        memory = strukt_app.get_memory()
        if memory is None:
            return MemoryStatsResponse(
                success=True,
                message=MEMORY_DISABLED,
                stats={
                    "total_graphs": 0,
                    "total_nodes": 0,
                    "total_edges": 0,
                    "nodes_by_category": {},
                    "graphs_by_user": {},
                },
            )

        # Note: This is a simplified response. You may need to implement
        # specific memory stats based on your memory engine
        return MemoryStatsResponse(
            success=True,
            message="Memory stats retrieved successfully",
            stats={
                "total_graphs": 0,  # Implement based on your memory engine
                "total_nodes": 0,
                "total_edges": 0,
                "nodes_by_category": {},
                "graphs_by_user": {},
            },
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error retrieving memory stats: {str(e)}"
        )


@app.post("/memory/cleanup", tags=["Memory"], response_model=MemoryCleanupResponse)
async def cleanup_memories(
    user_id: str = None, unit_id: str = None, max_age_days: int = 14
):
    """Clean up memories older than the specified number of days"""
    try:
        # Check if memory system is enabled
        use_memory = os.getenv("USE_MEMORY", "true").lower() == "true"
        if not use_memory:
            return MemoryCleanupResponse(
                success=True,
                message=MEMORY_DISABLED,
                cleanup_result={
                    "cleanup_performed": False,
                    "reason": MEMORY_DISABLED,
                    "nodes_removed": 0,
                    "edges_removed": 0,
                    "graphs_affected": 0,
                },
                max_age_days=max_age_days,
                user_id=user_id,
                unit_id=unit_id,
            )

        memory = strukt_app.get_memory()
        if memory is None:
            return MemoryCleanupResponse(
                success=True,
                message=MEMORY_DISABLED,
                cleanup_result={
                    "cleanup_performed": False,
                    "reason": MEMORY_DISABLED,
                    "nodes_removed": 0,
                    "edges_removed": 0,
                    "graphs_affected": 0,
                },
                max_age_days=max_age_days,
                user_id=user_id,
                unit_id=unit_id,
            )

        # Perform cleanup using StruktX memory
        try:
            cleanup_result = memory.cleanup(max_age_days=max_age_days)

            # Determine scope description for response
            if user_id and unit_id:
                scope_desc = f"user {user_id}, unit {unit_id}"
            elif user_id:
                scope_desc = f"user {user_id}"
            else:
                scope_desc = "all users"

            return MemoryCleanupResponse(
                success=True,
                message=f"Memory cleanup completed for {scope_desc}",
                cleanup_result=cleanup_result,
                max_age_days=max_age_days,
                user_id=user_id,
                unit_id=unit_id,
            )
        except Exception as e:
            return MemoryCleanupResponse(
                success=False,
                message=f"Memory cleanup failed: {str(e)}",
                cleanup_result={
                    "cleanup_performed": False,
                    "reason": str(e),
                    "nodes_removed": 0,
                    "edges_removed": 0,
                    "graphs_affected": 0,
                },
                max_age_days=max_age_days,
                user_id=user_id,
                unit_id=unit_id,
            )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error during memory cleanup: {str(e)}"
        )


@app.get("/cache/stats", tags=["Cache Management"])
async def get_cache_stats():
    """Get intent cache statistics."""
    try:
        stats = intent_cache_engine.get_stats()
        # Log stats with pretty formatting
        intent_cache_engine.log_stats()
        return {
            "total_entries": stats.total_entries,
            "hits": stats.hits,
            "misses": stats.misses,
            "hit_rate": stats.hit_rate,
            "average_similarity": stats.average_similarity,
            "handler_stats": stats.handler_stats
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error retrieving cache stats: {str(e)}"
        )


@app.post("/cache/cleanup", tags=["Cache Management"])
async def cleanup_cache():
    """Clean up expired cache entries."""
    try:
        stats = intent_cache_engine.cleanup()
        return {
            "expired_entries_removed": stats.expired_entries,
            "total_entries": stats.total_entries,
            "hits": stats.hits,
            "misses": stats.misses,
            "hit_rate": stats.hit_rate
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error during cache cleanup: {str(e)}"
        )


@app.delete("/cache/clear", tags=["Cache Management"])
async def clear_cache():
    """Clear all cache entries."""
    try:
        # Clear all entries by creating a new cache engine
        global intent_cache_engine
        memory_config = create_default_memory_config()
        intent_cache_engine = InMemoryIntentCacheEngine(memory_config.intent_cache)
        return {
            "success": True,
            "message": "Cache cleared successfully"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error clearing cache: {str(e)}"
        )
