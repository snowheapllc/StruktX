from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional
import time
import uuid

from .interfaces import Classifier, Handler, MemoryEngine
from .middleware import (
    Middleware,
    apply_after_classify,
    apply_after_handle,
    apply_before_classify,
    apply_before_handle,
    apply_should_run_background,
    apply_get_background_message,
    apply_get_return_query_type,
    apply_get_background_task_info,
    apply_get_all_background_tasks,
    apply_get_background_tasks_by_status,
)
from .types import (
    HandlerResult,
    InvocationState,
    QueryClassification,
    StruktQueryEnum,
    BackgroundTaskInfo,
)
from .logging import get_logger
from .tracing import (
    strukt_trace,
    unified_trace_context,
    generate_trace_name,
)
from .evaluation import log_post_run_evaluation


class Engine:
    def __init__(
        self,
        *,
        classifier: Classifier,
        handlers: Dict[str, Handler],
        default_route: str | None = None,
        memory: MemoryEngine | None = None,
        middleware: list[Middleware] | None = None,
        weave_config: Optional[object] = None,  # WeaveConfig type
        tracing_config: Optional[object] = None,  # TracingConfig type
        evaluation_config: Optional[object] = None,  # EvaluationConfig type
    ) -> None:
        self._classifier = classifier
        self._handlers = handlers
        self._default_route = default_route
        self._memory = memory
        self._middleware = list(middleware or [])
        self._weave_config = weave_config
        self._tracing_config = tracing_config
        self._evaluation_config = evaluation_config

        # Initialize logger for Weave tracking
        self._logger = get_logger("struktx-engine")
        self._weave_available = self._logger.is_weave_available()

        if self._weave_available:
            self._logger.info("Engine initialized with Weave logging enabled")
            # Apply dynamic tracing decorators with component label
            self._apply_dynamic_tracing()
        else:
            self._logger.info("Engine initialized without Weave logging")

    def _should_auto_track_user_context(self) -> bool:
        """Check if we should automatically track user context from operations."""
        if not self._weave_available or not self._weave_config:
            return False
        return getattr(self._weave_config, "track_user_context", False)

    def _extract_user_context(self, state: InvocationState) -> dict:
        """Extract full context from InvocationState without cherry-picking keys."""
        ctx = getattr(state, "context", None)
        return dict(ctx) if isinstance(ctx, dict) else {}

    def _create_contextual_operation_name(
        self, base_name: str, user_context: dict = None
    ) -> str:
        """Create a contextual operation name that includes user information."""
        if not user_context:
            return base_name

        # Extract user identifiers
        user_id = user_context.get("user_id", "")
        unit_id = user_context.get("unit_id", "")
        unit_name = user_context.get("unit_name", "")

        # Build contextual name
        context_parts = []
        if user_id:
            context_parts.append(f"user:{user_id}")
        if unit_id:
            context_parts.append(f"unit:{unit_id}")
        if unit_name:
            # Clean unit name for use in operation names
            clean_unit_name = unit_name.replace(" ", "_").replace("-", "_")[:20]
            context_parts.append(f"apt:{clean_unit_name}")

        if context_parts:
            return f"{base_name}[{','.join(context_parts)}]"

        return base_name

    def _log_operation(self, operation_name: str, user_context: dict = None, **kwargs):
        """Log an operation with Weave if available.

        Note: We create a real Weave op with a contextual name so it shows up
        clearly in the Weave trace/timeline UI.
        """
        if not self._weave_available:
            return

        try:
            import weave

            # Create a unique operation ID for tracking
            op_id = str(uuid.uuid4())

            # Create contextual operation name
            contextual_name = self._create_contextual_operation_name(
                operation_name, user_context
            )

            # Build attributes payload once
            attributes = {
                "operation_id": op_id,
                "operation_name": operation_name,
                "contextual_operation_name": contextual_name,
                "engine_component": "struktx_engine",
                "context": user_context or {},
                **kwargs,
            }

            # Collapse status ops to attributes when configured
            collapse = False
            try:
                if self._tracing_config is not None:
                    collapse = bool(
                        getattr(self._tracing_config, "collapse_status_ops", False)
                    )
            except Exception:
                collapse = False

            if collapse and operation_name in {
                "engine_run_start",
                "engine_run_complete",
                "engine_run_error",
                "grouped_handlers_start",
                "grouped_handlers_complete",
                "parallel_execution_start",
                "parallel_execution_complete",
                "parallel_execution_error",
                "background_task_created",
                "background_task_scheduled",
            }:
                with weave.attributes({f"status.{operation_name}": attributes}):
                    # Status operation collapsed to attributes
                    self._logger.debug(
                        f"Status operation {operation_name} logged as attributes"
                    )
            else:

                def _emit_operation(attrs: dict) -> dict:  # type: ignore[no-redef]
                    return attrs

                with weave.attributes(attributes):
                    _emit_operation(attributes)

            self._logger.debug(
                f"Weave operation logged: {contextual_name} (ID: {op_id})"
            )

        except Exception as e:
            self._logger.warn(f"Failed to log Weave operation {operation_name}: {e}")

    def _log_classification(
        self,
        state: InvocationState,
        classification: QueryClassification,
        duration: float,
        user_context: dict | None = None,
    ):
        """Log classification operation with detailed metrics."""
        self._log_operation(
            "query_classification",
            user_context=user_context,
            input_text=state.text,
            input_context=state.context,
            query_types=classification.query_types,
            confidences=classification.confidences,
            parts=classification.parts,
            duration_ms=duration * 1000,
            classifier_type=type(self._classifier).__name__,
        )

    def _log_handler_execution(
        self,
        query_type: str,
        handler: Handler,
        parts: List[str],
        state: InvocationState,
        duration: float,
        result: HandlerResult,
        user_context: dict | None = None,
    ):
        """Log handler execution with detailed metrics."""
        self._log_operation(
            "handler_execution",
            user_context=user_context,
            query_type=query_type,
            handler_type=type(handler).__name__,
            input_parts=parts,
            input_text=state.text,
            input_context=state.context,
            output_response=result.response,
            output_status=result.status,
            duration_ms=duration * 1000,
            success=result.status != "error",
        )

    def _log_memory_operation(self, operation: str, **kwargs):
        """Log memory operations."""
        if self._memory:
            self._log_operation(
                f"memory_{operation}", memory_type=type(self._memory).__name__, **kwargs
            )

    def _get_component_label(self) -> str:
        """Get the component label from tracing config, defaulting to 'Engine'."""
        try:
            if self._tracing_config is not None:
                return (
                    getattr(self._tracing_config, "component_label", "Engine")
                    or "Engine"
                )
        except Exception:
            pass
        return "Engine"

    def _apply_dynamic_tracing(self):
        """Apply tracing decorators dynamically with the correct component label."""
        try:
            component_label = self._get_component_label()

            # Apply tracing to key methods with dynamic component label
            self.run = strukt_trace(
                name="StruktX.Engine.run", call_display_name=f"{component_label}.run"
            )(self.run)

            self._classify = strukt_trace(
                name="StruktX.Engine.classify",
                call_display_name=f"{component_label}.classify",
            )(self._classify)

            self._execute_grouped_handlers = strukt_trace(
                name="StruktX.Engine.execute_grouped_handlers",
                call_display_name=f"{component_label}.execute_grouped_handlers",
            )(self._execute_grouped_handlers)

            self._execute_handlers_parallel = strukt_trace(
                name="StruktX.Engine.execute_handlers_parallel",
                call_display_name=f"{component_label}.execute_handlers_parallel",
            )(self._execute_handlers_parallel)

            self._execute_single_handler = strukt_trace(
                name="StruktX.Engine.execute_single_handler",
                call_display_name=f"{component_label}.execute_single_handler",
            )(self._execute_single_handler)

        except Exception as e:
            self._logger.debug(f"Failed to apply dynamic tracing: {e}")

    def run(self, state: InvocationState) -> List[HandlerResult]:
        """Main engine run method - all operations will be nested under this call."""
        # Determine thread/session id for grouping
        thread_id = None
        user_context = {}
        try:
            ctx = getattr(state, "context", {}) or {}
            thread_id = ctx.get("thread_id") or ctx.get("session_id")
            user_context = dict(ctx)
        except Exception:
            thread_id = None

        # Generate custom trace name using userID-unitID-threadID-timestamp format
        custom_trace_name = generate_trace_name(user_context)

        # Get component label from tracing config
        label = self._get_component_label()

        # Extract user ID or use context for display name
        user_id = user_context.get("user_id") if user_context else None
        display_context = user_id if user_id else "context"

        # Use unified trace context to ensure everything is nested
        with unified_trace_context(
            thread_id, f"{label}.run({display_context})", custom_trace_name
        ):
            start_time = time.time()
            run_id = str(uuid.uuid4())

            user_context = (
                self._extract_user_context(state)
                if self._should_auto_track_user_context()
                else user_context
            )

            results = self._run_with_context(state, start_time, run_id, user_context)

            # Post-run evaluation/logging (no-op unless enabled)
            try:
                display_name = None
                if thread_id:
                    display_name = f"engine_run:{thread_id}"
                log_post_run_evaluation(
                    getattr(self, "_evaluation_config", None),
                    input_text=state.text,
                    input_context=state.context,
                    results=results,
                    display_name=display_name,
                )
            except Exception:
                pass

            return results

    def _run_with_context(
        self,
        state: InvocationState,
        start_time: float,
        run_id: str,
        user_context: dict | None = None,
    ) -> List[HandlerResult]:
        """Internal method to run the engine with the given context."""
        self._log_operation(
            "engine_run_start",
            user_context=user_context,
            run_id=run_id,
            input_text=state.text,
            input_context=state.context,
            timestamp=start_time,
        )

        try:
            # Execute classification
            state, classification = self._classify(state, user_context)

            # Check for fallback
            fallback = self._maybe_fallback_handler()
            if self._should_fallback(state):
                if fallback:
                    self._logger.info(f"Using fallback handler for query: {state.text}")
                    result = fallback.handle(state, [state.text])
                    self._log_operation(
                        "fallback_handler_execution",
                        user_context=user_context,
                        run_id=run_id,
                        fallback_type=type(fallback).__name__,
                        input_text=state.text,
                        llm_response=result.response,  # Capture LLM output
                    )
                    return [result]
                else:
                    self._logger.warn("No fallback handler available")
                    return []

            # Group and execute handlers
            grouped = self._group_parts_by_type(state)
            results = self._execute_grouped_handlers(
                state, grouped, fallback, user_context
            )

            # Extract LLM responses and structured outputs from all results
            llm_responses = []
            structured_outputs = []
            for result in results:
                if hasattr(result, "response") and result.response:
                    llm_responses.append(result.response)

                # Capture structured outputs (like DeviceControlResponse)
                if hasattr(result, "commands") and result.commands:
                    structured_outputs.append(
                        {
                            "type": "device_commands",
                            "response": result.response,
                            "commands": result.commands,
                        }
                    )
                elif hasattr(result, "data") and result.data:
                    structured_outputs.append(
                        {
                            "type": "structured_data",
                            "response": result.response,
                            "data": result.data,
                        }
                    )
                elif hasattr(result, "result") and result.result:
                    structured_outputs.append(
                        {
                            "type": "handler_result",
                            "response": result.response,
                            "result": result.result,
                        }
                    )

            # Log completion with LLM outputs and structured data
            total_duration = time.time() - start_time
            self._log_operation(
                "engine_run_complete",
                user_context=user_context,
                run_id=run_id,
                total_duration_ms=total_duration * 1000,
                result_count=len(results),
                success=True,
                llm_responses=llm_responses,  # Capture all LLM outputs
                structured_outputs=structured_outputs,  # Capture structured data like DeviceControlResponse
                classification_result=classification.query_types,  # Capture classification
            )

            return results

        except Exception as e:
            # Log errors
            total_duration = time.time() - start_time
            self._log_operation(
                "engine_run_error",
                run_id=run_id,
                total_duration_ms=total_duration * 1000,
                error=str(e),
                error_type=type(e).__name__,
                success=False,
            )
            self._logger.error(f"Engine run failed: {e}")
            raise

    def _classify(
        self, state: InvocationState, user_context: dict = None
    ) -> tuple[InvocationState, QueryClassification]:
        start_time = time.time()

        # Apply middleware before classification
        state = apply_before_classify(self._middleware, state)

        # Execute classification - now traced via auto-instrumentation
        classification: QueryClassification = self._classifier.classify(state)

        # Apply middleware after classification
        state, classification = apply_after_classify(
            self._middleware, state, classification
        )

        # Update state with classification results
        state.query_types = list(classification.query_types)
        state.confidences = list(classification.confidences)
        state.parts = list(classification.parts)

        # Log classification with timing
        duration = time.time() - start_time
        self._log_classification(state, classification, duration, user_context)

        return state, classification

    def _maybe_fallback_handler(self) -> Handler | None:
        return self._handlers.get(self._default_route or StruktQueryEnum.GENERAL)

    def _should_fallback(self, state: InvocationState) -> bool:
        return not state.query_types or not state.parts

    def _group_parts_by_type(self, state: InvocationState) -> Dict[str, List[str]]:
        grouped: Dict[str, List[str]] = defaultdict(list)
        for idx, qtype in enumerate(state.query_types):
            part = state.parts[idx] if idx < len(state.parts) else state.text
            grouped[qtype].append(part)
        return grouped

    def _execute_grouped_handlers(
        self,
        state: InvocationState,
        grouped: Dict[str, List[str]],
        fallback: Handler | None,
        user_context: dict | None = None,
    ) -> List[HandlerResult]:
        start_time = time.time()
        execution_id = str(uuid.uuid4())
        results: List[HandlerResult] = []
        background_tasks = []
        normal_tasks = []

        # Execute grouped handlers - now traced via decorator
        self._do_grouped_handlers(
            state,
            grouped,
            fallback,
            user_context,
            execution_id,
            results,
            background_tasks,
            normal_tasks,
        )

        duration = time.time() - start_time
        self._log_operation(
            "grouped_handlers_complete",
            execution_id=execution_id,
            duration_ms=duration * 1000,
            total_results=len(results),
            background_tasks=len(background_tasks),
            normal_tasks=len(normal_tasks),
            success=True,
        )

        return results

    def _do_grouped_handlers(
        self,
        state: InvocationState,
        grouped: Dict[str, List[str]],
        fallback: Handler | None,
        user_context: dict | None,
        execution_id: str,
        results: List[HandlerResult],
        background_tasks: List[str],
        normal_tasks: List[tuple[str, List[str], Handler]],
    ) -> None:
        for qtype, parts in grouped.items():
            handler = self._handlers.get(qtype) or fallback
            if handler is None:
                self._logger.warn(f"No handler found for query type: {qtype}")
                continue

            state, parts = apply_before_handle(self._middleware, state, qtype, parts)

            if apply_should_run_background(self._middleware, state, qtype, parts):
                background_message = apply_get_background_message(
                    self._middleware, state, qtype, parts
                )
                return_query_type = apply_get_return_query_type(
                    self._middleware, state, qtype, parts
                )
                task_id = self._create_background_task(handler, state, qtype, parts)
                background_tasks.append(task_id)
                result = HandlerResult(
                    response=background_message + f" (Task ID: {task_id})",
                    status=return_query_type,
                )
                results.append(result)
                self._log_operation(
                    "background_task_scheduled",
                    execution_id=execution_id,
                    task_id=task_id,
                    query_type=qtype,
                    return_status=return_query_type,
                )
            else:
                normal_tasks.append((qtype, parts, handler))

        if normal_tasks:
            parallel_results = self._execute_handlers_parallel(
                state, normal_tasks, user_context
            )
            results.extend(parallel_results)

    def _execute_handlers_parallel(
        self,
        state: InvocationState,
        tasks: List[tuple[str, List[str], Handler]],
        user_context: dict | None = None,
    ) -> List[HandlerResult]:
        """Execute multiple handlers in parallel."""
        start_time = time.time()
        parallel_id = str(uuid.uuid4())

        results: List[HandlerResult] = []

        # Use weave ThreadPoolExecutor for proper trace nesting
        try:
            import weave

            with weave.ThreadPoolExecutor(max_workers=len(tasks)) as executor:
                future_to_task = {}
                for qtype, parts, handler in tasks:
                    future = executor.submit(
                        self._execute_single_handler,
                        state,
                        qtype,
                        parts,
                        handler,
                        user_context,
                        None,
                    )
                    future_to_task[future] = (qtype, parts, handler)

                for future in as_completed(future_to_task):
                    qtype, parts, handler = future_to_task[future]
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        error_result = HandlerResult(
                            response=f"Error executing {qtype}: {str(e)}",
                            status="error",
                        )
                        results.append(error_result)
                        self._log_operation(
                            "parallel_execution_error",
                            parallel_id=parallel_id,
                            query_type=qtype,
                            error=str(e),
                            error_type=type(e).__name__,
                        )
        except ImportError:
            # Fallback to regular ThreadPoolExecutor if weave not available
            self._execute_handlers_parallel_fallback(
                state, tasks, user_context, results, parallel_id
            )

        duration = time.time() - start_time
        self._log_operation(
            "parallel_execution_complete",
            parallel_id=parallel_id,
            duration_ms=duration * 1000,
            result_count=len(results),
            success_count=len([r for r in results if r.status != "error"]),
            error_count=len([r for r in results if r.status == "error"]),
        )

        return results

    def _execute_handlers_parallel_fallback(
        self,
        state: InvocationState,
        tasks: List[tuple[str, List[str], Handler]],
        user_context: dict | None,
        results: List[HandlerResult],
        parallel_id: str,
    ) -> None:
        try:
            from opentelemetry import context as otel_context  # type: ignore
        except Exception:
            otel_context = None  # type: ignore

        with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
            future_to_task = {}
            parent_ctx = None
            try:
                if otel_context is not None:
                    parent_ctx = otel_context.get_current()
            except Exception:
                parent_ctx = None
            for qtype, parts, handler in tasks:
                future = executor.submit(
                    self._execute_single_handler,
                    state,
                    qtype,
                    parts,
                    handler,
                    user_context,
                    parent_ctx,
                )
                future_to_task[future] = (qtype, parts, handler)
            for future in as_completed(future_to_task):
                qtype, parts, handler = future_to_task[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    error_result = HandlerResult(
                        response=f"Error executing {qtype}: {str(e)}",
                        status="error",
                    )
                    results.append(error_result)
                    self._log_operation(
                        "parallel_execution_error",
                        parallel_id=parallel_id,
                        query_type=qtype,
                        error=str(e),
                        error_type=type(e).__name__,
                    )

    def _execute_single_handler(
        self,
        state: InvocationState,
        query_type: str,
        parts: List[str],
        handler: Handler,
        user_context: dict | None = None,
        parent_otel_ctx: object | None = None,
    ) -> HandlerResult:
        """Execute a single handler and apply middleware."""
        start_time = time.time()

        # Attach OTEL parent context if provided (for cross-thread propagation)
        token = None
        try:
            if parent_otel_ctx is not None:
                from opentelemetry import context as otel_context  # type: ignore

                token = otel_context.attach(parent_otel_ctx)  # type: ignore
        except Exception:
            token = None

        try:
            # Execute handler - now traced via auto-instrumentation
            result = handler.handle(state, parts)

            # Apply middleware after handling
            final_result = apply_after_handle(
                self._middleware, state, query_type, result
            )

            # Log successful execution
            duration = time.time() - start_time
            self._log_handler_execution(
                query_type, handler, parts, state, duration, final_result, user_context
            )

            return final_result

        except Exception as e:
            # Log error execution
            duration = time.time() - start_time
            error_result = HandlerResult(
                response=f"Error executing {query_type}: {str(e)}", status="error"
            )

            self._log_handler_execution(
                query_type, handler, parts, state, duration, error_result, user_context
            )

            return error_result
        finally:
            try:
                if token is not None:
                    from opentelemetry import context as otel_context  # type: ignore

                    otel_context.detach(token)  # type: ignore
            except Exception:
                pass

    def _create_background_task(
        self,
        handler: Handler,
        state: InvocationState,
        query_type: str,
        parts: List[str],
    ) -> str:
        """Create a background task for handler execution."""
        # Get the background task middleware if it exists
        background_middleware = self._get_background_task_middleware()

        if background_middleware:
            task_id = background_middleware.create_background_task(
                handler, state, query_type, parts
            )
        else:
            # Fallback: return a simple task ID
            task_id = str(uuid.uuid4())

        # Log background task creation
        self._log_operation(
            "background_task_created",
            task_id=task_id,
            query_type=query_type,
            handler_type=type(handler).__name__,
            input_text=state.text,
            input_parts=parts,
            has_background_middleware=background_middleware is not None,
        )

        return task_id

    def _get_background_task_middleware(self):
        """Get the background task middleware if it exists."""
        for middleware in self._middleware:
            if hasattr(middleware, "create_background_task"):
                return middleware
        return None

    def get_background_task_info(self, task_id: str) -> Optional[BackgroundTaskInfo]:
        """Get information about a specific background task."""
        return apply_get_background_task_info(self._middleware, task_id)

    def get_all_background_tasks(self) -> List[BackgroundTaskInfo]:
        """Get all background tasks."""
        return apply_get_all_background_tasks(self._middleware)

    def get_background_tasks_by_status(self, status: str) -> List[BackgroundTaskInfo]:
        """Get background tasks filtered by status."""
        return apply_get_background_tasks_by_status(self._middleware, status)

    def get_running_background_tasks(self) -> List[BackgroundTaskInfo]:
        """Get all currently running background tasks."""
        return self.get_background_tasks_by_status("running")

    def get_completed_background_tasks(self) -> List[BackgroundTaskInfo]:
        """Get all completed background tasks."""
        return self.get_background_tasks_by_status("completed")

    def get_failed_background_tasks(self) -> List[BackgroundTaskInfo]:
        """Get all failed background tasks."""
        return self.get_background_tasks_by_status("failed")
