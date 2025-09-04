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
    ) -> None:
        self._classifier = classifier
        self._handlers = handlers
        self._default_route = default_route
        self._memory = memory
        self._middleware = list(middleware or [])
        self._weave_config = weave_config

        # Initialize logger for Weave tracking
        self._logger = get_logger("struktx-engine")
        self._weave_available = self._logger.is_weave_available()

        if self._weave_available:
            self._logger.info("Engine initialized with Weave logging enabled")
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

            # Friendly op name with StruktX namespace and contextual display name
            op_base_name = f"StruktX.Engine.{operation_name}"

            @weave.op(name=op_base_name, call_display_name=contextual_name)
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

    def run(self, state: InvocationState) -> List[HandlerResult]:
        # Wrap the entire engine run in a Weave operation
        if self._weave_available:
            try:
                import weave

                # Extract user context
                user_context = (
                    self._extract_user_context(state)
                    if self._should_auto_track_user_context()
                    else {}
                )

                # Create contextual operation name for engine run
                base_op_name = "StruktX.Engine.run"
                contextual_name = self._create_contextual_operation_name(
                    base_op_name, user_context
                )

                @weave.op(name=base_op_name, call_display_name=contextual_name)
                def _execute_engine_run(engine_obj, state_obj):
                    start_time = time.time()
                    run_id = str(uuid.uuid4())
                    results = engine_obj._run_with_context(
                        state_obj, start_time, run_id, user_context
                    )

                    # Extract LLM responses and structured outputs for the main trace
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

                    return results, llm_responses, structured_outputs

                with weave.attributes(
                    {
                        "engine_component": "struktx_engine",
                        "context": user_context,
                        "input_text": state.text,
                        "input_context": state.context,
                    }
                ):
                    results, llm_responses, structured_outputs = _execute_engine_run(
                        self, state
                    )

                    # Add LLM outputs and structured data to the main trace attributes
                    if llm_responses or structured_outputs:
                        import weave

                        attributes = {
                            "llm_responses": llm_responses,
                            "llm_response_count": len(llm_responses),
                        }
                        if structured_outputs:
                            attributes["structured_outputs"] = structured_outputs
                            attributes["structured_output_count"] = len(
                                structured_outputs
                            )

                        with weave.attributes(attributes):
                            pass  # This adds the attributes to the current trace

                    return results
            except Exception:
                # Fallback to direct execution if Weave fails
                pass

        # Direct execution fallback
        start_time = time.time()
        run_id = str(uuid.uuid4())
        user_context = (
            self._extract_user_context(state)
            if self._should_auto_track_user_context()
            else None
        )
        return self._run_with_context(state, start_time, run_id, user_context)

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

        # Execute classification with Weave logging wrapper
        if self._weave_available:
            try:
                import weave

                # Extract user context for classifier operation
                user_context = {}
                if hasattr(state, "context") and state.context:
                    user_context = dict(state.context)

                # Create contextual operation name for classifier
                classifier_name = type(self._classifier).__name__
                base_op_name = f"StruktX.Classifier.{classifier_name}.classify"
                contextual_name = self._create_contextual_operation_name(
                    base_op_name, user_context
                )

                @weave.op(name=base_op_name, call_display_name=contextual_name)
                def _execute_classifier_op(classifier_obj, state_obj):
                    return classifier_obj.classify(state_obj)

                with weave.attributes(
                    {
                        "classifier_type": classifier_name,
                        "context": user_context,
                        "input_text": state.text,
                    }
                ):
                    classification: QueryClassification = _execute_classifier_op(
                        self._classifier, state
                    )
            except Exception:
                # Fallback to direct execution if Weave fails
                classification: QueryClassification = self._classifier.classify(state)
        else:
            # Execute classifier directly if Weave not available
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

        self._log_operation(
            "grouped_handlers_start",
            user_context=user_context,
            execution_id=execution_id,
            query_types=list(grouped.keys()),
            total_parts=sum(len(parts) for parts in grouped.values()),
            input_text=state.text,
        )

        results: List[HandlerResult] = []
        background_tasks = []
        normal_tasks = []

        for qtype, parts in grouped.items():
            handler = self._handlers.get(qtype) or fallback
            if handler is None:
                self._logger.warn(f"No handler found for query type: {qtype}")
                continue

            # Apply middleware before handling
            state, parts = apply_before_handle(self._middleware, state, qtype, parts)

            # Check if middleware wants to run this in background
            if apply_should_run_background(self._middleware, state, qtype, parts):
                # Get background message from middleware
                background_message = apply_get_background_message(
                    self._middleware, state, qtype, parts
                )

                # Get custom return query type from middleware
                return_query_type = apply_get_return_query_type(
                    self._middleware, state, qtype, parts
                )

                # Create background task and return immediate response
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
                # Add to normal tasks for parallel execution
                normal_tasks.append((qtype, parts, handler))

        # Execute normal tasks in parallel (regardless of background tasks)
        if normal_tasks:
            parallel_results = self._execute_handlers_parallel(
                state, normal_tasks, user_context
            )
            results.extend(parallel_results)

        # Log execution completion
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

    def _execute_handlers_parallel(
        self,
        state: InvocationState,
        tasks: List[tuple[str, List[str], Handler]],
        user_context: dict | None = None,
    ) -> List[HandlerResult]:
        """Execute multiple handlers in parallel."""
        start_time = time.time()
        parallel_id = str(uuid.uuid4())

        self._log_operation(
            "parallel_execution_start",
            user_context=user_context,
            parallel_id=parallel_id,
            task_count=len(tasks),
            input_text=state.text,
        )

        results: List[HandlerResult] = []

        # Use ThreadPoolExecutor for parallel execution
        with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
            # Submit all tasks
            future_to_task = {}
            for qtype, parts, handler in tasks:
                future = executor.submit(
                    self._execute_single_handler,
                    state,
                    qtype,
                    parts,
                    handler,
                    user_context,
                )
                future_to_task[future] = (qtype, parts, handler)

            # Collect results as they complete
            for future in as_completed(future_to_task):
                qtype, parts, handler = future_to_task[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    # Handle any exceptions from parallel execution
                    error_result = HandlerResult(
                        response=f"Error executing {qtype}: {str(e)}",
                        status="error",
                    )
                    results.append(error_result)

                    # Log parallel execution error
                    self._log_operation(
                        "parallel_execution_error",
                        parallel_id=parallel_id,
                        query_type=qtype,
                        error=str(e),
                        error_type=type(e).__name__,
                    )

        # Log parallel execution completion
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

    def _execute_single_handler(
        self,
        state: InvocationState,
        query_type: str,
        parts: List[str],
        handler: Handler,
        user_context: dict | None = None,
    ) -> HandlerResult:
        """Execute a single handler and apply middleware."""
        start_time = time.time()

        try:
            # Execute handler with Weave logging wrapper
            if self._weave_available:
                try:
                    import weave

                    # Extract user context for handler operation
                    user_context = {}
                    if hasattr(state, "context") and state.context:
                        user_context = dict(state.context)

                    # Create contextual operation name for handler
                    handler_name = type(handler).__name__
                    base_op_name = f"StruktX.Handler.{handler_name}.handle"
                    contextual_name = self._create_contextual_operation_name(
                        base_op_name, user_context
                    )

                    @weave.op(name=base_op_name, call_display_name=contextual_name)
                    def _execute_handler_op(handler_obj, state_obj, parts_obj):
                        return handler_obj.handle(state_obj, parts_obj)

                    with weave.attributes(
                        {
                            "handler_type": handler_name,
                            "query_type": query_type,
                            "context": user_context,
                            "input_parts": parts,
                            "input_text": state.text,
                        }
                    ):
                        result = _execute_handler_op(handler, state, parts)
                except Exception:
                    # Fallback to direct execution if Weave fails
                    result = handler.handle(state, parts)
            else:
                # Execute handler directly if Weave not available
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
