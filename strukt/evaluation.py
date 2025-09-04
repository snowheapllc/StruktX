from __future__ import annotations

from typing import Any, Dict, List, Optional

from .config import EvaluationConfig


def _weave_available() -> bool:
    try:
        import weave  # noqa: F401

        return True
    except Exception:
        return False


def log_post_run_evaluation(
    cfg: EvaluationConfig,
    *,
    input_text: str,
    input_context: Dict[str, Any] | None,
    results: List[Any],
    display_name: Optional[str] = None,
) -> None:
    """Lightweight post-run evaluation/logging.

    - No-op unless cfg.enabled is True and weave is available.
    - If built-in scorer toggles are enabled, attaches minimal summary metrics.
    - Avoids heavy token use; does not invoke LLM-based scorers unless explicitly
      configured in the future.
    """
    if not (cfg and getattr(cfg, "enabled", False)):
        return
    if not _weave_available():
        return
    try:
        import weave

        outputs = []
        for r in results:
            try:
                val = getattr(r, "response", None)
                outputs.append(val)
            except Exception:
                outputs.append(None)

        def _post_run(attrs: dict) -> dict:  # type: ignore[no-redef]
            return attrs

        metrics: dict[str, Any] = {
            "output_count": len(outputs),
            "empty_outputs": len([o for o in outputs if not o]),
        }

        # Built-in scorers (safe and cheap only)
        if cfg.use_valid_json_scorer:
            import json

            json_ok = 0
            for o in outputs:
                try:
                    if o:
                        json.loads(o)
                        json_ok += 1
                except Exception:
                    pass
            metrics["json_valid_true_fraction"] = (
                float(json_ok) / float(len(outputs)) if outputs else 0.0
            )

        # OpenAI moderation and embedding similarity are intentionally not
        # executed here to avoid unexpected external calls.

        attrs = {
            "evaluation_component": "post_run_logger",
            "input_text": input_text,
            "input_context": input_context or {},
            "outputs": outputs,
            "metrics": metrics,
        }
        with weave.attributes(attrs):
            _post_run(attrs)
    except Exception:
        # Silently degrade; evaluation should never affect primary run
        return
