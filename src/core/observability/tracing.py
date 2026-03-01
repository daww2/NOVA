"""Langfuse tracing setup.

Provides a configured ``observe`` decorator and a Langfuse client singleton.
When Langfuse is disabled (keys missing or ``LANGFUSE_ENABLED=False``), the
``observe`` decorator becomes a transparent no-op so instrumented code keeps
working without any runtime cost.
"""

import logging
from functools import wraps
from typing import Any, Callable
from langfuse import Langfuse
from src.config import settings
from langfuse import observe as _lf_observe


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singleton Langfuse client
# ---------------------------------------------------------------------------
langfuse_client = None
_langfuse_available = False

_cfg = settings.langfuse
if _cfg.enabled and _cfg.is_configured:
    try:

        langfuse_client = Langfuse(
            secret_key=_cfg.secret_key,
            public_key=_cfg.public_key,
            host=_cfg.host,
        )
        _langfuse_available = True
        logger.info("Langfuse tracing enabled (host=%s)", _cfg.host)
    except Exception as exc:
        logger.warning("Langfuse init failed — tracing disabled: %s", exc)
else:
    reason = "disabled by config" if not _cfg.enabled else "keys not configured"
    logger.info("Langfuse tracing disabled (%s)", reason)


# ---------------------------------------------------------------------------
# observe() decorator — real or no-op
# ---------------------------------------------------------------------------
if _langfuse_available:

    def observe(**kwargs: Any) -> Callable:
        """Wrap ``langfuse.decorators.observe`` keeping the same API."""
        return _lf_observe(**kwargs)
else:
    def observe(**kwargs: Any) -> Callable:  # type: ignore[misc]
        """No-op decorator when Langfuse is unavailable."""
        def decorator(fn: Callable) -> Callable:
            @wraps(fn)
            async def async_wrapper(*a: Any, **kw: Any) -> Any:
                return await fn(*a, **kw)

            @wraps(fn)
            def sync_wrapper(*a: Any, **kw: Any) -> Any:
                return fn(*a, **kw)

            import asyncio
            return async_wrapper if asyncio.iscoroutinefunction(fn) else sync_wrapper
        return decorator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def flush_langfuse() -> None:
    """Flush pending Langfuse events (call on shutdown)."""
    if langfuse_client is not None:
        try:
            langfuse_client.flush()
            logger.info("Langfuse events flushed")
        except Exception as exc:
            logger.warning("Langfuse flush failed: %s", exc)


def shutdown_langfuse() -> None:
    """Flush and shut down the Langfuse client."""
    if langfuse_client is not None:
        try:
            langfuse_client.shutdown()
            logger.info("Langfuse client shut down")
        except Exception as exc:
            logger.warning("Langfuse shutdown failed: %s", exc)
