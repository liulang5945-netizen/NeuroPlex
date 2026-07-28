"""Memory watchdog stub — not yet implemented.

This is a minimal stub that allows imports to succeed. It is imported
at module top-level by ``taiji.tools.rag``, so the symbols below must
be importable without side effects AND the call signatures must match
rag.py's actual usage so runtime degrades gracefully.

Real implementation will monitor process memory usage and provide a
decorator for graceful degradation under memory pressure (triggering
GC, shedding load, or refusing new allocations). Until then, all
checks report "healthy" so callers proceed normally.
"""
import functools
import logging

logger = logging.getLogger(__name__)


class _StatusStub:
    """Stub status object. ``level`` stays at 0 (no pressure)."""

    level = 0


class MemoryWatchdog:
    """Stub. Real implementation will track RSS and trigger GC.

    All checks report a healthy state so callers proceed normally;
    this is the graceful-degradation path while the real watchdog
    is unavailable.
    """

    def __init__(self):
        self._threshold = 0.9
        self.status = _StatusStub()

    def check(self) -> bool:
        """Return True when memory pressure is detected.

        The stub never reports pressure.
        """
        return False

    def is_critical(self) -> bool:
        """Return True when memory pressure is critical.

        The stub never reports critical pressure.
        """
        return False

    def can_proceed(self, min_avail_pct: float = 0.0):
        """Return (can_proceed, message).

        The stub always allows the caller to proceed.
        """
        return True, "memory_watchdog stub: checks skipped"

    @classmethod
    def can_build_embeddings(cls, num_chunks: int, embed_dim: int):
        """Return (can_build, message).

        The stub always allows the build to proceed.
        """
        return True, "memory_watchdog stub: embedding build allowed"


def memory_guarded(func=None, *, min_avail_pct=None, on_critical=None):
    """Decorator stub — always calls func without memory check.

    Supports both usage forms:
      - ``@memory_guarded``                    (bare decorator)
      - ``@memory_guarded(min_avail_pct=..., on_critical=...)``  (factory)

    The real implementation will short-circuit or trigger cleanup when
    the watchdog reports critical memory pressure. The stub ignores the
    parameters and always invokes the wrapped function.
    """

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            return fn(*args, **kwargs)

        return wrapper

    if func is not None:
        # Used as @memory_guarded without parentheses.
        return decorator(func)
    # Used as @memory_guarded(...) with arguments.
    return decorator
