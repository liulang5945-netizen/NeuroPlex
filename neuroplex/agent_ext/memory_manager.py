"""Memory manager stub — not yet implemented.

This is a minimal stub that allows imports to succeed. The real
implementation will back the agent's short-term / long-term memory
stores with persistent storage.
"""
import logging

logger = logging.getLogger(__name__)


class _MemoryStub:
    """Stub memory manager. All operations are no-ops returning safe defaults."""

    def get_status(self):
        return {"enabled": False, "message": "memory_manager not implemented"}

    def get_context(self, last_n=20):
        return []

    def add_message(self, role, content):
        pass

    def clear(self):
        pass


memory = _MemoryStub()
