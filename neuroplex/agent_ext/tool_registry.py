"""Tool registry stub — not yet implemented.

This is a minimal stub that allows imports to succeed. The real
implementation will hold a process-wide registry of tool definitions
and dispatch execution to their handlers.
"""
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ToolDef:
    """Declarative tool definition."""

    name: str = ""
    description: str = ""
    parameters: dict = None
    handler: Callable = None


class _RegistryStub:
    """Stub registry. Listings return empty, execute raises."""

    def list_tools(self):
        return []

    def get_tool(self, name):
        return None

    def register(self, tool):
        pass

    def execute(self, name, **kwargs):
        raise NotImplementedError


registry = _RegistryStub()
