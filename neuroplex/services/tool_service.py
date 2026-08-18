"""Tool service stub — not yet implemented.

This is a minimal stub that allows imports to succeed. The real
implementation will bridge the agent_ext tool registry to the API
layer, exposing tool schemas and dispatching execution.
"""
import logging

logger = logging.getLogger(__name__)


def list_tools():
    return []


def get_registry_schemas():
    return {}


def execute_tool(name, args):
    raise NotImplementedError(f"Tool '{name}' not yet implemented")
