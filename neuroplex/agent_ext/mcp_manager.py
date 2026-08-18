"""MCP manager stub — not yet implemented.

This is a minimal stub that allows imports to succeed. The real
implementation will register / discover Model Context Protocol
servers and expose their tools to the agent runtime.
"""
import logging

logger = logging.getLogger(__name__)


class _MCPManagerStub:
    """Stub MCP manager. Listings return empty, mutations raise."""

    def list_servers(self):
        return []

    def get_server(self, name):
        return None

    def add_server(self, *args, **kwargs):
        raise NotImplementedError

    def remove_server(self, name):
        raise NotImplementedError

    def list_tools(self, server_name=None):
        return []


mcp_manager = _MCPManagerStub()
