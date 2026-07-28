"""Agent workspace stub — not yet implemented.

This is a minimal stub that allows imports to succeed. The real
implementation will back project creation, code analysis, dependency
management, and plan/context persistence for the agent workspace.
"""
import logging

logger = logging.getLogger(__name__)


def create_project(*args, **kwargs):
    raise NotImplementedError("agent.create_project not yet implemented")


def analyze_code(*args, **kwargs):
    raise NotImplementedError("agent.analyze_code not yet implemented")


def install_dependency(*args, **kwargs):
    raise NotImplementedError("agent.install_dependency not yet implemented")


def list_plans(*args, **kwargs):
    return []


def load_context(*args, **kwargs):
    return {}


def save_context(*args, **kwargs):
    pass
