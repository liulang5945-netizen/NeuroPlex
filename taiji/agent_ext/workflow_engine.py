"""Workflow engine stub — not yet implemented.

This is a minimal stub that allows imports to succeed. It is imported
at module top-level by ``api.routes_workflows``, so all classes here
must be importable and instantiable without side effects.

The real implementation will execute multi-step workflows against the
agent runtime and persist them via WorkflowStore.
"""
import logging
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class WorkflowDefinition:
    """A declarative workflow definition.

    The dataclass is fully usable today; the real execution semantics
    land with WorkflowEngine.execute().
    """

    id: str = ""
    name: str = ""
    description: str = ""
    steps: list = field(default_factory=list)


class WorkflowEngine:
    """Stub workflow engine. Execution raises NotImplementedError."""

    def execute(self, workflow, **kwargs):
        raise NotImplementedError("WorkflowEngine not yet implemented")


class WorkflowStore:
    """Stub workflow store. Listings return empty, mutations raise."""

    def list_all(self):
        return []

    def load(self, workflow_id):
        return None

    def save(self, workflow):
        raise NotImplementedError

    def delete(self, workflow_id):
        raise NotImplementedError
