"""Runtime service stub — not yet implemented.

This is a minimal stub that allows imports to succeed. The real
implementation will report on the model runtime, bootstrap progress,
and hardware readiness for the API layer.
"""
import logging

logger = logging.getLogger(__name__)


def get_runtime_status():
    return {"status": "stub", "message": "runtime_service not implemented"}


def get_bootstrap_status():
    return {"status": "stub", "message": "runtime_service not implemented"}
