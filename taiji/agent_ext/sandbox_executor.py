"""Sandbox executor stub — not yet implemented.

This is a minimal stub that allows imports to succeed. The real
implementation will run Python code inside a hardened sandbox
(files provided via a virtual filesystem).
"""
import logging

logger = logging.getLogger(__name__)


def execute_python_with_files(files, code):
    """Stub — executes Python code with a set of virtual files.

    Raises NotImplementedError until the real sandbox lands.
    """
    raise NotImplementedError("Sandbox executor not yet implemented")


def execute_python_code_safe(code):
    """Stub — executes Python code in a safe sandbox.

    Raises NotImplementedError until the real sandbox lands.
    """
    raise NotImplementedError("Sandbox executor not yet implemented")
