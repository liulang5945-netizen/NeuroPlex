"""Data collector stub — not yet implemented.

This is a minimal stub that allows imports to succeed. The real
implementation will collect ReAct / conversation traces for
fine-tuning pipelines.
"""
import logging

logger = logging.getLogger(__name__)


class DataCollector:
    """Stub data collector. Returns empty training data."""

    def load_as_training_data(self):
        """Return a (react_data, conv_data) tuple of empty lists."""
        return [], []


_collector = None


def get_collector():
    """Return a process-wide singleton DataCollector (stub)."""
    global _collector
    if _collector is None:
        _collector = DataCollector()
    return _collector
