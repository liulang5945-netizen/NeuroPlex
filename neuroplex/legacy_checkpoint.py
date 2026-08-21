"""Scoped loader for checkpoints created under the historical ``taiji`` alias.

This module belongs to the frozen NeuroPlex runtime.  It never installs a
process-wide alias at import time, so the native top-level :mod:`taiji` package
keeps its identity during normal execution.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import sys
from threading import RLock
from typing import Any, Iterator

import torch


_ALIAS_LOCK = RLock()


@contextmanager
def historical_taiji_namespace() -> Iterator[None]:
    """Temporarily map old pickle imports to NeuroPlex, then fully restore."""

    with _ALIAS_LOCK:
        saved = {
            name: module
            for name, module in tuple(sys.modules.items())
            if name == "taiji" or name.startswith("taiji.")
        }
        for name in saved:
            sys.modules.pop(name, None)
        sys.modules["taiji"] = sys.modules["neuroplex"]
        try:
            yield
        finally:
            for name in tuple(sys.modules):
                if name == "taiji" or name.startswith("taiji."):
                    sys.modules.pop(name, None)
            sys.modules.update(saved)


def load_legacy_checkpoint(
    path: str | Path,
    *,
    map_location: Any = "cpu",
) -> Any:
    """Load an unsafe historical pickle inside the compatibility namespace."""

    with historical_taiji_namespace():
        return torch.load(path, map_location=map_location, weights_only=False)
