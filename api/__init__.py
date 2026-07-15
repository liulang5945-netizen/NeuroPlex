"""Taiji API package."""

__all__ = ["app", "create_app", "get_startup_download_progress"]


def __getattr__(name: str):
    if name in __all__:
        from .app import app, create_app, get_startup_download_progress

        exports = {
            "app": app,
            "create_app": create_app,
            "get_startup_download_progress": get_startup_download_progress,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
