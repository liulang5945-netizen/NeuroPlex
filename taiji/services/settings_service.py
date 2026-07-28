"""Settings service stub — not yet implemented.

This is a minimal stub that allows imports to succeed. Settings are
kept in an in-process dict; the real implementation will persist them
to disk and broadcast changes to subscribers.
"""
import logging

logger = logging.getLogger(__name__)

_settings = {}


def get_setting(key, default=None):
    return _settings.get(key, default)


def load_settings():
    return dict(_settings)


def save_settings(data):
    global _settings
    _settings = dict(data)


def update_settings(updates):
    _settings.update(updates)
