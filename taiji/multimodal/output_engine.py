"""Multimodal output engine stub — not yet implemented.

This is a minimal stub that allows imports to succeed. It is imported
by ``api.routes_multimodal``, so the class below must be importable
and instantiable without side effects.

The real implementation will generate images / audio / video and
describe images via the multimodal ensemble.
"""
import logging

logger = logging.getLogger(__name__)


class MultimodalOutputEngine:
    """Stub. Real implementation will generate images/audio/video."""

    def __init__(self, *args, **kwargs):
        pass

    def generate_image(self, prompt, **kwargs):
        raise NotImplementedError("Image generation not yet implemented")

    def generate_audio(self, text, **kwargs):
        raise NotImplementedError("Audio generation not yet implemented")

    def describe_image(self, image_path, **kwargs):
        raise NotImplementedError("Image description not yet implemented")
