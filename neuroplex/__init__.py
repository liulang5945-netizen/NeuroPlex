"""Legacy NeuroPlex Transformer runtime.

The top-level :mod:`taiji` namespace is now owned by the independent native
Taiji architecture.  Historical checkpoints that serialized ``taiji.*``
class paths require an explicit conversion environment; importing NeuroPlex
must never shadow the new architecture globally.
"""
