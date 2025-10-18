"""
Decoding utilities for diffusion large language models.

This submodule implements the top-K remasking scheduler and the
self-adaptive generation loop used by S3.
"""

from .generator import SelfAdaptiveGenerator, GenerationOutput  # noqa: F401
