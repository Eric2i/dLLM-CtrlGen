"""
Model loading utilities for diffusion LLM controllable generation.

The paper relies on the instruction-tuned LLaDA diffusion LLM.
This helper centralises device selection and dtype management so the
generation pipeline can depend on a single entry point.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from transformers import AutoModel, AutoTokenizer


@dataclass
class ModelLoadConfig:
    """Configuration for loading a diffusion LLM."""

    model_id: str = "GSAI-ML/LLaDA-1.5"
    """Hugging Face model identifier."""

    dtype: torch.dtype = torch.bfloat16
    """Desired torch dtype for the model weights."""

    trust_remote_code: bool = True
    """Whether to trust custom modelling code provided by the checkpoint."""


def _detect_device(preferred: Optional[str] = None) -> torch.device:
    """
    Resolve the target torch device.

    Args:
        preferred: Optional user-specified device string.
    """
    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_diffusion_llm(
    config: Optional[ModelLoadConfig] = None, device: Optional[str] = None
) -> Tuple[torch.nn.Module, AutoTokenizer, torch.device]:
    """
    Load the diffusion LLM and corresponding tokenizer.

    Args:
        config: Optional configuration overriding default LLaDA settings.
        device: Optional preferred device; falls back to auto-detection.

    Returns:
        model, tokenizer, resolved_device
    """
    cfg = config or ModelLoadConfig()
    resolved_device = _detect_device(device)

    model = AutoModel.from_pretrained(
        cfg.model_id,
        trust_remote_code=cfg.trust_remote_code,
        torch_dtype=cfg.dtype,
    ).to(resolved_device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_id,
        trust_remote_code=cfg.trust_remote_code,
    )

    return model, tokenizer, resolved_device
