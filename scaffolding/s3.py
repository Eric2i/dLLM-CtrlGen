"""
Implementation of the Self-adaptive Schema Scaffolding (S3) method.

This component prepares prompts and template scaffolds that warm-start
the diffusion LLM from a partially denoised state, while guiding the
model to emit `null` placeholders for missing values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

from transformers import PreTrainedTokenizerBase

from .schema import SchemaTemplate, build_schema_template


DEFAULT_FIELD_BUDGET: int = 16
"""Fallback token budget allocated to fields without explicit overrides."""


@dataclass
class SelfAdaptiveSchemaConfig:
    """Configuration for building the S3 schema scaffold."""

    fields: Sequence[str] = ()
    """Ordered list of fields to appear in the JSON output."""

    token_budgets: Optional[Dict[str, int]] = None
    """Optional per-field token budgets; defaults to :data:`DEFAULT_FIELD_BUDGET`."""

    mask_token: str = "<|mdm_mask|>"
    """Mask token used by the diffusion model."""

    include_codeblock: bool = True
    """Whether to wrap the scaffold in ```json ... ``` markers."""

    null_token: str = "<none>"
    """Consensus null token used in prompts and post-processing."""

    prompt_template: str = (
        "Extract the following fields from the text and return ONLY a JSON object with keys "
        "{fields}. If a field is unavailable in the text, output the literal token '{null}' "
        "in place of its value. You may repeat '{null}' to indicate confidence (e.g., "
        "'{null}{null}{null}').\n\nText: {text}"
    )

    @property
    def resolved_token_budgets(self) -> Dict[str, int]:
        overrides = self.token_budgets or {}
        return {field: overrides.get(field, DEFAULT_FIELD_BUDGET) for field in self.fields}


class SelfAdaptiveSchemaScaffolder:
    """High-level builder that exposes prompt and scaffold construction."""

    def __init__(self, config: Optional[SelfAdaptiveSchemaConfig] = None):
        self.config = config or SelfAdaptiveSchemaConfig()

    def make_prompt(self, text: str) -> str:
        """Render the S3 instruction prompt for the provided passage."""
        return self.config.prompt_template.format(
            fields=", ".join(self.config.fields),
            null=self.config.null_token,
            text=text,
        )

    def build_template(self, tokenizer: PreTrainedTokenizerBase) -> SchemaTemplate:
        """Construct the schema scaffold as a :class:`SchemaTemplate`."""
        budgets = self.config.resolved_token_budgets
        field_pairs = [(field, budgets[field]) for field in self.config.fields]

        return build_schema_template(
            tokenizer=tokenizer,
            fields=field_pairs,
            mask_token=self.config.mask_token,
            include_codeblock=self.config.include_codeblock,
        )
