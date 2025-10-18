"""
Self-adaptive generation loop for diffusion LLMs with schema scaffolding.

The implementation follows the top-K remasking strategy proposed in the
paper, exposing a small interface that accepts a schema template and
returns the denoised response sequence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerBase

from scaffolding.schema import SchemaTemplate


@dataclass
class GenerationConfig:
    """Hyperparameters controlling the denoising schedule."""

    steps: int = 16
    temperature: float = 0.0
    cfg_scale: float = 0.0
    topk_remask: Optional[int] = None


@dataclass
class TraceStep:
    """Trace information for visualising the denoising process."""

    step: int
    revealed_indices: Sequence[int]
    revealed_tokens: Sequence[str]


@dataclass
class GenerationOutput:
    """Structured output returned by :class:`SelfAdaptiveGenerator`."""

    text: str
    token_ids: List[int]
    steps_executed: int
    trace: List[TraceStep] = field(default_factory=list)


def _apply_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0.0:
        return logits
    noise = torch.rand_like(logits)
    noise = noise.clamp_min(1e-10)
    gumbel = -torch.log(-torch.log(noise))
    return logits / temperature + gumbel


class SelfAdaptiveGenerator:
    """Runs the S3 denoising loop with top-K remasking."""

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: PreTrainedTokenizerBase,
        device: torch.device,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def _build_messages(self, prompt: str) -> List[Dict[str, str]]:
        return [{"role": "user", "content": prompt}]

    def _encode_prompt(self, prompt: str) -> torch.Tensor:
        chat_ids = self.tokenizer.apply_chat_template(
            self._build_messages(prompt),
            add_generation_prompt=True,
            tokenize=True,
        )
        return torch.tensor(chat_ids, dtype=torch.long, device=self.device).unsqueeze(0)

    def _initialise_sequence(
        self, prompt_ids: torch.Tensor, template: SchemaTemplate
    ) -> torch.Tensor:
        gen_length = len(template.tokens)
        seq = torch.full(
            (1, prompt_ids.shape[1] + gen_length),
            fill_value=template.mask_token_id,
            dtype=torch.long,
            device=self.device,
        )
        seq[:, : prompt_ids.shape[1]] = prompt_ids
        seq[:, prompt_ids.shape[1] :] = torch.tensor(
            template.tokens, dtype=torch.long, device=self.device
        )
        return seq

    @staticmethod
    def _resolve_budget(
        config: GenerationConfig, initial_variable_count: int
    ) -> int:
        if config.topk_remask is not None and config.topk_remask > 0:
            return config.topk_remask
        return max(1, math.ceil(initial_variable_count / config.steps))

    def _forward(
        self,
        x: torch.Tensor,
        prompt_mask: torch.Tensor,
        template: SchemaTemplate,
        cfg_scale: float,
    ) -> torch.Tensor:
        if cfg_scale <= 0.0:
            return self.model(x).logits

        uncond = x.clone()
        uncond[prompt_mask] = template.mask_token_id
        model_input = torch.cat([x, uncond], dim=0)
        logits = self.model(model_input).logits
        cond_logits, uncond_logits = logits.chunk(2, dim=0)
        return uncond_logits + (cfg_scale + 1.0) * (cond_logits - uncond_logits)

    def generate(
        self,
        prompt: str,
        template: SchemaTemplate,
        config: Optional[GenerationConfig] = None,
        trace: bool = False,
    ) -> GenerationOutput:
        cfg = config or GenerationConfig()
        prompt_ids = self._encode_prompt(prompt)
        sequence = self._initialise_sequence(prompt_ids, template)

        prompt_length = prompt_ids.shape[1]
        prompt_mask = torch.zeros_like(sequence, dtype=torch.bool, device=self.device)
        prompt_mask[:, :prompt_length] = True

        variable_positions = [
            prompt_length + position for segment in template.field_segments for position in segment.value_positions
        ]
        initial_variable_count = len(variable_positions)
        budget = self._resolve_budget(cfg, initial_variable_count)

        generation_trace: List[TraceStep] = []

        for step in range(cfg.steps):
            mask_positions = (sequence == template.mask_token_id)
            mask_positions[:, :prompt_length] = False
            mask_indices = torch.nonzero(mask_positions[0], as_tuple=False).squeeze(-1)

            if mask_indices.numel() == 0:
                executed_steps = step
                break

            logits = self._forward(sequence, prompt_mask, template, cfg.cfg_scale)
            logits = _apply_gumbel_noise(logits, cfg.temperature)
            predictions = torch.argmax(logits, dim=-1)

            probabilities = F.softmax(logits, dim=-1)
            confidences = probabilities.gather(
                -1, predictions.unsqueeze(-1)
            ).squeeze(-1)

            mask_conf = confidences[0, mask_indices]

            remaining = mask_indices.numel()
            remaining_steps = cfg.steps - step
            if remaining_steps <= 1:
                k = remaining
            else:
                k = min(budget, remaining)

            topk = torch.topk(mask_conf, k)
            selected = mask_indices[topk.indices]

            sequence[0, selected] = predictions[0, selected]

            if trace:
                revealed_tokens = [
                    self.tokenizer.decode(
                        [int(sequence[0, idx].item())],
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    )
                    for idx in selected.tolist()
                ]
                generation_trace.append(
                    TraceStep(
                        step=step,
                        revealed_indices=selected.tolist(),
                        revealed_tokens=revealed_tokens,
                    )
                )
        else:
            executed_steps = cfg.steps

        response_tokens = sequence[0, prompt_length:]
        text = self.tokenizer.decode(
            response_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        return GenerationOutput(
            text=text,
            token_ids=response_tokens.tolist(),
            steps_executed=executed_steps,
            trace=generation_trace,
        )
