# dLLM-CtrlGen

[![arXiv](https://img.shields.io/badge/arXiv-2507.04504-b31b1b.svg)](https://arxiv.org/abs/2507.04504)
[![project page](https://img.shields.io/badge/Project-Page-green)](https://zhenxiong123.github.io/dLLM-CtrlGen/)

> **The Potential of Diffusion Large Language Model in Controllable Generation**
> 
> 🎉 Officially accepted by the ICRL 2026
>
> Zhen Xiong¹, Yujun Cai²˒³(*), Zhecheng Li⁴, Yiwei Wang⁵
>
>¹USC, ²UQ, ³Ant Group, ⁴UCSD, ⁵UC Merced

This toolbox implements **Self-adaptive Schema Scaffolding (S3)** for controllable generation with diffusion large language models (dLLMs).

## ✨ Highlights

- Build schema-aware scaffolds and prompts that warm-start diffusion LLM decoding.
- Run S3’s top-*K* remasking denoiser for reliable structured outputs.
- Inspect denoising traces and evaluation metrics with a few lines of Python.
- Customize the output schema (fields, token budgets, null tokens) without touching core code.

## 📦 Installation

```bash
git clone https://github.com/eric2i/dLLM-CtrlGen.git
cd dLLM-CtrlGen
```

## 🚀 Quick Start

### Customize the schema

```python
from scaffolding import SelfAdaptiveSchemaScaffolder, SelfAdaptiveSchemaConfig

schema_cfg = SelfAdaptiveSchemaConfig(
    fields=("name", "birth_place", "birth_date"),
)
scaffolder = SelfAdaptiveSchemaScaffolder(schema_cfg)
```

Each field receives a 16-token mask budget by default; override specific fields via
`token_budgets` when you need more or fewer diffusion steps.
The scaffolder also defaults the null token to `<none>`; adjust `null_token` if your
pipeline expects a different placeholder.

### Structured Generation

```python
from models import load_diffusion_llm
from decoding import SelfAdaptiveGenerator, GenerationConfig

model, tokenizer, device = load_diffusion_llm()
template = scaffolder.build_template(tokenizer)

text = "Albert Einstein was born on March 14, 1879, in Ulm, Germany..."
prompt = scaffolder.make_prompt(text)

generator = SelfAdaptiveGenerator(model, tokenizer, device)
result = generator.generate(prompt, template, config=GenerationConfig(steps=16), trace=True)

print(result.text)      # JSON-formatted string
print(result.steps_executed)
```

## 📊 Extending & Customizing

- Override denoising hyperparameters through `GenerationConfig`.
- Modify scaffold templates (code fences, indentation, mask budgets) by subclassing or configuring the scaffolder.

## 📑 Citation

Please cite the accompanying paper when using this implementation:

```
@article{xiong2025unveiling,
  title={Unveiling the Potential of Diffusion Large Language Model in Controllable Generation},
  author={Xiong, Zhen and Cai, Yujun and Li, Zhecheng and Wang, Yiwei},
  journal={arXiv preprint arXiv:2507.04504},
  year={2025}
}
```
