# Prompt Compiler (deterministic template layer)

Maps user intent to UMT5-ready strings **before** `encode_prompt`. Does not
replace UMT5 or DiT weights. External OpenAI/Gemma expansion backends are not
shipped in this runtime; the only supported backend is deterministic `off`.

## Flow

```
user_text → compile_avatar_turn() → positive/negative strings → UMT5 → DiT
```

| Backend | Where | Requires |
|---------|-------|----------|
| `off` | Default / shipped | Avatar template merge only (lipsync, static camera) |

## Environment

| Variable | Default | Purpose |
|----------|---------|---------|
| `ARACHNE_PROMPT_COMPILER` | `off` | Default backend if CLI flag omitted |
| `ARACHNE_COMPILER_FALLBACK` | `off` | Reserved for compatibility; only `off` is shipped |

Legacy `ARACHNE_PROMPT_ENHANCER_ENABLED=1` is no longer part of the shipped compiler path.

## CLI

```bash
# Baseline (templates only when prompt empty; merge defaults on avatar modes)
python scripts/infer.py --mode ai2v --prompt_compiler off ...
```

Worker job: omit `promptCompiler` or set `"promptCompiler": "off"`.

## Latent export (train ≡ infer)

```bash
python scripts/export_latent_training_sample.py \
  --prompt_compiler off \
  --prompt "$(cat pair5.txt)" ...
```

## Code layout

- `arachne_x/prompt_compiler/` — compile API
- `arachne_x/runtime/prompt_compiler_runtime.py` — infer/serving wire-up
