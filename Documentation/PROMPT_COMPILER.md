# Prompt Compiler (LTX-style instruction layer)

Maps short user intent to UMT5-ready strings **before** `encode_prompt`. Does not replace UMT5 or DiT weights.

## Flow

```
user_text → compile_avatar_turn() → positive/negative strings → UMT5 → DiT
```

| Backend | Where | Requires |
|---------|-------|----------|
| `off` | Default | Avatar template merge only (lipsync, static camera) |
| `openai` | Production | `OPENAI_API_KEY` |
| `gemma` | RunPod GPU | CUDA + `ARACHNE_GEMMA_MODEL` |

## Environment

| Variable | Default | Purpose |
|----------|---------|---------|
| `ARACHNE_PROMPT_COMPILER` | `off` | Default backend if CLI flag omitted |
| `ARACHNE_GEMMA_MODEL` | `google/gemma-2-2b-it` | Local Gemma HF id |
| `ARACHNE_COMPILER_FALLBACK` | `off` | On Gemma failure: `openai` or `off` |

Legacy `ARACHNE_PROMPT_ENHANCER_ENABLED=1` only affects direct `prompt_enhancer` calls; compiler OpenAI path uses `force=True`.

## CLI

```bash
# Baseline (templates only when prompt empty; merge defaults on avatar modes)
python scripts/infer.py --mode ai2v --prompt_compiler off ...

# Production
export OPENAI_API_KEY=sk-...
python scripts/infer.py --mode ai2v --prompt_compiler openai \
  --image assets/avatar/single/elena/elena/prompt/5.png \
  --audio assets/avatar/single/elena/elena6_6s.wav \
  --prompt "$(cat assets/avatar/single/elena/elena/prompt/5.txt)" \
  --output out_openai.mp4

# RunPod Gemma
export ARACHNE_GEMMA_MODEL=google/gemma-2-2b-it
python scripts/infer.py --mode ai2v --prompt_compiler gemma \
  --prompt_compiler_fallback openai ...
```

Worker job (optional): `"promptCompiler": "openai"` in JSON — default `off`.

## Latent export (train ≡ infer)

```bash
python scripts/export_latent_training_sample.py \
  --prompt_compiler openai \
  --prompt "$(cat pair5.txt)" ...
```

## RunPod A/B (Elena pair 5)

Use the same `--image`, `--audio` (e.g. `elena6_6s.wav`), `--num_frames`, `--audio_guidance_scale 5.5`, and pair-5 prompt; only change `--prompt_compiler`:

```bash
bash scripts/ab_prompt_compiler_elena.sh /workspace/ARACHNE-X/weights/arachne-avatar-runtime
```

Compare lipsync and camera stability across `off`, `openai`, `gemma`.

## Code layout

- `arachne_x/prompt_compiler/` — compile API
- `arachne_x/runtime/prompt_compiler_runtime.py` — infer/serving wire-up
- `arachne_x/utils/prompt_enhancer.py` — OpenAI expansion (shared prompts)
