# RunPod validation gate — ARACHNE-X-ULTRA-V3 NIGHTCORE

Manual gate **before** tagging `v3-nightcore`. Local Windows dev machine: CPU contract tests only.

## Pre-flight (each pod)

```bash
export NULLXES_PRODUCTION=1
export NULLXES_CHECKPOINT_DIR=/workspace/weights/arachne-avatar-runtime
python scripts/prod_doctor.py --role worker --checkpoint-dir "$NULLXES_CHECKPOINT_DIR"
python -m arachne_x.weights_resolve --doctor "$NULLXES_CHECKPOINT_DIR"
```

## Matrix

| Tier | VRAM | Avatar infer @ 480p operational | Cinematic 720p | Train/LoRA smoke | Infer reject |
|------|------|----------------------------------|----------------|------------------|--------------|
| H200 | >85GB | Required green | Optional doc | — | — |
| H100 80GB | ~80GB | Required green | Skip | — | — |
| A100 80GB | ~80GB | Required green | Skip | — | — |
| A100 40GB | ≤45GB | Must fail `get_avatar_pipeline()` | — | `train_lora_avatar` smoke | Explicit error |

## Eval command (80GB tiers)

```bash
python scripts/gpu/eval_stability_bench.py \
  --tier 80gb \
  --image assets/avatar/ref.jpg \
  --audio assets/audio/speech.wav \
  --output_dir /tmp/arachne_eval_80gb
```

## Eval command (H200 optional cinematic)

```bash
python scripts/gpu/eval_stability_bench.py \
  --tier h200 \
  --image assets/avatar/ref.jpg \
  --audio assets/audio/speech.wav \
  --output_dir /tmp/arachne_eval_h200
```

## Artifacts to attach to release

1. `eval_stability_report.json` per tier
2. One `.run.json` with `"streaming_mode": "chunked_ai2v"` (not `legacy_monolithic`)
3. `eval/eval_baseline.json` updated with checkpoint SHA + `ARACHNE_INFER_ENABLE_BSA` value used

## Rollback

Pin previous checkpoint SHA in `eval/eval_baseline.json` and redeploy worker with prior weights snapshot.
