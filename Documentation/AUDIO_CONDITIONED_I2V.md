# Audio-Conditioned I2V (experimental)

Lab-only path: frozen base VIDEO DiT + trainable audio-conditioning adapter.

## Architecture hooks

| Layer | Base VIDEO (`dit/`) | Avatar (`avatar_single/`) | Audio I2V adapter |
| ----- | ------------------- | ------------------------- | ----------------- |
| DiT class | `LongCatVideoTransformer3DModel` | `LongCatVideoAvatarTransformer3DModel` | wraps frozen base DiT |
| Text cross-attn | `blocks.*.cross_attn` | same | unchanged (frozen) |
| Audio path | none | `audio_proj` + `blocks.*.audio_cross_attn` | external `AudioConditioningAdapter` |
| Audio CFG | none | pipeline-level null audio embedding | `audio_conditioning_scale` |

### Injection points (code)

- Base DiT forward loop: [`arachne_x/modules/arachne_video_dit.py`](../arachne_x/modules/arachne_video_dit.py) — blocks without audio.
- Avatar reference implementation: [`arachne_x/modules/avatar/arachne_avatar_dit.py`](../arachne_x/modules/avatar/arachne_avatar_dit.py) — `audio_proj`, `audio_cross_attn`, `audio_adaLN_modulation`.
- Adapter module: [`arachne_x/modules/audio_conditioning/adapter.py`](../arachne_x/modules/audio_conditioning/adapter.py).
- Wrapper (frozen base + inject after selected blocks): [`arachne_x/modules/audio_conditioning/wrapped_dit.py`](../arachne_x/modules/audio_conditioning/wrapped_dit.py).
- Pipeline: [`arachne_x/pipeline_audio_i2v.py`](../arachne_x/pipeline_audio_i2v.py).
- Loader: `load_audio_i2v_pipeline()` in [`arachne_x/loader.py`](../arachne_x/loader.py).

### Identity guarantee

When `--audio_conditioning_scale 0.0`, `generate_audio_i2v()` delegates to `generate_i2v()` — bit-identical path to base i2v.

## CLI (experimental)

Checkpoint: **VIDEO** tree (`ARACHNE-X-ULTRA-VIDEO` or merged runtime with `dit/` + `audio/wav2vec2`).

```bash
# A: base identity (must match i2v)
python scripts/infer.py \
  --checkpoint_dir "$VIDEO_CKPT" \
  --mode audio_i2v \
  --image assets/avatar/single/elena/image.jpg \
  --audio assets/avatar/single/elena/audio.wav \
  --prompt "Professional portrait, subtle motion, stable identity." \
  --negative_prompt "blurry, low quality, watermark" \
  --resolution 480p \
  --num_frames 49 \
  --num_inference_steps 25 \
  --text_guidance_scale 4.0 \
  --audio_conditioning_scale 0.0 \
  --output output/audio_i2v_scale0.mp4

# B: adapter engaged (zero-init adapter = near-base until trained)
python scripts/infer.py \
  --checkpoint_dir "$VIDEO_CKPT" \
  --mode audio_i2v \
  --image assets/avatar/single/elena/image.jpg \
  --audio assets/avatar/single/elena/audio.wav \
  --prompt "Professional portrait speaking naturally, stable identity, subtle head motion." \
  --negative_prompt "blurry, low quality, watermark" \
  --resolution 480p \
  --num_frames 49 \
  --num_inference_steps 25 \
  --text_guidance_scale 4.0 \
  --audio_conditioning_scale 1.0 \
  --audio_conditioning_adapter output/audio_i2v_adapter.safetensors \
  --preset_hint audio_i2v_smoke \
  --output output/audio_i2v_scale1.mp4
```

Sidecar metadata: `<output>.run.json` includes adapter path, scale, num_frames.

## Adapter weights

- Format: `.safetensors` + `.meta.json` (config, block indices, version).
- Save/load: `save_audio_conditioning_adapter()` / `load_audio_conditioning_adapter()`.
- Gates initialized to **zero** — adapter starts as no-op even with scale=1 until training.

## Training (frozen base)

See [`scripts/train_audio_conditioning_adapter.py`](../scripts/train_audio_conditioning_adapter.py) and dataset manifest [`assets/training/audio_i2v_pairs.example.json`](../assets/training/audio_i2v_pairs.example.json).

Freeze: VIDEO DiT, VAE, text encoder, wav2vec encoder.  
Train: adapter `audio_proj`, per-block injection gates and cross-attn weights only.

## Imagine I2V (TTS + adapter, no user WAV)

Product-style path: **image + prompt → TTS speech → audio adapter → muxed MP4**.

No base DiT / UMT5 weight changes. The prompt compiler is deterministic/offline only in this runtime.

```bash
# needs: VIDEO ckpt + wav2vec + requirements-tts.txt

python scripts/infer.py \
  --checkpoint_dir "$VIDEO_CKPT" \
  --mode imagine_i2v \
  --image assets/avatar/single/elena/image.jpg \
  --prompt "Elena greets the candidate calmly in a modern office" \
  --speak_text "Здравствуйте, рада познакомиться." \
  --tts_provider qwen \
  --tts_language Russian \
  --tts_speaker Ryan \
  --resolution 480p \
  --num_frames 49 \
  --num_inference_steps 25 \
  --text_guidance_scale 4.0 \
  --audio_conditioning_scale 1.0 \
  --audio_conditioning_adapter output/audio_i2v_adapter.safetensors \
  --output output/imagine_i2v.mp4
```

Rules:
- **No `--audio`** — speech is synthesized internally (use `--mode audio_i2v` if you have WAV).
- **`--speak_text`** optional if `--prompt` is short (≤320 chars); long scene prompts need explicit speak line.
- Prompt compiler for `imagine_i2v`: deterministic `off` / template merge only.
- Output is **muxed MP4** with TTS audio (like avatar modes).

Pipeline:

```text
prompt → UMT5 cross-attn (frozen)
      ↘ speak_text → TTS → wav2vec → adapter → frozen DiT → video
                                                      ↘ ffmpeg mux
```


- Frontend V1/V2
- Elena identity bank / avatar ai2v prod
- Gateway / RunPod worker HTTP
- Replacing UMT5 text encoder with an external LLM prompt compiler

## Operational review

| Risk | Mitigation |
| ---- | ---------- |
| Adapter destroys texture | zero-init gates; scale=0 identity path; A/B smokes |
| Audio/video length mismatch | 4n+1 frames; log frame-budget in run.json |
| Checkpoint ABI drift | adapter stored separately from base VIDEO dit |
| GPU memory | inject only middle/late blocks (default 24..46 step 2) |
