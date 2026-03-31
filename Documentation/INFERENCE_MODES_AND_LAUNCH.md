# ARACHNE-X Inference Modes And Launch

## Source Of Truth
Основной entrypoint для inference: `scripts/infer.py`.

Поддерживаемые режимы:

| Mode | Inputs | Output | Use case |
| --- | --- | --- | --- |
| `t2v` | `prompt` | video | Text-to-video smoke tests |
| `i2v` | `image`, `prompt` | video | Animate from image |
| `vc` | `video`, `prompt` | video | Video continuation |
| `ai2v` | `image`, `audio`, `prompt` | video+audio | Main single-avatar talking-head mode |
| `at2v` | `audio`, `prompt` | video+audio | Avatar audio-driven without image anchor |
| `avc` | `video`, `audio`, `prompt` | video+audio | Avatar continuation from video context |
| `streaming_ai2v` | `image`, `audio`, `prompt` | video+audio | Streaming-oriented avatar path |
| `enroll_identity` | `image`, `identity_id` | saved bank | Identity token enrollment |

## Fastest Practical Modes

### Fast text-to-video
- mode: `t2v`
- resolution: `480 x 832`
- steps: `8`

### Fast talking avatar
- mode: `ai2v`
- resolution: `480p`
- steps: `8`
- inputs: one reference image + one speech audio

## Recommended Pod Environment

```bash
source /workspace/ARACHNE-X/.venv/bin/activate
export PYTHONPATH=/workspace/ARACHNE-X
```

## Launch Examples

### 1. Fast T2V

```bash
python scripts/infer.py \
  --checkpoint_dir /workspace/weights/ARACHNE-X \
  --mode t2v \
  --prompt "A cinematic close-up of a beautiful young woman, 24+ years old, with long white hair, wearing a black NULLXES suit, futuristic luxury fashion aesthetic, soft studio lighting, highly detailed face, natural eye movement, subtle head motion, elegant clean background, realistic skin texture, premium high-end look" \
  --negative_prompt "low quality, blurry, deformed face, bad anatomy, flicker, artifacts, watermark, text" \
  --height 480 \
  --width 832 \
  --num_frames 93 \
  --num_inference_steps 8 \
  --text_guidance_scale 4.0 \
  --output /workspace/out_nullxes_t2v.mp4
```

### 2. Fast AI2V

```bash
python scripts/infer.py \
  --checkpoint_dir /workspace/weights/ARACHNE-X-Avatar \
  --mode ai2v \
  --image /workspace/ARACHNE-X/assets/avatar/single/MaximOnyushko/image.png \
  --audio /workspace/ARACHNE-X/assets/avatar/single/MaximOnyushko/voice.wav \
  --prompt "A realistic close-up of a young man speaking directly to camera, natural facial expression, precise lip movements synchronized with speech, subtle head motion, stable identity, soft cinematic lighting, high facial detail" \
  --resolution 480p \
  --num_frames 93 \
  --num_inference_steps 8 \
  --text_guidance_scale 4.0 \
  --audio_guidance_scale 4.0 \
  --output /workspace/out_maxim_ai2v.mp4
```

## Optional: custom avatar LoRA at inference

After training with `scripts/train_lora_avatar.py`, load the adapter on the avatar DiT (same `lora_rank` / `lora_alpha` as in training), then enable before generation — same pattern as `Demo/run_streamlit.py` (`pipe.dit.load_lora(path, key, lora_network_dim=..., lora_network_alpha=...)` → `pipe.dit.enable_loras([key])`). A minimal round-trip check: `python scripts/verify_lora_avatar.py [--checkpoint_dir /path/to/weights]`.

## Notes

- `Demo/run_demo_text_to_video.py` is not a flexible CLI. It contains hardcoded prompts and outputs.
- For reproducible pod runs, prefer `scripts/infer.py`.
- `ai2v` is the best current mode for fast avatar tests with existing LongCat Avatar weights.
