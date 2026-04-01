# ARACHNE-X Inference Modes And Launch

## Source Of Truth
Основной entrypoint для inference: `scripts/infer.py`.

Поддерживаемые режимы:

| Mode | Inputs | Output | Use case |
| --- | --- | --- | --- |
| `t2v` | `prompt` | video | Text-to-video smoke tests |
| `i2v` | `image`, `prompt` | video | Animate from image |
| `vc` | `video`, `prompt` | video | Video continuation |
| `ai2v` | `image`, `audio` **or** `--speak_text`, `prompt` | video+audio | Main single-avatar talking-head mode |
| `at2v` | `audio` **or** `--speak_text`, `prompt` | video+audio | Avatar audio-driven without image anchor |
| `avc` | `video`, `audio` **or** `--speak_text`, `prompt` | video+audio | Avatar continuation from video context |
| `streaming_ai2v` | `image`, `audio` **or** `--speak_text`, `prompt` | video+audio | Streaming-oriented avatar path |
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

## Weights path and Hugging Face Hub

`--checkpoint_dir` is normally a **local directory** that contains `tokenizer/`, `vae/`, etc. (`WeightsLayout`).

Optional: pass a Hub repo id (`org/model`) **and** `--allow_hub_download` on `scripts/infer.py`, `scripts/train.py`, `scripts/train_lora_avatar.py`, or `scripts/export_latent_training_sample.py`. Resolution is handled by [`arachne_x/weights_resolve.py`](../arachne_x/weights_resolve.py) (`snapshot_download`). Private repos need `HF_TOKEN` in the environment. Air-gapped runs: omit `--allow_hub_download` and use a pre-synced folder.

## TTS: text → wav → avatar (mp4 with audio)

For `ai2v`, `at2v`, `avc`, and `streaming_ai2v`, you may omit `--audio` and pass **`--speak_text`** instead. The CLI synthesizes a temporary WAV via a pluggable backend (**`--tts_provider`**, default `qwen`), then runs the same avatar pipeline and **muxes** with [`save_video_ffmpeg`](../arachne_x/audio_process/torch_utils.py). If both `--audio` and `--speak_text` are set, **`--audio` wins**.

Install optional deps (see [Qwen3-TTS-12Hz-1.7B-CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice)):

```bash
pip install -r requirements-tts.txt
```

Useful flags: `--tts_model` (HF id or local dir), `--tts_language`, `--tts_speaker`, `--tts_instruct`, `--tts_device_map`, `--tts_attn`. For **`streaming_ai2v`**, **`--audio_chunk_sec`** sets fixed-duration chunks (micro-turns; default `0.5`). Runtime contract: [`arachne_x/tts/realtime.py`](../arachne_x/tts/realtime.py), chunk helper [`arachne_x/tts/chunking.py`](../arachne_x/tts/chunking.py).

### LongCat-AudioDiT as TTS (`--tts_provider longcat_audiodit`)

[LongCat-AudioDiT](https://huggingface.co/meituan-longcat/LongCat-AudioDiT-1B) generates speech in waveform latent space; the synthesizer **resamples to 16 kHz** before writing WAV so existing **`get_audio_embedding`** / Wav2Vec conditioning stays unchanged.

```bash
pip install -r requirements-audiodit.txt
```

Flags: `--tts_model` (default `meituan-longcat/LongCat-AudioDiT-1B`), `--tts_device_map`, `--audiodit_nfe`, `--audiodit_guidance_strength`, `--audiodit_guidance_method` (`cfg` / `apg`), `--audiodit_seed`. Voice cloning: **`--audiodit_prompt_audio`** + **`--audiodit_prompt_text`** (transcript of the reference clip).

**VRAM:** AudioDiT (especially 3.5B) plus avatar DiT on one GPU may OOM or jitter; use one GPU with strict scheduling or a dedicated TTS worker.

**`transformers` version:** AudioDiT wants a recent transformers stack; verify compatibility with your main `requirements.txt` in the target venv.

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

### 2b. AI2V from text only (TTS + mux)

```bash
python scripts/infer.py \
  --checkpoint_dir /workspace/weights/ARACHNE-X-Avatar \
  --mode ai2v \
  --image /workspace/ARACHNE-X/assets/avatar/single/anna/anna.jpg \
  --speak_text "Hello, this line is synthesized with Qwen3-TTS, then lip-synced by the avatar DiT." \
  --tts_provider qwen \
  --tts_language English \
  --tts_speaker Ryan \
  --prompt "A realistic close-up speaking to camera, natural expression, lip synced to speech" \
  --resolution 480p \
  --num_frames 93 \
  --num_inference_steps 8 \
  --output /workspace/out_anna_speak.mp4
```

### 2c. AI2V from text (LongCat-AudioDiT TTS)

```bash
python scripts/infer.py \
  --checkpoint_dir /workspace/weights/ARACHNE-X-Avatar \
  --mode ai2v \
  --image /path/to/ref.png \
  --speak_text "Line synthesized with LongCat-AudioDiT, then lip-synced by the avatar." \
  --tts_provider longcat_audiodit \
  --tts_model meituan-longcat/LongCat-AudioDiT-1B \
  --audiodit_guidance_method apg \
  --prompt "A realistic close-up speaking to camera, lip synced to speech" \
  --resolution 480p \
  --num_frames 93 \
  --num_inference_steps 8 \
  --output /workspace/out_audiodit_speak.mp4
```

## Optional: custom avatar LoRA at inference

After `scripts/train_lora_avatar.py`, use **`lora_final.safetensors`** (and **`lora_train_meta.json`** next to it for rank/alpha) with `scripts/infer.py`:

```bash
python scripts/infer.py \
  --checkpoint_dir /path/to/weights \
  --mode ai2v \
  --image /path/ref.png \
  --audio /path/speech.wav \
  --prompt "..." \
  --lora_path /path/to/lora_final.safetensors \
  --lora_key train \
  --output out.mp4
```

If `lora_train_meta.json` sits in the **same directory** as `--lora_path`, rank/alpha are read automatically. Otherwise pass `--lora_rank` and `--lora_alpha` explicitly (must match training). Optional: `--lora_meta_json /path/to/lora_train_meta.json`.

The same flags apply to `at2v`, `avc`, `streaming_ai2v` (any avatar pipeline mode). Programmatic equivalent: `pipe.dit.load_lora(...)` then `pipe.dit.enable_loras([key])` as in `Demo/run_streamlit.py`.

Smoke (toy model, no full GPU stack): `python scripts/verify_lora_avatar.py`. With real weights: add `--checkpoint_dir` containing `avatar_single/`.

## Notes

- `Demo/run_demo_text_to_video.py` is not a flexible CLI. It contains hardcoded prompts and outputs.
- For reproducible pod runs, prefer `scripts/infer.py`.
- `ai2v` is the best current mode for fast avatar tests with existing LongCat Avatar weights.
