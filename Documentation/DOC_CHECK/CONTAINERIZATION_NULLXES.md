# ARACHNE-X (NULLXES) — Commands + Containerization (GPU)

## 0) Conventions used in commands

Paths below assume Linux (RunPod/Ubuntu/Docker). Replace with your actual locations.

```bash
export ARACHNE_REPO=/workspace/ARACHNE-X
export WEIGHTS=/weights
export OUT=/outputs
```

**Video weights (base):**
- `--checkpoint_dir "$WEIGHTS/ARACHNE-X-ULTRA-VIDEO"` (or `LongCat-Video` bundle)

**Avatar weights:**
- `--checkpoint_dir "$WEIGHTS/ARACHNE-X-ULTRA-AVATAR"` (must contain full expected layout; if avatar bundle lacks `tokenizer/` or `vae/`, link/copy them from the video bundle)

## 1) Video modes (no audio): t2v / i2v / vc

### 1.1 T2V (fast)

```bash
python "$ARACHNE_REPO/scripts/infer.py" \
  --checkpoint_dir "$WEIGHTS/ARACHNE-X-ULTRA-VIDEO" \
  --mode t2v \
  --prompt "Cinematic close-up portrait, premium studio lighting, realistic skin texture, subtle head motion, high detail" \
  --negative_prompt "low quality, blurry, artifacts, watermark, text, flicker" \
  --height 480 --width 832 \
  --num_frames 93 \
  --num_inference_steps 8 \
  --text_guidance_scale 4.0 \
  --output "$OUT/t2v_fast.mp4"
```

### 1.2 T2V (HQ / 720p)

```bash
python "$ARACHNE_REPO/scripts/infer.py" \
  --checkpoint_dir "$WEIGHTS/ARACHNE-X-ULTRA-VIDEO" \
  --mode t2v \
  --prompt "Cinematic close-up portrait, premium studio lighting, realistic skin texture, high detail, sharp focus" \
  --negative_prompt "low quality, blurry, artifacts, watermark, text, flicker" \
  --height 768 --width 1280 \
  --num_frames 181 \
  --num_inference_steps 25 \
  --text_guidance_scale 5.0 \
  --output "$OUT/t2v_hq_720p.mp4"
```

### 1.3 Longer video (segment + continuation)

15s at 30fps is **450 frames**. Practical approach: generate first segment, then continue with `vc` (repeat until target length).

Segment 1 (e.g. 6s = 181 frames):

```bash
python "$ARACHNE_REPO/scripts/infer.py" \
  --checkpoint_dir "$WEIGHTS/ARACHNE-X-ULTRA-VIDEO" \
  --mode t2v \
  --prompt "..." \
  --negative_prompt "..." \
  --height 768 --width 1280 \
  --num_frames 181 \
  --num_inference_steps 25 \
  --text_guidance_scale 5.0 \
  --output "$OUT/seg_00.mp4"
```

Continuation (repeat as needed):

```bash
python "$ARACHNE_REPO/scripts/infer.py" \
  --checkpoint_dir "$WEIGHTS/ARACHNE-X-ULTRA-VIDEO" \
  --mode vc \
  --video "$OUT/seg_00.mp4" \
  --prompt "..." \
  --negative_prompt "..." \
  --resolution 720p \
  --num_frames 181 \
  --num_cond_frames 13 \
  --num_inference_steps 25 \
  --text_guidance_scale 5.0 \
  --output "$OUT/seg_01.mp4"
```

### 1.4 I2V (image → video)

```bash
python "$ARACHNE_REPO/scripts/infer.py" \
  --checkpoint_dir "$WEIGHTS/ARACHNE-X-ULTRA-VIDEO" \
  --mode i2v \
  --image "$ARACHNE_REPO/assets/avatar/single/anna/anna.jpg" \
  --prompt "High-end corporate portrait speaking-like micro-motion, stable identity, clean background" \
  --negative_prompt "low quality, blurry, artifacts, watermark, text, flicker" \
  --resolution 720p \
  --num_frames 181 \
  --num_inference_steps 25 \
  --text_guidance_scale 5.0 \
  --output "$OUT/i2v_hq.mp4"
```

## 2) Avatar modes (with audio mux): ai2v / at2v / avc / streaming_ai2v

### 2.1 Audio prep (fix “video faster than audio” baseline)

Resample speech audio to **16kHz mono** before avatar inference.

```bash
ffmpeg -y -i "$ARACHNE_REPO/assets/avatar/demo emploey/hr.wav" -ar 16000 -ac 1 "$OUT/hr_16k.wav"
```

### 2.2 AI2V (image + audio file) — lip-sync tuned

```bash
python "$ARACHNE_REPO/scripts/infer.py" \
  --checkpoint_dir "$WEIGHTS/ARACHNE-X-ULTRA-AVATAR" \
  --mode ai2v \
  --image "$ARACHNE_REPO/assets/avatar/single/anna/anna.jpg" \
  --audio "$OUT/hr_16k.wav" \
  --prompt "Professional woman speaking directly to camera, precise lip articulation synchronized with speech, stable identity, subtle head motion, soft diffused studio light, high facial detail" \
  --negative_prompt "low quality, blurry, mismatched lip sync, lips out of sync, frozen mouth, jitter, flicker, artifacts, watermark, text" \
  --resolution 480p \
  --num_frames 181 \
  --num_inference_steps 25 \
  --text_guidance_scale 4.0 \
  --audio_guidance_scale 5.0 \
  --output "$OUT/ai2v_audio_lipsync.mp4"
```

### 2.3 AI2V (image + speak_text via Qwen TTS) — female speaker

Install once:

```bash
pip install -r requirements-tts.txt
```

Run:

```bash
python "$ARACHNE_REPO/scripts/infer.py" \
  --checkpoint_dir "$WEIGHTS/ARACHNE-X-ULTRA-AVATAR" \
  --mode ai2v \
  --image "$ARACHNE_REPO/assets/avatar/single/anna/anna.jpg" \
  --speak_text "Hello, this is Anna from NULLXES, demonstrating ARACHNE-X avatar pipeline." \
  --tts_provider qwen \
  --tts_model "$WEIGHTS/Qwen3-TTS" \
  --tts_language English \
  --tts_speaker Cherry \
  --prompt "Professional woman speaking directly to camera, precise lip articulation synchronized with speech, stable identity, subtle head motion, soft diffused studio light, high facial detail" \
  --negative_prompt "low quality, blurry, mismatched lip sync, lips out of sync, frozen mouth, jitter, flicker, artifacts, watermark, text" \
  --resolution 480p \
  --num_frames 181 \
  --num_inference_steps 25 \
  --text_guidance_scale 4.0 \
  --audio_guidance_scale 5.0 \
  --output "$OUT/ai2v_speak_qwen_female.mp4"
```

Notes:
- `--audio` has priority over `--speak_text`.
- If a speaker name is not supported by your local Qwen3-TTS snapshot, switch `--tts_speaker` to any valid voice name in that model.

### 2.4 AT2V (audio-only avatar)

```bash
python "$ARACHNE_REPO/scripts/infer.py" \
  --checkpoint_dir "$WEIGHTS/ARACHNE-X-ULTRA-AVATAR" \
  --mode at2v \
  --audio "$OUT/hr_16k.wav" \
  --prompt "A realistic talking head speaking naturally to camera, precise lip articulation synchronized with speech, minimal head movement, high detail" \
  --negative_prompt "low quality, blurry, mismatched lip sync, lips out of sync, frozen mouth, jitter, flicker, artifacts, watermark, text" \
  --height 480 --width 832 \
  --num_frames 181 \
  --num_inference_steps 25 \
  --text_guidance_scale 4.0 \
  --audio_guidance_scale 5.0 \
  --output "$OUT/at2v_audio.mp4"
```

### 2.5 AVC (video continuation + audio)

```bash
python "$ARACHNE_REPO/scripts/infer.py" \
  --checkpoint_dir "$WEIGHTS/ARACHNE-X-ULTRA-AVATAR" \
  --mode avc \
  --video "$ARACHNE_REPO/assets/avatar/single/anna/333.mp4" \
  --audio "$OUT/hr_16k.wav" \
  --prompt "Continue speaking naturally, stable identity, precise lip articulation synchronized with speech" \
  --negative_prompt "low quality, blurry, mismatched lip sync, lips out of sync, frozen mouth, jitter, flicker, artifacts, watermark, text" \
  --height 480 --width 832 \
  --num_frames 181 \
  --num_cond_frames 13 \
  --num_inference_steps 25 \
  --text_guidance_scale 4.0 \
  --audio_guidance_scale 5.0 \
  --output "$OUT/avc_continue.mp4"
```

### 2.6 STREAMING_AI2V (file-driven streaming path)

```bash
python "$ARACHNE_REPO/scripts/infer.py" \
  --checkpoint_dir "$WEIGHTS/ARACHNE-X-ULTRA-AVATAR" \
  --mode streaming_ai2v \
  --image "$ARACHNE_REPO/assets/avatar/single/anna/anna.jpg" \
  --audio "$OUT/hr_16k.wav" \
  --audio_chunk_sec 0.5 \
  --prompt "Professional interviewer speaking naturally, stable identity, precise lip sync" \
  --negative_prompt "low quality, blurry, mismatched lip sync, flicker, artifacts" \
  --resolution 480p \
  --num_frames 181 \
  --num_inference_steps 25 \
  --text_guidance_scale 4.0 \
  --audio_guidance_scale 5.0 \
  --output "$OUT/streaming_ai2v_file.mp4"
```

## 3) “Фото” in this repo

There is **no text-to-image** CLI mode in `scripts/infer.py`. “Image mode” in current codebase means:
- `i2v` (image → video)
- `enroll_identity` (store identity tokens from an image)

### 3.1 enroll_identity (identity bank from image)

```bash
python "$ARACHNE_REPO/scripts/infer.py" \
  --checkpoint_dir "$WEIGHTS/ARACHNE-X-ULTRA-AVATAR" \
  --mode enroll_identity \
  --image "$ARACHNE_REPO/assets/avatar/single/anna/anna.jpg" \
  --identity_id 10 \
  --identity_bank_save_path "$OUT/identity_bank.pt"
```

## 4) Containerization (GPU)

This repo includes:
- `docker/Dockerfile.gpu`
- `docker/compose.gpu.yml`

### 4.1 Build

From repo root:

```bash
docker build -f docker/Dockerfile.gpu -t arachne-x:gpu .
```

Optional: bake avatar/TTS deps into the image:

```bash
docker build -f docker/Dockerfile.gpu -t arachne-x:gpu \
  --build-arg INSTALL_AVATAR_DEPS=1 \
  --build-arg INSTALL_TTS_DEPS=1 \
  .
```

### 4.2 Run (interactive)

```bash
docker run --rm -it --gpus all --shm-size=8g \
  -e PYTHONPATH=/workspace/ARACHNE-X \
  -e HF_TOKEN \
  -v "$PWD":/workspace/ARACHNE-X \
  -v "$PWD/docker/weights":/weights \
  -v "$PWD/docker/outputs":/outputs \
  arachne-x:gpu
```

### 4.3 Run via compose

```bash
cd docker
docker compose -f compose.gpu.yml up --build
```

## 5) Weights: download pattern (safe)

Never put tokens directly in commands stored in shell history or docs. Use env:

```bash
export HF_TOKEN='***'
```

Then either `hf download ...` (CLI) or use the repo helper:

```bash
python "$ARACHNE_REPO/scripts/download_longcat_video.py" --local-dir "$WEIGHTS/LongCat-Video"
```

