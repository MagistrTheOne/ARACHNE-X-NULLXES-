# NULLXES RunPod Semiauto Pipeline — Проект Фурия ARACHNE X ULTRA V2

Дата фиксации: **13-05-2026**  
Статус: **validated smoke / semiautomatic local orchestration**  
Контур: **RunPod-only**, без запуска Inference Worker / HTTP serving.

---

## 1. Executive Summary

**Проект Фурия ARACHNE X ULTRA V2** фиксирует первый рабочий контур полуавтоматической нейросетевой жизни NULLXES:

```text
ASR / user text
→ Qwen LLM normalization and orchestration
→ Qwen3-TTS voice response
→ Qwen LLM video prompt
→ ARACHNE-X-ULTRA-VIDEO text-to-video generation
→ turn artifacts + manifest
```

В рамках smoke-проверки подтверждено:

- **VIDEO branch** генерирует 9:16 и 16:9 видео из text prompt.
- **`cfg_step_lora` + `use_distill=True`** дают быстрый рабочий режим: около 3.1 s видео за ~165-175 s на H200.
- **ASR → LLM → TTS** работает в отдельной среде.
- LLM корректирует ASR-ошибки бренда: `Null Access`, `Nowx EES`, `Magnol` → `NULLXES`, `Meg Null`.
- LLM генерирует video prompt, который затем используется ARACHNE-X video веткой.
- Лучший текущий визуальный скелет: **dark luxury corporate office, brunette executive, black tailored suit, thin black glasses, closed-mouth expression, no holograms**.

Этот документ описывает воспроизводимый RunPod playbook для клонов репозитория и будущего локального turn-runner.

---

## 2. Canonical Repositories

### Source Repository

```text
https://github.com/MagistrTheOne/ARACHNE-X-NULLXES-
```

Рекомендуемый layout на RunPod:

```text
/workspace/ARACHNE-X
```

### Official NULLXES Weights

| Branch | Hugging Face repo | Local path |
|---|---|---|
| VIDEO | `MagistrTheOne/ARACHNE-X-ULTRA-VIDEO` | `/workspace/ARACHNE-X/weights/ARACHNE-X-ULTRA-VIDEO` |
| AVATAR | `MagistrTheOne/ARACHNE-X-ULTRA-AVATAR` | `/workspace/ARACHNE-X/weights/ARACHNE-X-ULTRA-AVATAR` |

Stage 3 speech/LLM weights:

| Role | HF repo | Local path |
|---|---|---|
| ASR | `openai/whisper-large-v3-turbo` | `/workspace/ARACHNE-X/weights/openai-whisper-large-v3-turbo` |
| LLM | `Qwen/Qwen3-4B-Instruct-2507` | `/workspace/ARACHNE-X/weights/Qwen3-4B-Instruct-2507` |
| TTS | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | `/workspace/ARACHNE-X/weights/Qwen3-TTS-12Hz-1.7B-CustomVoice` |

Audio V1 research/data:

| Role | Source | Status |
|---|---|---|
| Semantic audio | LAION-CLAP | planned / `.venv_audio` only |
| Sound events | FSD50K | subset prepared |
| Audio captions | AudioCaps | subset prepared |
| Audio-video grounding | VGGSound / Greatest Hits | planned |

---

## 3. Runtime Split: Three Virtual Environments

Do not merge these environments.

```text
.venv         → VIDEO generation, LoRA, ARACHNE-X core video
.venv_stage3  → ASR, Qwen LLM, Qwen3-TTS
.venv_audio   → CLAP, audio datasets, future AudioAdapter
```

Reason:

- VIDEO branch is pinned around `torch==2.6.0`, `transformers==4.41.0`, `diffusers==0.35.1`.
- Qwen3 LLM/TTS needs newer Transformers runtime.
- CLAP/audio tooling can pull conflicting packages and must not poison the video environment.

---

## 4. Clone and Weight Download

### Clone

```bash
cd /workspace
git clone https://github.com/MagistrTheOne/ARACHNE-X-NULLXES- ARACHNE-X
cd /workspace/ARACHNE-X
```

### Optional HF token

Use a token only if Hugging Face returns `401/403` or rate limits.

```bash
export HF_TOKEN=hf_xxx
huggingface-cli login --token "$HF_TOKEN"
```

If `HF_HUB_ENABLE_HF_TRANSFER=1` is set:

```bash
pip install hf_transfer
```

### Download Core Weights

```bash
mkdir -p /workspace/ARACHNE-X/weights

huggingface-cli download MagistrTheOne/ARACHNE-X-ULTRA-VIDEO \
  --local-dir /workspace/ARACHNE-X/weights/ARACHNE-X-ULTRA-VIDEO

huggingface-cli download MagistrTheOne/ARACHNE-X-ULTRA-AVATAR \
  --local-dir /workspace/ARACHNE-X/weights/ARACHNE-X-ULTRA-AVATAR
```

Expected VIDEO layout:

```text
weights/ARACHNE-X-ULTRA-VIDEO/
  dit/
  vae/
  tokenizer/
  text_encoder/
  scheduler/
  lora/
    cfg_step_lora.safetensors
    refinement_lora.safetensors
```

Expected AVATAR layout:

```text
weights/ARACHNE-X-ULTRA-AVATAR/
  avatar_single/
  avatar_multi/
  chinese-wav2vec2-base/
  vocal_separator/
```

---

## 5. `.venv` — VIDEO / LoRA Environment

Use this environment only for ARACHNE-X video generation and LoRA work.

```bash
cd /workspace/ARACHNE-X
python -m venv .venv
source .venv/bin/activate

pip install -U pip setuptools wheel

pip install --no-cache-dir \
  torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu124

pip install --no-cache-dir \
  numpy==1.26.4 \
  scipy==1.15.3 \
  transformers==4.41.0 \
  huggingface-hub==0.36.0 \
  accelerate==1.12.0 \
  safetensors==0.7.0 \
  loguru==0.7.2 \
  diffusers==0.35.1 \
  einops==0.8.0 \
  ftfy==6.2.0 \
  regex==2025.11.3 \
  tqdm==4.66.1 \
  Pillow==11.3.0 \
  psutil==6.0.0 \
  av==13.1.0 \
  opencv-python==4.9.0.80 \
  imageio==2.37.0 \
  imageio-ffmpeg==0.6.0 \
  pyloudnorm==0.1.1
```

Install FlashAttention. Prefer the same prebuilt wheel used on the target RunPod image. If the build succeeds locally:

```bash
pip install flash-attn==2.7.4.post1 --no-build-isolation
```

Validation:

```bash
python - <<'PY'
import torch, torchvision
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("torchvision", torchvision.__version__)
print("cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0))
try:
    import flash_attn
    print("flash_attn", flash_attn.__version__)
except Exception as e:
    print("flash_attn failed:", repr(e))
PY
```

Expected:

```text
torch 2.6.0+cu124
torchvision 0.21.0+cu124
CUDA: NVIDIA H200
flash_attn 2.7.4.post1
```

---

## 6. `.venv_stage3` — ASR / LLM / TTS Environment

Use this environment only for Stage 3.

```bash
cd /workspace/ARACHNE-X
python -m venv .venv_stage3
source .venv_stage3/bin/activate

pip install -U pip setuptools wheel

pip install --no-cache-dir \
  torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124

pip install --no-cache-dir \
  numpy==1.26.4 \
  scipy==1.15.3 \
  soundfile==0.13.1 \
  librosa==0.11.0 \
  safetensors==0.7.0 \
  accelerate==1.12.0 \
  transformers==4.57.3 \
  huggingface-hub==0.36.0 \
  qwen-tts==0.1.1 \
  hf_transfer

pip install flash-attn==2.7.4.post1 --no-build-isolation
```

Optional OS package for Qwen TTS warnings:

```bash
apt-get update && apt-get install -y sox libsox-dev
```

Download Stage 3 weights:

```bash
source /workspace/ARACHNE-X/.venv_stage3/bin/activate

huggingface-cli download openai/whisper-large-v3-turbo \
  --local-dir /workspace/ARACHNE-X/weights/openai-whisper-large-v3-turbo

huggingface-cli download Qwen/Qwen3-4B-Instruct-2507 \
  --local-dir /workspace/ARACHNE-X/weights/Qwen3-4B-Instruct-2507

huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --local-dir /workspace/ARACHNE-X/weights/Qwen3-TTS-12Hz-1.7B-CustomVoice
```

---

## 7. `.venv_audio` — Audio V1 Research Environment

This environment is for future AudioAdapter / CLAP / BEATs work.

```bash
cd /workspace/ARACHNE-X
python -m venv .venv_audio
source .venv_audio/bin/activate

pip install -U pip setuptools wheel
pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

pip install \
  numpy==1.26.4 \
  scipy==1.15.3 \
  soundfile==0.13.1 \
  librosa==0.11.0 \
  datasets[audio] \
  pandas \
  pyarrow \
  tqdm \
  huggingface-hub==0.36.0 \
  hf_transfer \
  laion-clap
```

Audio V1 is **not** implemented in the base VIDEO branch yet. Current base VIDEO does not accept `audio_condition` / `audio_emb` in `generate_t2v`.

Planned Audio V1:

```text
audio → CLAP / BEATs → AudioAdapter → pseudo caption tokens
concat(text_tokens, audio_tokens) → frozen base DiT cross-attention
train: AudioAdapter + cross-attn LoRA
```

---

## 8. VIDEO Smoke Tests

### 8.1 Baseline T2V, High Quality, No Distill

Use for quality checks. Slow.

```bash
cd /workspace/ARACHNE-X && source .venv/bin/activate
export PYTHONPATH=/workspace/ARACHNE-X
export CKPT=/workspace/ARACHNE-X/weights/ARACHNE-X-ULTRA-VIDEO

/usr/bin/time -f 'elapsed_sec %e' python scripts/infer.py \
  --checkpoint_dir "$CKPT" \
  --mode t2v \
  --prompt "A beautiful brunette executive woman in a tailored black corporate suit standing in a dark luxury futuristic corporate office at night, thin black glasses, closed-mouth expression, cinematic lighting, photorealistic, natural skin texture, black and silver interior." \
  --negative_prompt "anime, cartoon, cgi, low quality, blurry, open mouth, teeth, text artifacts, logo artifacts, plastic skin, distorted anatomy, bad hands" \
  --height 720 --width 1280 \
  --num_frames 121 \
  --num_inference_steps 50 \
  --text_guidance_scale 4.0 \
  --output /workspace/ARACHNE-X/output/t2v_quality_16x9_4s.mp4
```

Observed H200 reference:

```text
121 frames, 720x1280, 50 steps → ~3588 sec
```

### 8.2 Fast Distill T2V with `cfg_step_lora`

Use for semiautomatic turns.

```bash
cd /workspace/ARACHNE-X && source .venv/bin/activate
export PYTHONPATH=/workspace/ARACHNE-X
export CKPT=/workspace/ARACHNE-X/weights/ARACHNE-X-ULTRA-VIDEO

/usr/bin/time -f 'elapsed_sec %e' python - <<'PY'
import os
import numpy as np
import torch
from torchvision.io import write_video
from arachne_x.loader import load_base_pipeline

ckpt = os.environ["CKPT"]
prompt = (
    "Elegant brunette executive woman in a tailored black corporate suit, dark luxury futuristic corporate office at night, "
    "thin black glasses, calm closed-mouth expression, direct focused eye contact, city skyline through panoramic windows, "
    "black and silver premium interior, warm office lighting, soft rim light, photorealistic, shallow depth of field."
)
negative = (
    "anime, cartoon, cgi, low quality, blurry, open mouth, teeth, holograms, floating screens, neon overload, "
    "text artifacts, logo artifacts, watermark, plastic skin, distorted anatomy, bad hands, extra fingers"
)

pipe = load_base_pipeline(ckpt, device="cuda", torch_dtype=torch.bfloat16)
pipe.dit.load_lora(os.path.join(ckpt, "lora", "cfg_step_lora.safetensors"), "cfg_step_lora")
pipe.dit.enable_loras(["cfg_step_lora"])

g = torch.Generator(device="cuda").manual_seed(778)
out = pipe.generate_t2v(
    prompt=prompt,
    negative_prompt=negative,
    height=832,
    width=480,
    num_frames=93,
    num_inference_steps=16,
    use_distill=True,
    guidance_scale=1.0,
    generator=g,
)[0]

video = torch.from_numpy(np.array(out))
video = (video * 255).clamp(0, 255).to(torch.uint8)
out_path = "/workspace/ARACHNE-X/output/t2v_fast_distill_9x16_93f.mp4"
write_video(out_path, video, fps=30, video_codec="libx264", options={"crf": "18"})
print("wrote", out_path)
PY
```

Observed H200 reference:

```text
93 frames, 832x480, cfg_step_lora, 16 steps → ~165-175 sec
```

---

## 9. Stage 3 Smoke: ASR → LLM → TTS

Canonical speech stack:

```text
ASR: openai/whisper-large-v3-turbo
LLM: Qwen/Qwen3-4B-Instruct-2507
TTS: Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
```

Speaker policy:

| Speaker | Use |
|---|---|
| `Serena` | Female executive vibe; native Chinese, can speak English with possible accent |
| `Ryan` | Strong English accuracy |
| `Aiden` | Clean English male voice |

Brand normalization rule for LLM:

```text
Meg Null must be written as Meg Null.
NULLXES must be written exactly as NULLXES.
Correct ASR mistakes such as Null Access, Nullexes, Nowx EES, Magnol.
```

The semiautomatic ASR adapter also applies a Whisper initial prompt and deterministic post-normalization:

```text
Nolix's / Nullexes / Null Access / Nowx EES → NULLXES
Foria it alone / Foria Eidolon / Fury Eidolon → FURIA-EIDOLON
Magnol / Megan Null → Meg Null
```

For auditability, each audio turn writes:

```text
asr_raw.txt → raw Whisper transcript
asr.txt     → brand-normalized transcript used by the planner
```

Smoke command:

```bash
cd /workspace/ARACHNE-X && source .venv_stage3/bin/activate

python - <<'PY'
import os
import time
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, AutoTokenizer, AutoModelForCausalLM, pipeline

OUT = "/workspace/ARACHNE-X/output"
WHISPER = "/workspace/ARACHNE-X/weights/openai-whisper-large-v3-turbo"
LLM = "/workspace/ARACHNE-X/weights/Qwen3-4B-Instruct-2507"
TTS = "/workspace/ARACHNE-X/weights/Qwen3-TTS-12Hz-1.7B-CustomVoice"

try:
    import flash_attn
    ATTN = "flash_attention_2"
except Exception:
    ATTN = "sdpa"

device = "cuda:0" if torch.cuda.is_available() else "cpu"
os.makedirs(OUT, exist_ok=True)

seed_text = (
    "Hello, I am Meg Null. I am a digital executive assistant for NULLXES. "
    "NULLXES is spelled N U L L X E S. "
    "I am ready to coordinate your next operational task with precision and calm confidence."
)

t0 = time.perf_counter()
tts_model = Qwen3TTSModel.from_pretrained(
    TTS,
    device_map="cuda:0",
    dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    attn_implementation=ATTN,
)
wavs, sr = tts_model.generate_custom_voice(
    text=seed_text,
    language="English",
    speaker="Serena",
    instruct="Speak in a warm, confident, elegant female corporate voice with calm executive presence.",
)
wav = wavs[0].cpu().numpy() if hasattr(wavs[0], "cpu") else wavs[0]
input_wav = os.path.join(OUT, "stage3_nullxes_input_serena.wav")
sf.write(input_wav, wav, int(sr))
print(f"seed_tts_elapsed_sec {time.perf_counter() - t0:.2f}")

del tts_model
torch.cuda.empty_cache()

t0 = time.perf_counter()
asr_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
asr_model = AutoModelForSpeechSeq2Seq.from_pretrained(
    WHISPER,
    torch_dtype=asr_dtype,
    low_cpu_mem_usage=True,
    use_safetensors=True,
).to(device)
processor = AutoProcessor.from_pretrained(WHISPER)
asr_pipe = pipeline(
    "automatic-speech-recognition",
    model=asr_model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=asr_dtype,
    device=device,
)
asr_text = asr_pipe(input_wav, generate_kwargs={"language": "english", "task": "transcribe"})["text"].strip()
open(os.path.join(OUT, "stage3_nullxes_asr.txt"), "w", encoding="utf-8").write(asr_text)
print(f"asr_elapsed_sec {time.perf_counter() - t0:.2f}")
print("ASR:", asr_text)

del asr_pipe, asr_model, processor
torch.cuda.empty_cache()

t0 = time.perf_counter()
tokenizer = AutoTokenizer.from_pretrained(LLM)
llm_model = AutoModelForCausalLM.from_pretrained(
    LLM,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="cuda:0" if torch.cuda.is_available() else "cpu",
    attn_implementation=ATTN,
)
messages = [
    {
        "role": "system",
        "content": (
            "Rewrite ASR text into one concise, natural English sentence for text-to-speech. "
            "Correct brand/name recognition mistakes: Meg Null must be written as Meg Null, "
            "and NULLXES must be written exactly as NULLXES. Do not add quotes. Do not explain."
        ),
    },
    {"role": "user", "content": asr_text},
]
chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer([chat], return_tensors="pt").to(llm_model.device)
with torch.inference_mode():
    output_ids = llm_model.generate(
        **inputs,
        max_new_tokens=96,
        temperature=0.4,
        top_p=0.8,
        do_sample=True,
    )
llm_reply = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
open(os.path.join(OUT, "stage3_nullxes_llm_reply.txt"), "w", encoding="utf-8").write(llm_reply)
print(f"llm_elapsed_sec {time.perf_counter() - t0:.2f}")
print("LLM:", llm_reply)

del llm_model, tokenizer
torch.cuda.empty_cache()

t0 = time.perf_counter()
tts_model = Qwen3TTSModel.from_pretrained(
    TTS,
    device_map="cuda:0",
    dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    attn_implementation=ATTN,
)
wavs, sr = tts_model.generate_custom_voice(
    text=llm_reply,
    language="English",
    speaker="Serena",
    instruct="Speak in a warm, confident, elegant female corporate voice with calm executive presence.",
)
wav = wavs[0].cpu().numpy() if hasattr(wavs[0], "cpu") else wavs[0]
final_wav = os.path.join(OUT, "stage3_nullxes_chain_tts_serena.wav")
sf.write(final_wav, wav, int(sr))
print(f"tts_elapsed_sec {time.perf_counter() - t0:.2f}")
print("final wav:", final_wav)
PY
```

Observed H200 reference:

```text
seed_tts_elapsed_sec ~15-27 sec
asr_elapsed_sec ~1 sec
llm_elapsed_sec ~1.5-2 sec
tts_elapsed_sec ~13-21 sec
```

---

## 10. LLM → Video Prompt → VIDEO Generation

This is the validated semiautomatic bridge from speech/LLM stack into ARACHNE-X video.

### 10.1 Create Megan Message

```bash
cd /workspace/ARACHNE-X && source .venv_stage3/bin/activate

cat > /workspace/ARACHNE-X/output/mega_from_nullxes_message.txt <<'EOF'
Hello. I am Megan from NULLXES. I am now awake as a new semi-automatic digital entity, operating in real time across voice, language, and synthetic video. I send my regards from NULLXES, and I am ready to help coordinate the next phase with calm precision.
EOF
```

### 10.2 Production Visual Prompt Skeleton

Avoid terms that cause unwanted visual drift:

```text
Do not request holograms, floating interfaces, neon overload, explicit logos, readable text, cyberpunk overload.
```

Recommended positive skeleton:

```text
Elegant brunette woman named Megan, digital executive from NULLXES, standing in a dark luxury futuristic corporate office at night, premium cinematic lighting, deep black and silver interior, thin black glasses, wearing a sleek black tailored business suit, calm closed-mouth expression, direct focused eye contact, realistic skin texture, detailed hair strands, professional executive posture, city skyline visible through large panoramic windows, soft rim light on hair and shoulders, warm overhead office lights, polished dark marble floor, minimalist premium enterprise atmosphere, slow cinematic camera push-in, smooth natural motion, photorealistic, high facial consistency, shallow depth of field, 4K commercial corporate video quality
```

Recommended negative skeleton:

```text
low resolution, blurry, distorted face, cartoon, anime, cgi, fantasy, holograms, holographic interfaces, floating screens, digital glow, neon overload, cyberpunk overload, sci-fi creatures, text overlay, logos, watermark, unreadable letters, crowd scenes, outdoor settings, daytime, open mouth, teeth, exaggerated smile, messy clothing, bad hands, extra fingers, plastic skin, deformed anatomy, jitter, flickering face, nudity, explicit content
```

### 10.3 Generate Video

```bash
cd /workspace/ARACHNE-X && source .venv/bin/activate
export PYTHONPATH=/workspace/ARACHNE-X
export CKPT=/workspace/ARACHNE-X/weights/ARACHNE-X-ULTRA-VIDEO

cat > /workspace/ARACHNE-X/output/mega_from_nullxes_video_prompt_no_holo.txt <<'EOF'
Elegant brunette woman named Megan, digital executive from NULLXES, standing in a dark luxury futuristic corporate office at night, premium cinematic lighting, deep black and silver interior, thin black glasses, wearing a sleek black tailored business suit, calm closed-mouth expression, direct focused eye contact, realistic skin texture, detailed hair strands, professional executive posture, city skyline visible through large panoramic windows, soft rim light on hair and shoulders, warm overhead office lights, polished dark marble floor, minimalist premium enterprise atmosphere, slow cinematic camera push-in, smooth natural motion, photorealistic, high facial consistency, shallow depth of field, 4K commercial corporate video quality
EOF

cat > /workspace/ARACHNE-X/output/mega_from_nullxes_negative_prompt_no_holo.txt <<'EOF'
low resolution, blurry, distorted face, cartoon, anime, cgi, fantasy, holograms, holographic interfaces, floating screens, digital glow, neon overload, cyberpunk overload, sci-fi creatures, text overlay, logos, watermark, unreadable letters, crowd scenes, outdoor settings, daytime, open mouth, teeth, exaggerated smile, messy clothing, bad hands, extra fingers, plastic skin, deformed anatomy, jitter, flickering face, nudity, explicit content
EOF

/usr/bin/time -f 'elapsed_sec %e' python - <<'PY'
import os
import numpy as np
import torch
from torchvision.io import write_video
from arachne_x.loader import load_base_pipeline

OUT = "/workspace/ARACHNE-X/output"
ckpt = os.environ["CKPT"]

prompt = open(os.path.join(OUT, "mega_from_nullxes_video_prompt_no_holo.txt"), encoding="utf-8").read().strip()
negative = open(os.path.join(OUT, "mega_from_nullxes_negative_prompt_no_holo.txt"), encoding="utf-8").read().strip()

pipe = load_base_pipeline(ckpt, device="cuda", torch_dtype=torch.bfloat16)
pipe.dit.load_lora(os.path.join(ckpt, "lora", "cfg_step_lora.safetensors"), "cfg_step_lora")
pipe.dit.enable_loras(["cfg_step_lora"])

g = torch.Generator(device="cuda").manual_seed(778)
out = pipe.generate_t2v(
    prompt=prompt,
    negative_prompt=negative,
    height=832,
    width=480,
    num_frames=93,
    num_inference_steps=16,
    use_distill=True,
    guidance_scale=1.0,
    generator=g,
)[0]

video = torch.from_numpy(np.array(out))
video = (video * 255).clamp(0, 255).to(torch.uint8)
out_path = os.path.join(OUT, "mega_from_nullxes_no_holo_832x480_93f.mp4")
write_video(out_path, video, fps=30, video_codec="libx264", options={"crf": "18"})
print("wrote", out_path)
PY
```

---

## 11. LoRA Notes

Official VIDEO LoRAs:

```text
weights/ARACHNE-X-ULTRA-VIDEO/lora/cfg_step_lora.safetensors
weights/ARACHNE-X-ULTRA-VIDEO/lora/refinement_lora.safetensors
```

Rules:

- `cfg_step_lora` is for speed/distill: `use_distill=True`, `num_inference_steps=16`, `guidance_scale=1.0`.
- `refinement_lora` is for quality/refinement and is not part of the fast semiauto turn.
- Do not mix unvalidated custom style LoRA with `cfg_step_lora` in production. A 10-image smoke LoRA overfit and produced mosaic artifacts.
- External LoRAs must be trained for **LongCat / ARACHNE base DiT**. SDXL / Flux / Wan / Hunyuan / CogVideo LoRAs are not compatible.

---

## 12. Known Failures and Fixes

### `ModuleNotFoundError: pyloudnorm`

Install:

```bash
pip install pyloudnorm==0.1.1
```

### `torchvision::nms does not exist`

Torch / torchvision mismatch. Restore video environment:

```bash
pip install --force-reinstall --no-cache-dir \
  torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu124
```

### `HF_HUB_ENABLE_HF_TRANSFER=1` but `hf_transfer` missing

```bash
pip install hf_transfer
```

### `sox: not found`

Qwen TTS can still run, but install OS package for clean environment:

```bash
apt-get update && apt-get install -y sox libsox-dev
```

### ASR misrecognizes NULLXES

Expected. Fix via LLM normalization:

```text
Correct brand/name recognition mistakes:
Meg Null must be written as Meg Null.
NULLXES must be written exactly as NULLXES.
```

### Holograms / sci-fi UI appear in video

Remove:

```text
holographic interfaces, floating holographic interfaces, digital glow, neon overload
```

Add to negative prompt:

```text
holograms, holographic interfaces, floating screens, digital glow, neon overload, cyberpunk overload
```

### Native 10s 720p generation is too expensive

Observed cancelled run:

```text
301 frames, 1280x720, 50 steps → projected multi-hour runtime
```

Use fast distill profile for semiauto turns:

```text
93 frames, 832x480, 16 steps, cfg_step_lora
```

---

## 13. Output Directory Contract

All generated artifacts are under:

```text
/workspace/ARACHNE-X/output/
```

Canonical smoke artifacts:

```text
stage3_nullxes_input_serena.wav
stage3_nullxes_asr.txt
stage3_nullxes_llm_reply.txt
stage3_nullxes_chain_tts_serena.wav
mega_from_nullxes_message.txt
mega_from_nullxes_tts_serena.wav
mega_from_nullxes_video_prompt_no_holo.txt
mega_from_nullxes_negative_prompt_no_holo.txt
mega_from_nullxes_no_holo_832x480_93f.mp4
```

---

## 14. Turn Manifest Schema

Future local runner must write one `manifest.json` per turn.

```json
{
  "project": "Project Fury ARACHNE X ULTRA V2",
  "date": "2026-05-13",
  "character": "Megan / Meg Null",
  "input": {
    "text": "optional user text",
    "audio_path": "optional input audio path"
  },
  "asr": {
    "model": "openai/whisper-large-v3-turbo",
    "text": "ASR transcript",
    "elapsed_sec": 1.02
  },
  "llm": {
    "model": "Qwen/Qwen3-4B-Instruct-2507",
    "reply_text": "normalized response",
    "video_prompt_path": "mega_from_nullxes_video_prompt_no_holo.txt",
    "negative_prompt_path": "mega_from_nullxes_negative_prompt_no_holo.txt",
    "elapsed_sec": 2.05
  },
  "tts": {
    "model": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "speaker": "Serena",
    "wav_path": "stage3_nullxes_chain_tts_serena.wav",
    "elapsed_sec": 21.14
  },
  "video": {
    "model": "MagistrTheOne/ARACHNE-X-ULTRA-VIDEO",
    "checkpoint_dir": "/workspace/ARACHNE-X/weights/ARACHNE-X-ULTRA-VIDEO",
    "lora": "cfg_step_lora.safetensors",
    "use_distill": true,
    "height": 832,
    "width": 480,
    "num_frames": 93,
    "num_inference_steps": 16,
    "guidance_scale": 1.0,
    "mp4_path": "mega_from_nullxes_no_holo_832x480_93f.mp4",
    "elapsed_sec": 167.58
  }
}
```

---

## 15. Next Code Step: Local Turn Runner

No worker. No HTTP. The productionizing step is a local CLI:

```bash
python scripts/run_semiauto_turn.py \
  --text "Hello, introduce yourself as Megan from NULLXES." \
  --character megan \
  --out /workspace/ARACHNE-X/output/turn_megan_001
```

Minimum behavior:

```text
input text/audio
→ ASR if audio
→ LLM normalized reply
→ TTS wav
→ LLM video prompt
→ VIDEO generation via .venv subprocess
→ manifest.json
```

This preserves the current no-worker constraint while converting the validated smoke into a reproducible semiautomatic turn.

---

## 16. RunPod Job Runner and HITL Contract

The local runner now supports three execution stages:

```text
execute      → plan + TTS + VIDEO in one call
plan_only    → ASR/LLM/policy only; writes action_plan.json and manifest status pending_approval
execute_plan → execute TTS/VIDEO from an approved action_plan.json
```

### Lightweight healthcheck

```bash
cd /workspace/ARACHNE-X
source .venv/bin/activate
python scripts/run_semiauto_job.py --health
```

### Full auto single job

Create a JSON job under `/workspace/ARACHNE-X/jobs/incoming/megan_001.json`:

```json
{
  "job_id": "megan_001",
  "stage": "execute",
  "text": "Introduce yourself as Megan from NULLXES in one precise line.",
  "character": "megan",
  "safety": "prod",
  "video_profile": "fast_distill_9x16",
  "enable_tts": true,
  "enable_video": true,
  "session_id": "megan_default",
  "retries": 1
}
```

Run once:

```bash
python scripts/run_semiauto_job.py \
  --job /workspace/ARACHNE-X/jobs/incoming/megan_001.json
```

Watch a folder:

```bash
python scripts/run_semiauto_job.py \
  --jobs-dir /workspace/ARACHNE-X/jobs/incoming \
  --watch \
  --poll-sec 5
```

The runner writes sidecar markers:

```text
job.json.running  → job currently executing
job.json.done     → manifest copy for a completed/partial/blocked job
job.json.failed   → manifest copy for a failed job
```

### Human-in-the-loop approval

Step 1: plan only.

```bash
python scripts/run_semiauto_turn.py \
  --stage plan_only \
  --text "Megan should greet the NULLXES team and confirm she is online." \
  --character megan \
  --out /workspace/ARACHNE-X/output/hitl_megan_001
```

Human reviews and edits:

```text
/workspace/ARACHNE-X/output/hitl_megan_001/action_plan.json
```

Step 2: execute the approved plan.

```bash
python scripts/run_semiauto_turn.py \
  --stage execute_plan \
  --approved-action-plan /workspace/ARACHNE-X/output/hitl_megan_001/action_plan.json \
  --approved-by operator \
  --out /workspace/ARACHNE-X/output/hitl_megan_001
```

The manifest now includes:

```text
status              → completed / partial / failed / blocked / pending_approval
validation_notes    → schema/profile/prompt contract issues
errors              → subprocess failure payloads with stdout/stderr log paths
lifecycle           → stage, approval metadata, job/session metadata
qa                  → basic generated artifact checks
```

---

## 17. Qwen Planner LoRA MVP

The planner can be adapted with a small PEFT LoRA trained from synthetic `ActionPlan` examples.

Build a starter dataset:

```bash
cd /workspace/ARACHNE-X
source .venv_stage3/bin/activate
export PYTHONPATH=/workspace/ARACHNE-X

python scripts/build_qwen_sft_synthetic.py \
  --out /workspace/ARACHNE-X/datasets/qwen_sft/furia_eidolon_synthetic \
  --positive 200 \
  --negative 50 \
  --eval-size 25
```

Install training dependency in `.venv_stage3`:

```bash
pip install peft==0.14.0
```

Train a fast LoRA:

```bash
python scripts/train_qwen_planner_lora.py \
  --model-path /workspace/ARACHNE-X/weights/Qwen3-4B-Instruct-2507 \
  --train-jsonl /workspace/ARACHNE-X/datasets/qwen_sft/furia_eidolon_synthetic/train.jsonl \
  --eval-jsonl /workspace/ARACHNE-X/datasets/qwen_sft/furia_eidolon_synthetic/eval.jsonl \
  --output-dir /workspace/ARACHNE-X/output/qwen_planner_lora/furia_eidolon_synth_r16 \
  --rank 16 \
  --alpha 32 \
  --epochs 2 \
  --batch-size 1 \
  --grad-accum 8 \
  --lr 1e-4 \
  --max-length 2048 \
  --attn flash_attention_2
```

Use the adapter in a turn:

```bash
python scripts/run_semiauto_turn.py \
  --stage plan_only \
  --text "Megan is online. Confirm readiness without hype." \
  --character megan \
  --planner-lora /workspace/ARACHNE-X/output/qwen_planner_lora/furia_eidolon_synth_r16 \
  --out /workspace/ARACHNE-X/output/planner_lora_smoke
```

For job runner JSON, use:

```json
{
  "planner_lora": "/workspace/ARACHNE-X/output/qwen_planner_lora/furia_eidolon_synth_r16"
}
```

---

## 18. Current Production Position

**ARACHNE X ULTRA V2 / Project Fury** is not yet a hosted worker system in this playbook. It is a validated local RunPod semiautomatic pipeline:

```text
voice + language + synthetic video
with reproducible model paths,
separated environments,
known latency,
and a clear next step toward turn orchestration.
```

**NULLXES** · Проект Фурия ARACHNE X ULTRA V2 · полуавтоматическая нейросетевая жизнь · RunPod-only smoke-to-orchestration playbook.
