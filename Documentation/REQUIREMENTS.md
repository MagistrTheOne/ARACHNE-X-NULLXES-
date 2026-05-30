# ARACHNE-X Dependencies (2026-05-27)

Operational dependency contract: what to install for **prod infer**, **orchestrator**, **worker**, and what stays in **lab/training** venvs.

Install order on Linux RunPod is **mandatory** — see [`NULLXES_ARACHNE_RUNPOD_27-05-2026.md`](NULLXES_ARACHNE_RUNPOD_27-05-2026.md) §4.

---

## File matrix

| File | When | Role |
|------|------|------|
| [`requirements.txt`](../requirements.txt) | After torch + flash-attn | Core GPU stack: torch 2.6, diffusers 0.35.1, transformers **4.41.0**, librosa, soundfile, einops, imageio |
| [`requirements_avatar.txt`](../requirements_avatar.txt) | GPU worker / `infer.py` | `-r requirements.txt` + soxr (librosa resampling) |
| [`requirements_orchestrator.txt`](../requirements_orchestrator.txt) | `src/server` gateway | aiohttp, faster-whisper — **not** on dumb GPU-only worker pods |
| [`services/arachnex-worker/requirements.txt`](../services/arachnex-worker/requirements.txt) | Worker HTTP | fastapi, uvicorn, pydantic |
| [`requirements-training.txt`](../requirements-training.txt) | Latent export / WDS | webdataset, opencv, av, sklearn, scikit-image |
| [`requirements-datasets.txt`](../requirements-datasets.txt) | Dataset prep scripts | HF `datasets`, pandas |

---

## Pinned core (prod avatar runtime)

| Component | Version |
|-----------|---------|
| Python | 3.10 or 3.11 (3.10 on RunPod) |
| PyTorch | 2.6.0+cu124 |
| torchvision | 0.21.0+cu124 |
| flash-attn | 2.7.4.post1 (Linux only) |
| diffusers | 0.35.1 |
| transformers | **4.41.0** |
| numpy | 1.26.4 |

---

## Removed from prod (2026-05-27 audit)

These were in older `requirements*.txt` but **no import** on canonical prod path (`avatar_serving`, `pipeline_arachne_x_video_avatar`, `src/server/*`):

| Package | Was listed for | Action |
|---------|----------------|--------|
| streamlit | Demo UI | Removed — no imports |
| pyarrow | Unused | Moved to `requirements-training.txt` |
| opencv-python, av | Training latent export only | Moved to `requirements-training.txt` |
| webdataset | `training_wds.py` only | Moved to `requirements-training.txt` |
| scikit-learn, scikit-image | No arachne_x imports | Moved to training |
| openai, chardet, cffi | `prompt_enhancer.py` (off in prod) | Removed |
| onnx, onnxruntime, audio-separator | Demo vocal separator path | Removed from prod |
| aiortc, silero-vad | Demo WebRTC / semiauto | Removed |
| nvidia-ml-py, tzdata | Listed but not imported (health uses `torch.cuda`) | Removed |
| qwen-tts, AudioDiT, edge-tts | `arachne_x.tts` / `arachne_x.speech` (deleted) | Removed — in-tree TTS gone; use external TTS |

**Note:** `Kim_Vocal_2.onnx` remains in checkpoint bundle layout; runtime does not load onnxruntime on prod streaming path.

**TTS removed (2026-05-30):** `arachne_x/tts/` (Qwen + LongCat-AudioDiT) and `arachne_x/speech/` (edge-tts / espeak providers) were deleted. The orchestrator (`src/server/tts_runner.py`) is now an external-TTS seam; CLI requires `--audio`.

---

## Install recipes

### GPU worker pod (RunPod)

```bash
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install ninja packaging psutil wheel
MAX_JOBS=8 pip install flash-attn==2.7.4.post1 --no-build-isolation
pip install -r requirements_avatar.txt
pip install -r services/arachnex-worker/requirements.txt
```

### Orchestrator + gateway (CPU or separate pod)

```bash
pip install -r requirements_orchestrator.txt
# TTS: in-tree backends removed — wire an external TTS service and add its client deps.
```

### Training / export only

```bash
pip install -r requirements-training.txt
# or full dataset prep:
pip install -r requirements-datasets.txt
```

---

## Version conflict policy

| Track | transformers | Co-install with avatar core? |
|-------|--------------|------------------------------|
| Avatar core | 4.41.0 | Yes |

---

## Import verification (prod paths)

Verified against repo imports on 2026-05-27:

- **Worker / infer:** `torch`, `transformers`, `diffusers`, `einops`, `librosa`, `soundfile`, `pyloudnorm`, `Pillow`, `imageio`, `tqdm`, `numpy`, `scipy`, `loguru`, `safetensors`, `accelerate`, `huggingface_hub`
- **Orchestrator:** `aiohttp`, `faster_whisper` (+ core stack if shared venv); TTS client deps added when an external TTS is wired
- **Worker HTTP:** `fastapi`, `uvicorn`, `pydantic`
