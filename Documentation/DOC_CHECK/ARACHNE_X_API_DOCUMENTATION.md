# ARACHNE-X API Documentation для деплоя

**Дата:** 20.02.2026  
**Версия:** 1.0

---

## 🚀 БЫСТРЫЙ СТАРТ: Деплой на GPU

### 1. Инициализация Pipeline (загрузка на GPU)

```python
from arachne_x.loader import load_avatar_pipeline
import torch

# Базовая загрузка
device = "cuda"  # или "cuda:0", "cuda:1" для multi-GPU
torch_dtype = torch.bfloat16  # рекомендуется для H200

pipe = load_avatar_pipeline(
    checkpoint_dir="./weights/ARACHNE-X-Avatar",  # путь к весам модели
    variant="single",  # или "multi" для multi-person avatars
    device=device,
    torch_dtype=torch_dtype,
)

# Pipeline автоматически загружается на указанный GPU
# Все компоненты (DiT, VAE, Text Encoder, Audio Encoder) на GPU
```

**Что происходит внутри:**
- Загружаются веса: `tokenizer/`, `text_encoder/`, `vae/`, `scheduler/`, `avatar_single/`, `audio/wav2vec2/`
- Все модели перемещаются на GPU через `.to(device)`
- Используется `torch.bfloat16` для экономии памяти на H200

---

### 2. Структура весов (checkpoint_dir)

```
weights/ARACHNE-X-Avatar/
├── tokenizer/              # UMT5 tokenizer
├── text_encoder/           # UMT5 text encoder
├── vae/                    # AutoencoderKLWan (video VAE)
├── scheduler/              # FlowMatchEulerDiscreteScheduler
├── avatar_single/         # LongCatVideoAvatarTransformer3DModel (single person)
├── avatar_multi/          # LongCatVideoAvatarTransformer3DModel (multi person)
└── audio/
    ├── wav2vec2/          # Wav2Vec2 audio encoder
    └── vocal_separator/   # ONNX vocal separator (опционально)
```

---

## 📡 API МЕТОДЫ ДЛЯ ГЕНЕРАЦИИ

### Метод 1: `generate_streaming_ai2v()` — Real-time Streaming (рекомендуется для интервью)

**Назначение:** Генерация аватара в реальном времени по аудио-потоку.

```python
# Генератор аудио-чанков (0.5 сек каждый)
def audio_stream_generator(audio_path: str, chunk_duration: float = 0.5):
    audio, sr = librosa.load(audio_path, sr=16000)
    chunk_samples = int(chunk_duration * sr)
    for i in range(0, len(audio), chunk_samples):
        chunk = audio[i:i+chunk_samples]
        if len(chunk) < chunk_samples:
            chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))
        yield chunk

# Использование
from diffusers.utils import load_image

image = load_image("hr_avatar_reference.png")  # Референсное фото HR
audio_gen = audio_stream_generator("candidate_speech.wav")

# Генерация кадров по одному (streaming)
for frame_np in pipe.generate_streaming_ai2v(
    image=image,
    prompt="A professional HR interviewer speaking naturally",
    audio_stream=audio_gen,
    resolution="480p",  # или "720p"
    num_frames=93,  # длина видео в кадрах
    num_inference_steps=8,  # distilled mode (быстро)
    text_guidance_scale=4.0,
    audio_guidance_scale=4.0,
    identity_id=0,  # ID HR из identity bank (опционально)
    identity_strength=1.0,  # сила identity (0.0-1.0)
    emotion_id="neutral",  # или "happy", "calm" и т.д.
    emotion_intensity=0.5,  # интенсивность эмоции (0.0-1.0)
):
    # frame_np - numpy array [H, W, 3] в диапазоне [0, 1]
    # Можно сразу отправлять в WebRTC/WebSocket для стриминга
    send_frame_to_client(frame_np)
```

**Параметры:**
- `image`: PIL.Image или путь к изображению (референс HR)
- `prompt`: текстовое описание (стиль, контекст)
- `audio_stream`: генератор аудио-чанков (numpy arrays, 16kHz)
- `resolution`: "480p" (480x832) или "720p" (768x1280)
- `num_frames`: количество кадров (93 = ~3 сек при 30 FPS)
- `num_inference_steps`: шаги денойзинга (8 = fast, 20-50 = качество)
- `identity_id`: ID из identity bank (для конкретного HR)
- `emotion_id`: "neutral", "happy", "sad", "angry", "surprised", "fearful", "disgusted", "calm"

**Возвращает:** Генератор numpy arrays (каждый кадр отдельно)

**Производительность:** 30 FPS на H200, latency ~33ms на кадр

---

### Метод 2: `generate_ai2v()` — Batch Audio-to-Video (оффлайн)

**Назначение:** Генерация полного видео из аудио файла (не streaming).

```python
from diffusers.utils import load_image

image = load_image("hr_avatar_reference.png")
audio_path = "full_interview.wav"  # полный аудио файл

# Генерация всего видео сразу
video_frames = pipe.generate_ai2v(
    image=image,
    prompt="A professional HR interviewer",
    audio=audio_path,  # путь к аудио или numpy array
    resolution="480p",
    num_frames=600,  # 20 секунд при 30 FPS
    num_inference_steps=20,
    text_guidance_scale=4.0,
    audio_guidance_scale=4.0,
    identity_id=0,
    identity_strength=1.0,
)

# video_frames - numpy array [T, H, W, 3] в диапазоне [0, 1]
# Сохранение видео
save_video(video_frames, "output.mp4", fps=30)
```

**Параметры:** Аналогично `generate_streaming_ai2v()`, но `audio` вместо `audio_stream`

**Возвращает:** numpy array [T, H, W, 3] со всеми кадрами

---

### Метод 3: `generate_at2v()` — Text + Audio to Video

**Назначение:** Генерация с текстовым промптом + аудио (для кастомных сценариев).

```python
video_frames = pipe.generate_at2v(
    image=image,
    prompt="A friendly HR interviewer asking questions",
    audio=audio_path,
    resolution="480p",
    num_frames=93,
    num_inference_steps=20,
    text_guidance_scale=4.0,
    audio_guidance_scale=4.0,
)
```

---

### Метод 4: `generate_avc()` — Avatar Video Continuation

**Назначение:** Продолжение существующего видео аватара (для длинных интервью).

```python
from diffusers.utils import load_video

reference_video = load_video("previous_segment.mp4")

video_frames = pipe.generate_avc(
    video=reference_video,
    prompt="Continuing the interview",
    audio=audio_path,
    resolution="480p",
    num_frames=93,
    num_cond_frames=13,  # сколько кадров использовать как контекст
    num_inference_steps=20,
    text_guidance_scale=4.0,
    audio_guidance_scale=4.0,
)
```

---

## 🔧 ДОПОЛНИТЕЛЬНЫЕ ВОЗМОЖНОСТИ

### Identity Bank (банк идентичностей HR)

```python
# Установка identity для конкретного HR
pipe.set_identity(
    identity_id=0,
    strength=1.0,  # сила влияния identity (0.0-1.0)
    negative_strength=0.0,  # negative guidance для identity
)

# Обновление identity bank во время генерации
pipe.update_identity_bank(
    identity_id=0,
    new_latent=reference_latent,  # новый латент для обновления
    momentum=0.25,  # скорость обновления
)
```

### Emotion Control (управление эмоциями)

```python
# Установка эмоции
pipe.set_emotion(
    emotion_id="happy",  # или "neutral", "calm", "surprised" и т.д.
    intensity=0.7,  # интенсивность (0.0-1.0)
    guidance_scale=0.5,  # сила влияния на генерацию
)
```

### Phoneme Conditioning (фонемная синхронизация губ)

```python
# Включение/выключение фонемной синхронизации
pipe.phoneme_enabled = True
pipe.phoneme_stream_scale = 0.20  # вес фонемного потока
```

---

## 🖥️ МУЛЬТИ-GPU ДЕПЛОЙ

### Вариант 1: Один GPU на сессию (рекомендуется для production)

```python
import os
import torch

# Каждая сессия интервью использует свой GPU
session_gpu_id = int(os.environ.get("SESSION_GPU_ID", 0))
device = f"cuda:{session_gpu_id}"

pipe = load_avatar_pipeline(
    checkpoint_dir="./weights/ARACHNE-X-Avatar",
    device=device,
    torch_dtype=torch.bfloat16,
)
```

### Вариант 2: Context Parallel (для больших моделей)

```python
from arachne_x.loader import load_avatar_pipeline

# Context parallel split для распределения внимания
cp_split_hw = (2, 2)  # разделение на 4 части

pipe = load_avatar_pipeline(
    checkpoint_dir="./weights/ARACHNE-X-Avatar",
    device="cuda",
    torch_dtype=torch.bfloat16,
    cp_split_hw=cp_split_hw,  # для multi-GPU через context parallel
)
```

---

## 📊 МОНИТОРИНГ И ПРОИЗВОДИТЕЛЬНОСТЬ

### Проверка использования памяти GPU

```python
import torch

# До загрузки
print(f"GPU Memory Before: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

pipe = load_avatar_pipeline(...)

# После загрузки
print(f"GPU Memory After: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"GPU Memory Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
```

**Ожидаемое использование на H200:**
- Модель: ~27 GB (bfloat16)
- KV-cache (streaming): ~45 GB (8 frames × 12 timesteps)
- Intermediate activations: ~30 GB
- **Итого: ~110-120 GB из 141 GB HBM3e**

### Бенчмарк производительности

```python
from benchmark_realtime import RealtimeBenchmark

benchmark = RealtimeBenchmark(pipe, device="cuda")

# Бенчмарк одного кадра
metrics = benchmark.benchmark_single_frame(
    latents=test_latents,
    audio_emb=test_audio_emb,
    prompt_embeds=test_prompt_embeds,
    prompt_mask=test_prompt_mask,
    timestep=test_timestep,
    num_runs=100,
)

print(f"Average FPS: {metrics['avg_fps']:.1f}")
print(f"P95 Latency: {metrics['p95_latency_ms']:.2f} ms")
```

---

## 🔌 ИНТЕГРАЦИЯ С ВНЕШНИМИ СЕРВИСАМИ

### Пример: REST API сервер (FastAPI)

```python
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
import numpy as np
from PIL import Image
import io

app = FastAPI()

# Глобальный pipeline (загружается при старте сервера)
pipe = None

@app.on_event("startup")
async def startup():
    global pipe
    pipe = load_avatar_pipeline(
        checkpoint_dir="./weights/ARACHNE-X-Avatar",
        device="cuda",
        torch_dtype=torch.bfloat16,
    )

@app.post("/generate_streaming")
async def generate_streaming(
    image: UploadFile = File(...),
    audio: UploadFile = File(...),
    prompt: str = "A professional HR interviewer",
    identity_id: int = 0,
):
    # Загрузка изображения
    image_data = await image.read()
    img = Image.open(io.BytesIO(image_data))
    
    # Загрузка аудио
    audio_data = await audio.read()
    audio_np = np.frombuffer(audio_data, dtype=np.float32)
    
    # Генерация streaming
    def frame_generator():
        audio_gen = audio_chunk_generator(audio_np)
        for frame in pipe.generate_streaming_ai2v(
            image=img,
            prompt=prompt,
            audio_stream=audio_gen,
            resolution="480p",
            num_frames=93,
            identity_id=identity_id,
        ):
            # Конвертация в bytes для отправки
            frame_bytes = (frame * 255).astype(np.uint8).tobytes()
            yield frame_bytes
    
    return StreamingResponse(frame_generator(), media_type="video/mp4")
```

### Пример: WebSocket для real-time стриминга

```python
from fastapi import WebSocket
import asyncio

@app.websocket("/ws/avatar_stream")
async def websocket_avatar_stream(websocket: WebSocket):
    await websocket.accept()
    
    # Получение референсного изображения
    image_data = await websocket.receive_bytes()
    img = Image.open(io.BytesIO(image_data))
    
    # Генератор аудио из WebSocket
    async def audio_stream():
        while True:
            audio_chunk = await websocket.receive_bytes()
            if not audio_chunk:
                break
            yield np.frombuffer(audio_chunk, dtype=np.float32)
    
    # Генерация и отправка кадров
    async for frame in pipe.generate_streaming_ai2v(
        image=img,
        prompt="A professional HR interviewer",
        audio_stream=audio_stream(),
        resolution="480p",
    ):
        frame_bytes = (frame * 255).astype(np.uint8).tobytes()
        await websocket.send_bytes(frame_bytes)
```

---

## ⚠️ ВАЖНЫЕ ЗАМЕЧАНИЯ ДЛЯ ДЕПЛОЯ

### 1. Управление памятью

```python
# Очистка кэша после генерации
import gc
torch.cuda.empty_cache()
gc.collect()
```

### 2. Обработка ошибок

```python
try:
    pipe = load_avatar_pipeline(...)
except RuntimeError as e:
    if "out of memory" in str(e):
        # Уменьшить batch size или использовать CPU offloading
        print("GPU memory insufficient, consider using CPU offloading")
```

### 3. Оптимизация для production

```python
# Включение inference mode для ускорения
pipe.dit.eval()
pipe.vae.eval()
pipe.text_encoder.eval()

# Torch compile для ускорения (PyTorch 2.0+)
if hasattr(torch, 'compile'):
    pipe.dit = torch.compile(pipe.dit, mode='reduce-overhead')
    pipe.vae = torch.compile(pipe.vae, mode='reduce-overhead')
```

---

## 📝 ЧЕКЛИСТ ДЕПЛОЯ

- [ ] Веса модели загружены в `checkpoint_dir`
- [ ] GPU доступен (`torch.cuda.is_available()`)
- [ ] Достаточно VRAM (минимум 120 GB для H200)
- [ ] Установлены зависимости (`requirements.txt` + `requirements_avatar.txt`)
- [ ] Pipeline загружен и протестирован на тестовых данных
- [ ] API сервер настроен (REST/WebSocket)
- [ ] Мониторинг производительности включен
- [ ] Обработка ошибок реализована

---

**Контакты для технической поддержки:**
- Документация: `Documentation/ARACHNE-X_IMPLEMENTATION_SUMMARY.md`
- Примеры использования: `Demo/run_demo_streaming_realtime.py`
