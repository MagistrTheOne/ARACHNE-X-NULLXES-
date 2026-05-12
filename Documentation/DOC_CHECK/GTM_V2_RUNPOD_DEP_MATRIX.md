# RunPod dependency matrix — NIGHT FURY V2

**Политика:** только **пиннутые** версии в `requirements*.txt`; без `latest` в production слое.  
Образ базы: [`docker/Dockerfile.gpu`](../../docker/Dockerfile.gpu) — `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime`.

## Текущий целевой стек (после unify requirements)

| Component | Pinned version (repo) | Notes |
|-----------|------------------------|-------|
| Python | 3.11.x (из базового образа torch) | Зафиксировать из фактического RunPod template |
| CUDA (driver / runtime) | 12.4 (image) | B200 может потребовать отдельную строку — заполнить после прогона |
| PyTorch | 2.6.0 | |
| torchvision | 0.21.0 | |
| triton | (bundled with torch wheel) | Не пинить отдельно, если не требуется |
| flash-attn | 2.7.4.post1 (Linux) | Должен совпадать с CUDA образа |
| xformers | optional | Нет в base `requirements.txt`; при `enable_xformers` — добавить пин под torch+CUDA |
| diffusers | 0.35.1 | |
| transformers | 4.41.0 | Конфликт с optional `requirements-audiodit` (`transformers>=5.3`) — **не ставить audiodit в тот же venv**, что core V2, либо отдельный env |
| accelerate | 1.12.0 | |
| onnxruntime | 1.18.0 (`requirements_avatar`) | CPU ORT для vocal separator; GPU ORT — отдельное решение |
| safetensors | 0.7.0 | |
| librosa | 0.11.0 | Унифицировано с avatar |
| av | 13.1.0 | Унифицировано с avatar |
| faster-whisper | 1.1.1 | Унифицировано |
| soundfile | 0.13.1 | Унифицировано |

## GPU class matrix (заполнить после RunPod прогонов)

| GPU class | Template ID | CUDA driver | torch | flash-attn | smoke OK | Date |
|-----------|-------------|-------------|-------|--------------|----------|------|
| H200 | | | 2.6.0 | 2.7.4.post1 | | |
| H100 | | | 2.6.0 | 2.7.4.post1 | | |
| B200 | | | TBD | TBD | | |

---

**NULLXES** · dep matrix · обновлять после каждого bump зависимостей
