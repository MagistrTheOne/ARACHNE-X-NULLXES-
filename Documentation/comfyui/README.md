# NULLXES ARACHNE-X — ComfyUI

## Workflow (loaders only)

**[`workflows/NULLXES_ARACHNE_loaders.json`](workflows/NULLXES_ARACHNE_loaders.json)** — только загрузка моделей + опциональные входы. Дальше сам допаиваешь sampler / save.

Требует: [ComfyUI-WanVideoWrapper `longcat_avatar`](https://github.com/kijai/ComfyUI-WanVideoWrapper/tree/longcat_avatar).

```
ComfyUI → Load → NULLXES_ARACHNE_loaders.json
```

## Веса

Comfy **не** читает `weights/arachne-avatar-runtime/` (HF shards). Нужны Kijai single-file: https://huggingface.co/Kijai/LongCat-Video_comfy

Production: `scripts/infer.py` + `NULLXES_CHECKPOINT_DIR` на RunPod.
