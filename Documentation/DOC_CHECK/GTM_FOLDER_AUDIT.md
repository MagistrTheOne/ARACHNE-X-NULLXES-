# ARACHNE-X GTM — аудит папок (keep / refactor / archive)

Статусы: **keep** — опираемся на GTM; **refactor** — нужна доработка без удаления; **archive** — вынести из prod path (examples, legacy).

## `arachne_x/` — **keep** (ядро SDK)

- Публичный вход: [`arachne_x/loader.py`](../../arachne_x/loader.py), пайплайны, `weights_resolve`.
- Новый программный слой: [`arachne_x/runtime/`](../../arachne_x/runtime/) — инференс без CLI.

## `arachne_x/modules/` — **keep** + **refactor**

- **keep:** DiT, VAE (`autoencoder_kl_wan`), scheduler, attention, avatar blocks.
- **refactor:** при смене VAE/tokenizer — согласовать `in_channels`, `z_dim`, `scale_factor_*`, `normalize_latents` (см. `GTM_VAE_ABI.md`).

## `scripts/` — **refactor**

| Файл | Роль | GTM |
|------|------|-----|
| `infer.py` | Тонкая обёртка над `arachne_x.runtime` | **keep** как CLI |
| `train.py`, `train_lora_avatar.py` | Обучение на латентах | **keep**; позже вынести в `runtime/train_engine` |
| `arachne_x_train.py` | Launcher env → subprocess | **keep** |
| `run_webrtc_server.py` | Сервер | **keep** |
| `export_latent_*.py`, `pack_latent_shards_wds.py` | Датапайплайн | **keep** |
| `run_fast_*.sh`, персональные one-off | Хардкод путей | **archive** → `examples/` или удалить позже |

## `Demo/` — **archive** (не prod path)

- `run_demo_*.py`: демо под `torchrun`/distributed; для GTM — только smoke/regression.
- `training_config_h200.py`: большая часть полей aspirational (см. docstring в файле); реально читают `train.py` / LoRA — **refactor** при появлении единого train config.
- `run_streamlit.py`: UI для демо — **archive**.

## `docker/` — **keep** + **refactor**

- [`docker/Dockerfile.gpu`](../../docker/Dockerfile.gpu): воспроизводимая среда — **keep**.
- [`docker/compose.gpu.yml`](../../docker/compose.gpu.yml): тома `./weights` относительно `docker/` — **refactor** (документировать абсолютные пути для клиентов).

## `data/` — **keep**

- [`data/datasets/README.md`](../../data/datasets/README.md): контракт сырых данных + HF — **keep**.

## `config/` — **keep**

- [`config/pipeline_config.defaults.json`](../../config/pipeline_config.defaults.json), `runpod.example.json` — источник правды для сервера/orchestration — **keep**; синхронизировать с продуктовой матрицей env.

## `src/server/` — **keep** (вне этого плана)

- Control-plane / WS — продолжать развивать отдельно от `scripts/`.

## Итог

- **Prod library path:** `arachne_x/` + `arachne_x/runtime/`.
- **Prod CLI thin:** `scripts/infer.py`, `scripts/run_webrtc_server.py`.
- **Не тащить в enterprise SLA:** `Demo/*` как единственный вход.
