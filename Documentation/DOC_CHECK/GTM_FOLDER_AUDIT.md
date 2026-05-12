# ARACHNE-X-ULTRA V2 — «Ночная Фурия» — аудит папок (keep / refactor / archive)

Статусы: **keep** — опираемся на GTM; **refactor** — доработка без удаления; **archive** — вне prod path (examples, legacy). Продуктовая линия: **stateful realtime avatar system** — см. [`GTM_ULTRA_V2_NIGHT_FURY.md`](GTM_ULTRA_V2_NIGHT_FURY.md).

**Канонический production-контур:** см. [`GTM_PRODUCTION_CONTRACT.md`](GTM_PRODUCTION_CONTRACT.md) — `arachne_x.runtime` + Inference Worker (`services/longcat-worker/`); `src/server/` = **internal** оркестрация.

---

## `arachne_x/` — **keep** (ядро SDK)

- Публичный вход: [`arachne_x/loader.py`](../../arachne_x/loader.py), пайплайны, `weights_resolve`.
- **Основной программный слой инференса:** [`arachne_x/runtime/`](../../arachne_x/runtime/) — без дублирования логики в CLI.

## `arachne_x/modules/` — **keep** + **refactor**

- **keep:** DiT, VAE (`autoencoder_kl_wan`), scheduler, attention, аватарные блоки.
- **refactor:** при смене VAE/tokenizer — согласовать `in_channels`, `z_dim`, `scale_factor_*`, `normalize_latents` (`GTM_VAE_ABI.md`).

## `scripts/` — **refactor**

| Файл | Роль | GTM |
|------|------|-----|
| `infer.py` | Тонкая обёртка над `arachne_x.runtime` | **keep** (CLI) |
| `train.py`, `train_lora_avatar.py` | Обучение на латентах | **keep**; целевой вынос — `arachne_x/train/` (см. `GTM_PRODUCTION_CONTRACT.md` §6) |
| `arachne_x_train.py` | Launcher env → subprocess | **keep** |
| `run_webrtc_server.py` | Точка входа сервера продукта | **keep** (вне детализации в GTM) |
| `export_latent_*.py`, `pack_latent_shards_wds.py` | Датапайплайн | **keep** |
| `run_fast_*.sh`, персональные one-off | Хардкод путей | **archive** → `examples/` или удаление по релизу |

## `Demo/` — **archive** (не prod path)

- `run_demo_*.py`, `run_streamlit.py` — smoke / внутренние демо, не enterprise SLA.

## `docker/` — **keep** + **refactor**

- [`docker/Dockerfile.gpu`](../../docker/Dockerfile.gpu) — **keep**.
- [`docker/compose.gpu.yml`](../../docker/compose.gpu.yml) — **refactor**: тома и пути к весам документировать явно; три роли образов — `GTM_PRODUCTION_CONTRACT.md` §7.

## `data/` — **keep**

- [`data/datasets/README.md`](../../data/datasets/README.md).

## `config/` — **keep**

- [`config/pipeline_config.defaults.json`](../../config/pipeline_config.defaults.json) и примеры — синхронизация с URL Inference Worker и prod-матрицей env.

## `services/longcat-worker/` — **keep** (prod serving)

- **Inference Worker** — канон GPU-исполнения для стрима/MP4 по HTTP API репозитория. Имя каталога — legacy internal identifier.

## `src/server/` — **internal** (оркестрация)

- Не второй владелец загрузки DiT на GPU при топологии «один inference-процесс»; вызовы к Inference Worker по конфигу.

## Итог

- **Prod library:** `arachne_x/` + **`arachne_x/runtime/`**.
- **Prod CLI:** `scripts/infer.py` (thin).
- **Prod GPU serving:** Inference Worker в `services/longcat-worker/`.
- **Не prod-only вход:** `Demo/*`.
