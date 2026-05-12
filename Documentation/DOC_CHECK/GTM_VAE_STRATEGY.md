# ARACHNE-X GTM — стратегия замены VAE / tokenizer

Цель: улучшить визуальное качество без «тихого» ломания чекпоинтов.

## Путь A — совместимый VAE (рекомендуемый для staged rollout)

**Идея:** новый энкодер/декодер, но **тот же** `z_dim`, temporal/spatial compression, семантика латента.

**Плюсы:** минимальный retune DiT (LoRA или короткий full FT), существующие latent training shards остаются валидными при совпадении шейпов.

**Минусы:** архитектурно ограничены текущим bottleneck.

**Шаги:** см. `GTM_VAE_ABI.md` → reconstruction gate → light DiT alignment → avatar head при необходимости.

## Путь B — новый tokenizer / другой latent ABI

**Идея:** другой `z_dim`, другие compression ratios, или discrete tokenizer (VQ) вместо KL-Gaussian.

**Плюсы:** потенциально лучше качество/скорость sequence length.

**Минусы:** **breaking** — полный цикл: новый VAE pretrain/finetune → DiT с нуля или heavy FT → переэкспорт всех training latents → обновление inference/export/train.

**Обязательные артефакты:** latent adapter (1×1 conv / linear bridge) *только как краткий мост* между старым и новым пространством — для production лучше end-to-end без adapter.

## Внешние ориентиры (не привязка к лицензии без проверки)

Для сравнения архитектур и API в экосистеме diffusers:

- **Wan / AutoencoderKLWan** — текущая база; см. документацию diffusers `AutoencoderKLWan`.
- **CogVideoX** — `AutoencoderKLCogVideoX` (3D causal VAE, другой ABI).
- **HunyuanVideo** — `AutoencoderKLHunyuanVideo`.
- **NVIDIA Cosmos tokenizer** — continuous/discrete video tokenizers (другая парадигма; интеграция = путь B).
- **Open-MAGVIT2 / MAGVIT** — discrete tokens; путь B + AR модель или hybrid.

Выбор: **Path A** до исчерпания качества; **Path B** когда нужен step-change и есть бюджет на полный retrain + новый data flywheel.
