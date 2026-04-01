## ARACHNE-X — схема деплоя (H200/H100/A100)

### 1. Что деплоим

- **Код ядра**: каталог `arachne_x/` + `scripts/` + `Demo/` (по желанию).
- **Веса моделей**:
  - vendor: `weights/LongCat-Video/` (или снапшот с другим именем);
  - свои: `weights/arachnex/*` или `checkpoints/<run_id>/*` (base/avatar).
- **Зависимости**: `requirements.txt` + `requirements_avatar.txt` (опционально).

Все машины в кластере видят:
- общий read-only шэр с кодом (`/opt/arachne_x` условно),
- общий или локальный путь с весами (`/mnt/weights` условно).

### 2. Базовый деплой на один H200 (offline / batch-инференс)

1. **Подготовка образа / окружения**
   - Установить CUDA‑совместимый PyTorch (для H200/H100/A100).
   - `pip install -r requirements.txt`
   - При необходимости: `pip install -r requirements_avatar.txt`.

2. **Развернуть код**
   - Склонировать/скопировать репозиторий в, например, `/opt/arachne_x`.

3. **Развернуть веса**
   - Скопировать/смонтировать:
     - `/mnt/weights/LongCat-Video/` (vendor),
     - `/mnt/weights/arachnex/` или `/mnt/checkpoints/` (свои чекпоинты).

4. **Инференс (CLI)**
   - Базовое видео (T2V/I2V/VC):
     - `python scripts/infer.py --mode t2v --checkpoint_dir /mnt/weights/LongCat-Video ...`
   - Аватар (audio‑to‑video, streaming avatar):
     - `python scripts/infer.py --mode ai2v --checkpoint_dir /mnt/checkpoints/avatar_single ...`

Всё управление версиями и конкретным пайплайном идёт через `--checkpoint_dir` и параметры `infer.py`.

### 3. Multi‑GPU и streaming (real‑time)

- **Multi‑GPU**:
  - Запуск через `torchrun` с `--nproc_per_node` и `--context_parallel_size`, пайплайн сам настраивает Ulysses‑split по `cp_split_hw`.
  - Код для context-parallel уже внутри `arachne_x/context_parallel/*`.

- **Streaming‑аватар**:
  - Используется `ArachneXVideoAvatarPipeline.generate_streaming_ai2v` + движок `streaming_inference.py`.
  - На уровне деплоя: поднимается Python‑процесс, который:
    - грузит avatar‑пайплайн через `load_avatar_pipeline(checkpoint_dir=...)`,
    - держит модель в GPU‑памяти,
    - принимает аудио‑стрим (WebSocket/GRPC/HTTP‑chunked — реализуется снаружи),
    - стримит сгенерированные кадры наружу (raw frames / encoded video).

### 4. Production‑сервис поверх ARACHNE-X

ARACHNE-X даёт **Python‑API и CLI**, а не готовый HTTP‑сервер. Production‑слой строится поверх:

- **Сервис‑обёртка** (микросервис):
  - Выделенный долгоживущий процесс per‑GPU/под.
  - При старте:
    - инициализирует PyTorch, CUDA,
    - вызывает `load_base_pipeline(...)` и/или `load_avatar_pipeline(...)`,
    - прогревает модель (warm‑up запрос).
  - Экспортирует API:
    - sync/async endpoints под T2V/I2V/VC/аватары,
    - streaming‑endpoint под `generate_streaming_ai2v`.

- **Скалирование**:
  - масштабирование по GPU‑нодам или подам;
  - внешняя оркестрация (K8s/Slurm/свой планировщик) распределяет запросы по живым пайплайнам;
  - state (identity‑bank, аккаунты, биллинг) хранится снаружи (DB/kv‑store), ARACHNE-X остаётся stateless‑инференсом + локальный кэш KV внутри процесса.

### 5. Обучение (train) vs инференс в проде

- **Обучение**:
  - запускается отдельно от прод‑кластеров, на выделенных H200/H100/A100;
  - использует `scripts/train.py` с:
    - `--checkpoint_dir` (стартовые веса: vendor или свои),
    - `--dataset_dir` (латенты в формате, который ждёт `LatentDataset`),
    - `--output_dir` (новые чекпоинты).
  - новые веса после валидации выкладываются в `weights/arachnex/` или `checkpoints/<tag>/`.

- **Инференс**:
  - прод‑сервис никогда не пишет веса, только читает;
  - переключение версии модели = смена `checkpoint_dir` (через конфиг/ENV) и перезапуск сервиса/подов.

### 6. Минимальный путь деплоя (кратко)

1. Собрать Docker‑образ или Anaconda‑env с зависимостями и CUDA‑совместимым PyTorch.
2. Положить код ARACHNE-X на машину/в образ (`/opt/arachne_x`).
3. Положить веса (vendor + свои) в общий путь (`/mnt/weights/...`).
4. Написать тонкий сервис (FastAPI/GRPC/WebSocket), который:
   - при старте грузит нужный пайплайн через `load_base_pipeline`/`load_avatar_pipeline`,
   - реализует HTTP/WS‑методы поверх методов пайплайна (`generate_*`).
5. Завернуть сервис в оркестратор (K8s/другой) и настроить health‑checks, авто‑рестарт, логирование.

Всё остальное (архитектура моделей, пайплайны, multi‑GPU, streaming) уже инкапсулировано в ARACHNE-X и не требует специальных шагов при деплое — только корректный `checkpoint_dir` и достаточный GPU‑ресурс.

