# Датасеты (сырые выгрузки)

Каталог **`raw/`** в git не хранится (см. `.gitignore`); после прогона скрипта появятся подпапки:

| Путь | Источник |
|------|----------|
| `raw/tiger200k_preview/` | [tinytigerpan/tiger200k_preview](https://huggingface.co/datasets/tinytigerpan/tiger200k_preview) |
| `raw/hdvila_100m/subset_*_rows/` | выборка из [TempoFunk/hdvila-100M](https://huggingface.co/datasets/TempoFunk/hdvila-100M) |
| `raw/hq_openhumanvid/subset_*_rows/` | выборка из [Owen777/HQ-OpenHumanVid](https://huggingface.co/datasets/Owen777/HQ-OpenHumanVid) |

Запуск: `bash scripts/fetch_datasets.sh` или `pip install -r requirements-datasets.txt && python scripts/fetch_hf_datasets.py --all`.

Только OpenHumanVid (например после Tiger): `python scripts/fetch_hf_datasets.py --openhumanvid --openhumanvid-max-rows 200000`

Для Tiger: залогиньтесь на Hugging Face, примите условия датасета, задайте **`HF_TOKEN`**.
