#!/usr/bin/env python3
"""
Один прогон на поде: выгрузка датасетов в data/datasets/raw/.

  tinytigerpan/tiger200k_preview   — snapshot (CSV). HF_TOKEN если gated.
  TempoFunk/hdvila-100M            — потоковая выборка N строк → Parquet (или --hdvila-full).
  Owen777/HQ-OpenHumanVid          — то же: subset по умолчанию (или --openhumanvid-full).

Примеры:

  pip install -r requirements-datasets.txt
  export HF_TOKEN=hf_...   # для tiger200k после accept на HF

  python scripts/fetch_hf_datasets.py --all
  python scripts/fetch_hf_datasets.py --openhumanvid --openhumanvid-max-rows 200000
  python scripts/fetch_hf_datasets.py --hdvila-full   # осторожно: огромный объём
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_out(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    return root


def fetch_tiger_preview(out_root: Path) -> Path:
    from huggingface_hub import snapshot_download

    dest = _ensure_out(out_root / "tiger200k_preview")
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    snapshot_download(
        repo_id="tinytigerpan/tiger200k_preview",
        repo_type="dataset",
        local_dir=str(dest),
        token=token,
    )
    meta = {
        "repo_id": "tinytigerpan/tiger200k_preview",
        "local_dir": str(dest.resolve()),
        "note": "Non-commercial license; accept terms on Hugging Face before download.",
    }
    (dest / "NULLXES_FETCH_META.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return dest


def _fetch_streaming_subset(
    out_root: Path,
    *,
    repo_id: str,
    dest_folder: str,
    max_rows: int,
    chunk_rows: int,
    split: str = "train",
    meta_extra: dict | None = None,
    trust_remote_code: bool = False,
) -> Path:
    from datasets import load_dataset
    import pandas as pd

    dest = _ensure_out(out_root / dest_folder)
    subset_dir = dest / f"subset_{max_rows}_rows"
    subset_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(
        repo_id, split=split, streaming=True, trust_remote_code=trust_remote_code
    )
    buffer: list[dict] = []
    total = 0
    part = 0
    for row in ds:
        buffer.append(dict(row))
        total += 1
        if len(buffer) >= chunk_rows:
            path = subset_dir / f"part-{part:05d}.parquet"
            pd.DataFrame(buffer).to_parquet(path, index=False)
            buffer.clear()
            part += 1
        if total >= max_rows:
            break
    if buffer:
        path = subset_dir / f"part-{part:05d}.parquet"
        pd.DataFrame(buffer).to_parquet(path, index=False)

    meta = {
        "repo_id": repo_id,
        "mode": "streaming_subset",
        "max_rows": max_rows,
        "rows_written": total,
        "split": split,
        "output_dir": str(subset_dir.resolve()),
    }
    if meta_extra:
        meta.update(meta_extra)
    (dest / "NULLXES_FETCH_META.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return subset_dir


def fetch_hdvila_streaming(
    out_root: Path,
    *,
    max_rows: int,
    chunk_rows: int,
) -> Path:
    return _fetch_streaming_subset(
        out_root,
        repo_id="TempoFunk/hdvila-100M",
        dest_folder="hdvila_100m",
        max_rows=max_rows,
        chunk_rows=chunk_rows,
        meta_extra={
            "license_note": "Dataset card: AGPL-3.0 — verify compliance for your use case.",
        },
    )


def fetch_openhumanvid_streaming(
    out_root: Path,
    *,
    max_rows: int,
    chunk_rows: int,
) -> Path:
    return _fetch_streaming_subset(
        out_root,
        repo_id="Owen777/HQ-OpenHumanVid",
        dest_folder="hq_openhumanvid",
        max_rows=max_rows,
        chunk_rows=chunk_rows,
        trust_remote_code=True,
        meta_extra={
            "paper": "https://huggingface.co/papers/2412.00115",
            "license_note": "Read dataset card on Hugging Face; human-centric video + captions.",
        },
    )


def fetch_hdvila_full_snapshot(out_root: Path) -> Path:
    from huggingface_hub import snapshot_download

    dest = _ensure_out(out_root / "hdvila_100m_full_snapshot")
    snapshot_download(
        repo_id="TempoFunk/hdvila-100M",
        repo_type="dataset",
        local_dir=str(dest),
    )
    meta = {
        "repo_id": "TempoFunk/hdvila-100M",
        "mode": "full_snapshot",
        "local_dir": str(dest.resolve()),
        "license_note": "AGPL-3.0 — expect very large disk usage.",
    }
    (dest / "NULLXES_FETCH_META.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return dest


def fetch_openhumanvid_full_snapshot(out_root: Path) -> Path:
    from huggingface_hub import snapshot_download

    dest = _ensure_out(out_root / "hq_openhumanvid_full_snapshot")
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    snapshot_download(
        repo_id="Owen777/HQ-OpenHumanVid",
        repo_type="dataset",
        local_dir=str(dest),
        token=token,
    )
    meta = {
        "repo_id": "Owen777/HQ-OpenHumanVid",
        "mode": "full_snapshot",
        "local_dir": str(dest.resolve()),
        "license_note": "See HF dataset card; use subset mode if disk is limited.",
    }
    (dest / "NULLXES_FETCH_META.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch HF datasets into data/datasets/raw/")
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output root (default: <repo>/data/datasets/raw)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Tiger + HD-VILA subset + HQ-OpenHumanVid subset (defaults below)",
    )
    parser.add_argument("--tiger", action="store_true", help="tinytigerpan/tiger200k_preview")
    parser.add_argument("--hdvila", action="store_true", help="TempoFunk/hdvila-100M (subset or full)")
    parser.add_argument(
        "--openhumanvid",
        action="store_true",
        help="Owen777/HQ-OpenHumanVid (subset or full)",
    )
    parser.add_argument(
        "--hdvila-max-rows",
        type=int,
        default=100_000,
        help="HD-VILA streaming subset row cap (ignored with --hdvila-full)",
    )
    parser.add_argument(
        "--openhumanvid-max-rows",
        type=int,
        default=100_000,
        help="HQ-OpenHumanVid streaming subset row cap (ignored with --openhumanvid-full)",
    )
    parser.add_argument(
        "--hdvila-chunk-rows",
        type=int,
        default=10_000,
        help="Rows per Parquet part (HD-VILA and OpenHumanVid subsets)",
    )
    parser.add_argument(
        "--openhumanvid-chunk-rows",
        type=int,
        default=None,
        help="Override chunk size for OpenHumanVid only (default: same as --hdvila-chunk-rows)",
    )
    parser.add_argument(
        "--hdvila-full",
        action="store_true",
        help="Download entire HD-VILA dataset repo snapshot (huge)",
    )
    parser.add_argument(
        "--openhumanvid-full",
        action="store_true",
        help="Download entire HQ-OpenHumanVid repo snapshot",
    )
    args = parser.parse_args()

    repo = _repo_root()
    sys.path.insert(0, str(repo))

    out = Path(args.out) if args.out else repo / "data" / "datasets" / "raw"
    out = out.resolve()

    do_tiger = args.tiger or args.all
    do_hdvila = args.hdvila or args.all
    do_ohv = args.openhumanvid or args.all

    if not do_tiger and not do_hdvila and not do_ohv:
        parser.error("Specify --all and/or --tiger, --hdvila, --openhumanvid")

    chunk_h = max(1000, args.hdvila_chunk_rows)
    chunk_ohv = max(1000, args.openhumanvid_chunk_rows or args.hdvila_chunk_rows)

    if do_tiger:
        print(f"[tiger200k_preview] -> {out / 'tiger200k_preview'}")
        fetch_tiger_preview(out)
        print("[tiger200k_preview] done.")

    if do_hdvila:
        if args.hdvila_full:
            print(f"[hdvila-100M] FULL snapshot -> {out / 'hdvila_100m_full_snapshot'}")
            fetch_hdvila_full_snapshot(out)
            print("[hdvila-100M] full snapshot done.")
        else:
            n = max(1, args.hdvila_max_rows)
            print(f"[hdvila-100M] streaming first {n} rows -> {out / 'hdvila_100m'}")
            fetch_hdvila_streaming(out, max_rows=n, chunk_rows=chunk_h)
            print("[hdvila-100M] subset done.")

    if do_ohv:
        if args.openhumanvid_full:
            print(f"[HQ-OpenHumanVid] FULL snapshot -> {out / 'hq_openhumanvid_full_snapshot'}")
            fetch_openhumanvid_full_snapshot(out)
            print("[HQ-OpenHumanVid] full snapshot done.")
        else:
            n = max(1, args.openhumanvid_max_rows)
            print(f"[HQ-OpenHumanVid] streaming first {n} rows -> {out / 'hq_openhumanvid'}")
            fetch_openhumanvid_streaming(out, max_rows=n, chunk_rows=chunk_ohv)
            print("[HQ-OpenHumanVid] subset done.")

    print(f"Output root: {out}")


if __name__ == "__main__":
    main()
