from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arachne_x.orchestration.job_runner import healthcheck, run_job_file, watch_jobs


def main() -> None:
    parser = argparse.ArgumentParser(description="NULLXES FURIA-EIDOLON RunPod job runner")
    parser.add_argument("--job", type=str, default=None, help="Single JSON job file.")
    parser.add_argument("--jobs-dir", type=str, default=None, help="Directory with pending *.json jobs.")
    parser.add_argument("--once", action="store_true", help="Process current pending jobs once and exit.")
    parser.add_argument("--watch", action="store_true", help="Watch --jobs-dir forever.")
    parser.add_argument("--poll-sec", type=float, default=5.0)
    parser.add_argument("--health", action="store_true", help="Run a lightweight path healthcheck and exit.")
    parser.add_argument("--repo-root", type=str, default=str(ROOT))
    parser.add_argument("--video-python", type=str, default=None)
    parser.add_argument("--stage3-python", type=str, default=None)
    args = parser.parse_args()

    if args.health:
        print(
            json.dumps(
                healthcheck(
                    repo_root=args.repo_root,
                    video_python=args.video_python,
                    stage3_python=args.stage3_python,
                ),
                indent=2,
                ensure_ascii=False,
            )
        )
        return

    if args.job:
        manifest = run_job_file(
            args.job,
            repo_root=args.repo_root,
            video_python=args.video_python,
            stage3_python=args.stage3_python,
        )
        print(json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False))
        return

    if not args.jobs_dir:
        parser.error("--job, --jobs-dir, or --health is required")
    if not (args.once or args.watch):
        parser.error("--once or --watch is required with --jobs-dir")
    watch_jobs(
        jobs_dir=args.jobs_dir,
        repo_root=args.repo_root,
        poll_sec=args.poll_sec,
        once=args.once,
        video_python=args.video_python,
        stage3_python=args.stage3_python,
    )


if __name__ == "__main__":
    main()
