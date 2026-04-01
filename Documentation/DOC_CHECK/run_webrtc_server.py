#!/usr/bin/env python3
import runpy
from pathlib import Path


if __name__ == "__main__":
    # Root-level convenience entrypoint expected by deployment docs.
    root = Path(__file__).resolve().parent
    runpy.run_path(str(root / "scripts" / "run_webrtc_server.py"), run_name="__main__")

