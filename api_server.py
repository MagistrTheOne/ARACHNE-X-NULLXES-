from pathlib import Path
import os
import subprocess
import uuid

import torch
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from arachne_x.loader import load_avatar_pipeline


app = FastAPI(title="ARACHNE-X API", version="1.0.0")

BASE_DIR = Path("/workspace/ARACHNE-X")
CHECKPOINT_DIR = Path("/workspace/weights/ARACHNE-X-Avatar")
INPUT_DIR = Path("/workspace/api_inputs")
OUTPUT_DIR = Path("/workspace/api_outputs")
INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Runtime workaround for current PyTorch + avatar path on H200
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

_PIPE = None


def _ensure_pipeline_loaded() -> None:
    global _PIPE
    if _PIPE is not None:
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    _PIPE = load_avatar_pipeline(
        str(CHECKPOINT_DIR),
        variant="single",
        device=device,
        torch_dtype=dtype,
    )


app.mount("/files", StaticFiles(directory="/workspace"), name="files")


@app.get("/health")
def health() -> dict:
    return {
        "ok": True,
        "cuda": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "checkpoint_dir": str(CHECKPOINT_DIR),
    }


@app.post("/avatar/ai2v")
async def avatar_ai2v(
    prompt: str = Form("A realistic close-up talking avatar, stable face, natural lip sync"),
    num_frames: int = Form(93),
    num_inference_steps: int = Form(12),
    resolution: str = Form("480p"),
    image: UploadFile = File(...),
    audio: UploadFile = File(...),
) -> JSONResponse | dict:
    _ensure_pipeline_loaded()

    request_id = str(uuid.uuid4())[:8]
    image_path = INPUT_DIR / f"{request_id}_{image.filename}"
    audio_path = INPUT_DIR / f"{request_id}_{audio.filename}"
    output_path = OUTPUT_DIR / f"{request_id}_out.mp4"

    image_path.write_bytes(await image.read())
    audio_path.write_bytes(await audio.read())

    env = os.environ.copy()
    env["PYTHONPATH"] = str(BASE_DIR)
    env["TORCHDYNAMO_DISABLE"] = "1"

    cmd = [
        "python",
        "scripts/infer.py",
        "--mode",
        "ai2v",
        "--checkpoint_dir",
        str(CHECKPOINT_DIR),
        "--image",
        str(image_path),
        "--audio",
        str(audio_path),
        "--prompt",
        prompt,
        "--num_frames",
        str(num_frames),
        "--num_inference_steps",
        str(num_inference_steps),
        "--resolution",
        resolution,
        "--output",
        str(output_path),
    ]

    proc = subprocess.run(
        cmd,
        cwd=str(BASE_DIR),
        env=env,
        capture_output=True,
        text=True,
    )

    if proc.returncode != 0:
        return JSONResponse(
            status_code=500,
            content={
                "ok": False,
                "error": proc.stderr[-4000:],
                "stdout_tail": proc.stdout[-1000:],
            },
        )

    return {
        "ok": True,
        "request_id": request_id,
        "video_url": f"/files/api_outputs/{output_path.name}",
    }

