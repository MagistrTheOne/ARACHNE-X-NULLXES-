import os
import json
import binascii
import imageio
import subprocess
import numpy as np
import os.path as osp
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torchvision

from einops import rearrange

from ..context_parallel import context_parallel_util


def linear_interpolation(features, seq_len):
    features = features.transpose(1, 2)
    output_features = F.interpolate(features, size=seq_len, align_corners=True, mode='linear')
    return output_features.transpose(1, 2)


@torch.compile
def calculate_x_ref_attn_map(noise_q, ref_k, ref_target_masks, attn_bias=None):
    ref_k = ref_k.to(device=noise_q.device, dtype=noise_q.dtype)
    scale = 1.0 / noise_q.shape[-1] ** 0.5
    noise_q = noise_q * scale
    noise_q = noise_q.transpose(1, 2)
    ref_k = ref_k.transpose(1, 2)
    attn = noise_q @ ref_k.transpose(-2, -1)

    if attn_bias is not None:
        attn = attn + attn_bias

    attn_probs = attn.softmax(-1)

    masks = ref_target_masks.to(device=noise_q.device, dtype=noise_q.dtype)
    denom = masks.sum(-1).clamp_min(1e-6)
    weighted = attn_probs @ masks.transpose(0, 1)
    weighted = weighted / denom.view(1, 1, 1, -1)
    # [B, q, M] -> [M, B, q] -> [M * B, q] (legacy concat dim=0 over mask loop)
    q_len = weighted.shape[2]
    return weighted.mean(1).permute(2, 0, 1).reshape(-1, q_len)


def get_attn_map_with_target(noise_q, key, shape, ref_target_masks=None, split_num=1, cp_split_hw=None):
    N_t, N_h, N_w = shape
    x_seqlens = N_h * N_w
    cp_split_hw = (1, 1) if cp_split_hw is None else tuple(cp_split_hw)
    if cp_split_hw[0] * cp_split_hw[1] > 1:
        (split_h, split_w) = cp_split_hw

        assert N_h % split_h == 0 and N_w % split_w == 0

        N_h_ = N_h // split_h
        N_w_ = N_w // split_w
        x_seqlens = N_h_ * N_w_

    ref_k = key[:, :x_seqlens]
    noise_q = noise_q.contiguous()

    if cp_split_hw[0] * cp_split_hw[1] > 1:
        _, _, H, _ = ref_k.shape
        ref_k = ref_k.permute(0, 2, 1, 3)
        ref_k = rearrange(ref_k, "b h m k -> b (h m) k")
        ref_k = context_parallel_util.gather_cp_2d(ref_k, shape=(H, N_h, N_w), split_hw=cp_split_hw)
        ref_k = rearrange(ref_k, "b (h m) k -> b h m k", h=H)
        ref_k = ref_k.permute(0, 2, 1, 3)

    _, seq_lens, _, _ = noise_q.shape

    if split_num <= 1:
        return calculate_x_ref_attn_map(noise_q, ref_k, ref_target_masks)

    chunk_len = max(1, (seq_lens + split_num - 1) // split_num)
    chunks = []
    for start in range(0, seq_lens, chunk_len):
        q_chunk = noise_q[:, start:start + chunk_len, :, :]
        chunks.append(calculate_x_ref_attn_map(q_chunk, ref_k, ref_target_masks))
    return torch.cat(chunks, dim=-1)


def rand_name(length=8, suffix=''):
    name = binascii.b2a_hex(os.urandom(length)).decode('utf-8')
    if suffix:
        if not suffix.startswith('.'):
            suffix = '.' + suffix
        name += suffix
    return name


def cache_video(tensor,
                save_file=None,
                fps=30,
                suffix='.mp4',
                nrow=8,
                normalize=True,
                value_range=(-1, 1)):
    cache_file = osp.join('/tmp', rand_name(
        suffix=suffix)) if save_file is None else save_file

    tensor = tensor.clamp(min(value_range), max(value_range))
    tensor = torch.stack([
            torchvision.utils.make_grid(
                u, nrow=nrow, normalize=normalize, value_range=value_range)
            for u in tensor.unbind(2)
        ],
                             dim=1).permute(1, 2, 3, 0)
    tensor = (tensor * 255).type(torch.uint8).cpu()

    writer = imageio.get_writer(cache_file, fps=fps, codec='libx264', quality=10, ffmpeg_params=["-crf", "10"])
    for frame in tensor.numpy():
        writer.append_data(frame)
    writer.close()
    return cache_file


def get_audio_duration(audio_path):
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_entries", "format=duration",
        audio_path,
    ]
    out = subprocess.check_output(cmd)
    info = json.loads(out)
    return float(info["format"]["duration"])


def _ffmpeg_subprocess_kwargs(*, quiet: bool) -> dict:
    if quiet:
        return {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
    return {}


def _frame_to_uint8(frame: np.ndarray) -> np.ndarray:
    frame = np.asarray(frame)
    if frame.dtype == np.uint8:
        return frame
    if frame.max() <= 1.0:
        return (np.clip(frame, 0.0, 1.0) * 255.0).astype(np.uint8)
    return np.clip(frame, 0.0, 255.0).astype(np.uint8)


def _encode_rgb_frames_pipe(
    frames_uint8: np.ndarray,
    save_path: str,
    fps: int,
    *,
    crf: int,
    preset: str,
    quiet: bool,
) -> None:
    if frames_uint8.shape[0] == 0:
        raise ValueError("no frames to encode")

    height, width = int(frames_uint8.shape[1]), int(frames_uint8.shape[2])
    cmd = [
        "ffmpeg",
        "-y",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "pipe:0",
        "-an",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", str(crf),
        "-preset", preset,
        save_path,
    ]
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        **_ffmpeg_subprocess_kwargs(quiet=quiet),
    )
    frame_iter = frames_uint8 if quiet else tqdm(frames_uint8, desc="Saving video")
    try:
        assert proc.stdin is not None
        for frame in frame_iter:
            proc.stdin.write(_frame_to_uint8(frame).tobytes())
    finally:
        if proc.stdin is not None:
            proc.stdin.close()
    rc = proc.wait()
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)


def _imageio_quality_to_crf(quality: int) -> int:
    quality = int(np.clip(quality, 1, 10))
    return int(round(32 - (quality - 1) * 2.2))


def save_video_ffmpeg(
    gen_video_samples,
    save_path,
    audio_path=None,
    fps=25,
    quality=5,
    high_quality_save=False,
    export_crf: int | None = None,
    *,
    quiet: bool = False,
):
    output_base, output_ext = os.path.splitext(save_path)
    output_base = output_base if output_ext.lower() == ".mp4" else save_path
    final_output_path = output_base + ".mp4"
    save_path_tmp = output_base + "-temp.mp4"

    output_dir = os.path.dirname(os.path.abspath(final_output_path))
    os.makedirs(output_dir, exist_ok=True)

    if isinstance(gen_video_samples, torch.Tensor):
        video_audio = gen_video_samples.detach().cpu().numpy()
    else:
        video_audio = np.asarray(gen_video_samples)

    if high_quality_save:
        encode_crf, encode_preset = 0, "veryslow"
    elif export_crf is not None:
        encode_crf, encode_preset = int(export_crf), "slow"
    else:
        encode_crf, encode_preset = _imageio_quality_to_crf(quality), "medium"

    _encode_rgb_frames_pipe(
        video_audio,
        save_path_tmp,
        fps=int(fps),
        crf=encode_crf,
        preset=encode_preset,
        quiet=quiet,
    )

    if audio_path is None:
        os.replace(save_path_tmp, final_output_path)
        return
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"audio_path not found: {audio_path}")

    T = int(video_audio.shape[0])
    duration = T / fps
    save_path_crop_audio = output_base + "-cropaudio.wav"
    save_path_crop_tmp = output_base + "-cropvideo.mp4"

    ffmpeg_kwargs = _ffmpeg_subprocess_kwargs(quiet=quiet)
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                audio_path,
                "-t",
                f"{duration}",
                save_path_crop_audio,
            ],
            check=True,
            **ffmpeg_kwargs,
        )

        crop_audio_duration = get_audio_duration(save_path_crop_audio)
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i", save_path_tmp,
                "-t", f"{crop_audio_duration}",
                "-c:v", "copy",
                "-c:a", "copy",
                save_path_crop_tmp,
            ],
            check=True,
            **ffmpeg_kwargs,
        )

        mux_cmd = [
            "ffmpeg",
            "-y",
            "-i", save_path_crop_tmp,
            "-i", save_path_crop_audio,
            "-c:a", "aac",
            "-shortest",
        ]
        if high_quality_save:
            mux_cmd.extend([
                "-c:v", "libx264",
                "-crf", "0",
                "-preset", "veryslow",
            ])
        else:
            mux_cmd.extend(["-c:v", "copy"])
        mux_cmd.append(final_output_path)
        subprocess.run(mux_cmd, check=True, **ffmpeg_kwargs)
    finally:
        for tmp_path in (save_path_tmp, save_path_crop_tmp, save_path_crop_audio):
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
