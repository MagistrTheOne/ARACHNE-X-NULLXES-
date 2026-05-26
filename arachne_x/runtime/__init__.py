"""GTM runtime layer: programmatic inference and future train/service APIs."""

from arachne_x.runtime.avatar_serving import (
    generate_mp4_bytes_from_job,
    get_avatar_pipeline,
    stream_avatar_frames_raw_sync,
)
__all__ = [
    "execute_infer",
    "get_avatar_pipeline",
    "stream_avatar_frames_raw_sync",
    "generate_mp4_bytes_from_job",
]


def execute_infer(*args, **kwargs):
    from arachne_x.runtime.inference_engine import execute_infer as _impl

    return _impl(*args, **kwargs)
