__all__ = [
    "WeightsLayout",
    "load_base_pipeline",
    "load_avatar_pipeline",
    "get_vocal_separator_path",
]


def __getattr__(name):
    if name in __all__:
        from .loader import WeightsLayout, get_vocal_separator_path, load_avatar_pipeline, load_base_pipeline

        values = {
            "WeightsLayout": WeightsLayout,
            "load_base_pipeline": load_base_pipeline,
            "load_avatar_pipeline": load_avatar_pipeline,
            "get_vocal_separator_path": get_vocal_separator_path,
        }
        return values[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
