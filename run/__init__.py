__all__ = ["PipelineManager"]


def __getattr__(name: str):
    if name == "PipelineManager":
        from .run import PipelineManager

        return PipelineManager
    raise AttributeError(f"module 'run' has no attribute {name!r}")
