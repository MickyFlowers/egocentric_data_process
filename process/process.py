from __future__ import annotations

from .core import BaseProcess, Pipeline, build_pipeline, register_process

# Import modules for side effects so process classes register themselves.
from . import basic_processes as _basic_processes
from . import inverse_kinematics_process as _inverse_kinematics_process
from . import load_data_process as _load_data_process
from . import retarget_process as _retarget_process
from . import visualize_process as _visualize_process
from . import write_data_process as _write_data_process

__all__ = [
    "BaseProcess",
    "Pipeline",
    "build_pipeline",
    "register_process",
]
