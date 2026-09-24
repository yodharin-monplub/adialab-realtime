"""Crunch entry points for the timeout-safe frozen Model 6 ensemble."""

from __future__ import annotations

from typing import Iterable

try:  # Package import during repository validation.
    from .solution import infer as _infer
    from .solution import train as _train
except ImportError:  # Direct import in Crunch's /context/code directory.
    from solution import infer as _infer
    from solution import train as _train


# The current Crunch runner partitions complete series across workers. The
# six-component profile needs about 10.5 ms per point in the direct benchmark.
# Four workers completed Crunch's 50,983-point reduced test in 4m35s while
# retaining memory headroom; Crunch caps this at the available CPU count.
INFER_PARALLELISM = 4


def train(datasets: Iterable[tuple], model_directory_path: str) -> None:
    """Validate and preserve the uploaded train-derived frozen model."""

    _train(datasets, model_directory_path)


def infer(datasets: Iterable[tuple], model_directory_path: str):
    """Yield Crunch's ready signal followed by causal online probabilities."""

    yield from _infer(datasets, model_directory_path)


__all__ = ["INFER_PARALLELISM", "infer", "train"]
