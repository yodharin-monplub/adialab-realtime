"""Crunch submission entry point for the pretrained TSAUC ensemble.

The implementation lives in ``solution.py`` so it can be tested independently
of Crunch's filename convention.  These thin wrappers deliberately keep the
default ``main.py`` interface expected by ``crunch test`` and ``crunch push``.
"""

from __future__ import annotations

from typing import Iterable

from solution import infer as _infer
from solution import train as _train


# Streaming inference is CPU-bound while each LightGBM call is single-threaded.
# Crunch runs these workers as separate processes over disjoint series.
INFER_PARALLELISM = 4


def train(datasets: Iterable[tuple], model_directory_path: str) -> None:
    """Preserve and validate the uploaded ensemble, or train the fallback."""

    _train(datasets, model_directory_path)


def infer(datasets: Iterable[tuple], model_directory_path: str):
    """Yield Crunch's readiness sentinel and one score per streamed value."""

    yield from _infer(datasets, model_directory_path)
