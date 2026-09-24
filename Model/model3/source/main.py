"""Crunch entry point for train-from-scratch TSAUC ensemble.

No pretrained model is bundled. ``train`` fits both selected feature families
using only Crunch's labeled training iterator; ``infer`` loads the resulting
relative-path ensemble from the supplied resources directory.
"""

from __future__ import annotations

from typing import Iterable

from ensemble_training import train_ensemble
from solution import infer as _infer


INFER_PARALLELISM = 4


def train(datasets: Iterable[tuple], model_directory_path: str) -> None:
    """Fit core and candidate models and save their configured logit ensemble."""

    train_ensemble(datasets, model_directory_path)


def infer(datasets: Iterable[tuple], model_directory_path: str):
    """Yield readiness, then one ensemble probability per streamed value."""

    yield from _infer(datasets, model_directory_path)
