#!/usr/bin/env python3
"""Smoke-test Submission 6's exact train/infer contract locally."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from competition_data import TrainingData
from solution import infer, train


BUNDLE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = BUNDLE_ROOT.parents[2]


class OnePass:
    """Online iterator that raises if inference tries to consume it twice."""

    def __init__(self, values: np.ndarray) -> None:
        self._values = iter(np.asarray(values, dtype=np.float64))
        self._started = False

    def __iter__(self):
        if self._started:
            raise RuntimeError("online stream was iterated more than once")
        self._started = True
        return self

    def __next__(self) -> float:
        return float(next(self._values))


def _run_once(
    historical: np.ndarray,
    online: np.ndarray,
    model_directory: Path,
) -> np.ndarray:
    generator = infer(
        [(historical, OnePass(online))],
        str(model_directory),
    )
    if next(generator) is not None:
        raise AssertionError("first inference yield must be the ready sentinel")
    prediction = np.asarray(list(generator), dtype=np.float64)
    if prediction.shape != online.shape:
        raise AssertionError("inference output count differs from online input count")
    if not np.all(np.isfinite(prediction)):
        raise AssertionError("inference emitted a non-finite probability")
    if not np.all((prediction >= 0.0) & (prediction <= 1.0)):
        raise AssertionError("inference emitted a value outside [0, 1]")
    return prediction


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-directory",
        type=Path,
        default=BUNDLE_ROOT / "resources",
    )
    parser.add_argument(
        "--train-directory",
        type=Path,
        default=PROJECT_ROOT / "train",
    )
    parser.add_argument("--series-position", type=int, default=0)
    parser.add_argument("--steps", type=int, default=5)
    args = parser.parse_args()

    # Crunch may call train() before infer(), even for a pretrained bundle.
    train((), str(args.model_directory))
    record = TrainingData(args.train_directory).record(args.series_position)
    online = record.online[: args.steps]
    first = _run_once(record.historical, online, args.model_directory)
    second = _run_once(record.historical, online, args.model_directory)
    maximum_error = float(np.max(np.abs(first - second), initial=0.0))
    if maximum_error > 1e-12:
        raise AssertionError(f"inference is not deterministic: {maximum_error:g}")
    print(
        f"PASS: {online.size} one-pass scores; deterministic max error "
        f"{maximum_error:.3g}; range [{first.min():.6f}, {first.max():.6f}]"
    )


if __name__ == "__main__":
    main()
