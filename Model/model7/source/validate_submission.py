"""Local smoke test for the exact cloud streaming contract.

This is intentionally separate from statistical CV.  It checks readiness,
one-pass consumption, output cardinality/range, and deterministic replay for
the actual saved model bundle used by ``solution.infer``.
"""

from __future__ import annotations

import argparse

import numpy as np

from competition_data import TrainingData
from solution import infer


class OnePass:
    """Iterator that fails if submission code asks to iterate it twice."""

    def __init__(self, values: np.ndarray) -> None:
        self.values = iter(np.asarray(values, dtype=np.float64))
        self.started = False

    def __iter__(self):
        if self.started:
            raise RuntimeError("online stream was iterated more than once")
        self.started = True
        return self

    def __next__(self) -> float:
        return float(next(self.values))


def run_once(
    historical: np.ndarray,
    online: np.ndarray,
    model_directory: str,
) -> np.ndarray:
    generator = infer(
        [(historical, OnePass(online))],
        model_directory,
    )
    if next(generator) is not None:
        raise AssertionError("first inference yield must be the readiness sentinel")
    output = np.asarray(list(generator), dtype=np.float64)
    if output.shape != online.shape:
        raise AssertionError(f"received {output.size} scores for {online.size} points")
    if not np.all(np.isfinite(output)) or not np.all((0 <= output) & (output <= 1)):
        raise AssertionError("submission produced an invalid probability")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-directory", default="models/artifacts/model_ensemble"
    )
    parser.add_argument("--train-directory", default="train")
    parser.add_argument("--series-position", type=int, default=0)
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()

    record = TrainingData(args.train_directory).record(args.series_position)
    online = record.online[: args.steps]
    first = run_once(record.historical, online, args.model_directory)
    second = run_once(record.historical, online, args.model_directory)
    maximum_error = float(np.max(np.abs(first - second), initial=0.0))
    if maximum_error > 1e-12:
        raise AssertionError(f"inference is not deterministic: {maximum_error:g}")
    print(
        f"PASS: {online.size} one-pass scores, deterministic max error "
        f"{maximum_error:.3g}, range [{first.min():.6f}, {first.max():.6f}]"
    )


if __name__ == "__main__":
    main()
