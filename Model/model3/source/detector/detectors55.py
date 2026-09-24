"""Empirical conformal power-martingale evidence on detector innovations.

Each detectors3 aspect is first divided by its own history-measured null
standard deviation.  Its current innovation is then converted to a fixed
historical empirical rank.  Resettable power-martingales accumulate repeated
two-sided or directional tail surprises, providing a nonlinear persistence
geometry that is distinct from raw CUSUM sums and EWMA filters.

The historical reference is immutable, so both paths are causal.  The
offline implementation maps complete arrays but applies the recurrence one
step at a time; the streaming implementation reads only the current
detectors3 increment.  Both are checked below on correlated and broken data.
"""

from __future__ import annotations

import time

import numpy as np

import detectors3 as d3
import detectors15 as d15


EPSILONS = np.asarray((0.25, 0.5, 0.75), dtype=np.float64)
DIR_EPSILON = 0.5
CLIP_INNOVATION = 12.0
CLIP_STATE = 60.0
N_ASPECTS = len(d3.ASPECTS)
N_BASE_PER_ASPECT = len(EPSILONS) + 2  # two-sided + dir max + dir signed
N_SUMMARY_PER_EPSILON = 4              # max/top3/top6/mean
N_DIRECTIONAL_SUMMARIES = 5
N_SUMMARIES = len(EPSILONS) * N_SUMMARY_PER_EPSILON + N_DIRECTIONAL_SUMMARIES
N_EXTRA = N_ASPECTS * N_BASE_PER_ASPECT + N_SUMMARIES


def feature_names():
    names = []
    for aspect in d3.ASPECTS:
        for epsilon in EPSILONS:
            names.append(f"pm_{aspect}_e{epsilon:g}")
        names.extend((f"pm_{aspect}_dirmax", f"pm_{aspect}_dirsigned"))
    for epsilon in EPSILONS:
        tag = f"e{epsilon:g}"
        names.extend((f"pm_{tag}_max", f"pm_{tag}_top3",
                      f"pm_{tag}_top6", f"pm_{tag}_mean"))
    names.extend(("pm_dir_max", "pm_dir_top3", "pm_dir_top6",
                  "pm_dir_mean", "pm_dir_imbalance"))
    assert len(names) == N_EXTRA
    return names


def _empirical_probability(reference, values):
    values = np.round(np.asarray(values, dtype=np.float64), 8)
    left = np.searchsorted(reference, values, side="left")
    right = np.searchsorted(reference, values, side="right")
    return (left + right + 1.0) / (2.0 * (reference.size + 1.0))


def _scalar_probability(reference, value):
    rounded = round(float(value), 8)
    left = reference.searchsorted(rounded, side="left")
    right = reference.searchsorted(rounded, side="right")
    return float((left + right + 1.0) /
                 (2.0 * (reference.size + 1.0)))


def _power_log_increment(probability, epsilon):
    p = np.clip(np.asarray(probability, dtype=np.float64), 1e-12, 1.0)
    return np.log(float(epsilon)) + (float(epsilon) - 1.0) * np.log(p)


def _summaries(two_state, up_state, down_state):
    output = []
    for column in range(len(EPSILONS)):
        values = two_state[:, column]
        ordered = np.sort(values)
        output.extend((
            float(ordered[-1]),
            float(ordered[-3:].mean()),
            float(ordered[-6:].mean()),
            float(values.mean()),
        ))
    directional = np.maximum(up_state, down_state)
    ordered = np.sort(directional)
    output.extend((
        float(ordered[-1]),
        float(ordered[-3:].mean()),
        float(ordered[-6:].mean()),
        float(directional.mean()),
        float(np.max(up_state) - np.max(down_state)),
    ))
    return np.asarray(output, dtype=np.float64)


class EmpiricalInnovationNull:
    def __init__(self, x_hist):
        hs, hist_inc, _ = d15._hist_and_online_increments(x_hist)
        self.hs = hs
        self.reference = []
        for aspect in d3.ASPECTS:
            values = np.clip(
                np.asarray(hist_inc[aspect], dtype=np.float64) /
                hs.s[aspect],
                -CLIP_INNOVATION, CLIP_INNOVATION,
            )
            self.reference.append(np.sort(np.round(values, 8)))

    def probabilities(self, matrix):
        matrix = np.asarray(matrix, dtype=np.float64)
        output = np.empty_like(matrix)
        for aspect_index, reference in enumerate(self.reference):
            output[:, aspect_index] = _empirical_probability(
                reference, matrix[:, aspect_index])
        return output

    def probability(self, aspect_index, value):
        return _scalar_probability(self.reference[int(aspect_index)], value)


def _features_from_probabilities(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    steps = probabilities.shape[0]
    output = np.empty((steps, N_EXTRA), dtype=np.float64)
    two_state = np.zeros((N_ASPECTS, len(EPSILONS)), dtype=np.float64)
    up_state = np.zeros(N_ASPECTS, dtype=np.float64)
    down_state = np.zeros(N_ASPECTS, dtype=np.float64)
    for step in range(steps):
        u = probabilities[step]
        two_p = np.clip(2.0 * np.minimum(u, 1.0 - u), 1e-12, 1.0)
        two_increment = np.column_stack([
            _power_log_increment(two_p, epsilon)
            for epsilon in EPSILONS
        ])
        two_state = np.clip(
            np.maximum(0.0, two_state + two_increment),
            0.0, CLIP_STATE,
        )
        up_state = np.clip(
            np.maximum(
                0.0,
                up_state + _power_log_increment(1.0 - u, DIR_EPSILON),
            ),
            0.0, CLIP_STATE,
        )
        down_state = np.clip(
            np.maximum(
                0.0,
                down_state + _power_log_increment(u, DIR_EPSILON),
            ),
            0.0, CLIP_STATE,
        )
        base = np.empty((N_ASPECTS, N_BASE_PER_ASPECT), dtype=np.float64)
        base[:, :len(EPSILONS)] = two_state
        base[:, len(EPSILONS)] = np.maximum(up_state, down_state)
        base[:, len(EPSILONS) + 1] = up_state - down_state
        output[step, :N_ASPECTS * N_BASE_PER_ASPECT] = base.ravel()
        output[step, N_ASPECTS * N_BASE_PER_ASPECT:] = _summaries(
            two_state, up_state, down_state)
    return output


def features_from_channels(x_hist, channels):
    """Research-cache adapter; channels must be detector3 innovations."""
    calibration = EmpiricalInnovationNull(x_hist)
    matrix = np.clip(
        np.asarray(channels, dtype=np.float64)[:, :N_ASPECTS],
        -CLIP_INNOVATION, CLIP_INNOVATION,
    )
    return _features_from_probabilities(calibration.probabilities(matrix))


def extra_features(x_hist, x_online):
    calibration = EmpiricalInnovationNull(x_hist)
    hs, _, increments = d15._hist_and_online_increments(x_hist, x_online)
    matrix = np.column_stack([
        np.clip(
            np.asarray(increments[aspect], dtype=np.float64) / hs.s[aspect],
            -CLIP_INNOVATION, CLIP_INNOVATION,
        )
        for aspect in d3.ASPECTS
    ])
    return _features_from_probabilities(calibration.probabilities(matrix))


class StreamDetector55:
    def __init__(self, x_hist, inner=None):
        self.inner = d3.StreamDetector3(x_hist) if inner is None else inner
        self.owns_inner = inner is None
        self.calibration = EmpiricalInnovationNull(x_hist)
        self.two_state = np.zeros(
            (N_ASPECTS, len(EPSILONS)), dtype=np.float64)
        self.up_state = np.zeros(N_ASPECTS, dtype=np.float64)
        self.down_state = np.zeros(N_ASPECTS, dtype=np.float64)

    def _from_current_inner(self):
        position = (self.inner.t - 1) % self.inner._wmax
        u = np.empty(N_ASPECTS, dtype=np.float64)
        for aspect_index, aspect in enumerate(d3.ASPECTS):
            value = np.clip(
                self.inner._ring[aspect][position] /
                self.inner.hs.s[aspect],
                -CLIP_INNOVATION, CLIP_INNOVATION,
            )
            u[aspect_index] = self.calibration.probability(
                aspect_index, value)
        two_p = np.clip(2.0 * np.minimum(u, 1.0 - u), 1e-12, 1.0)
        increments = np.column_stack([
            _power_log_increment(two_p, epsilon)
            for epsilon in EPSILONS
        ])
        self.two_state = np.clip(
            np.maximum(0.0, self.two_state + increments),
            0.0, CLIP_STATE,
        )
        self.up_state = np.clip(
            np.maximum(
                0.0,
                self.up_state + _power_log_increment(
                    1.0 - u, DIR_EPSILON),
            ),
            0.0, CLIP_STATE,
        )
        self.down_state = np.clip(
            np.maximum(
                0.0,
                self.down_state + _power_log_increment(u, DIR_EPSILON),
            ),
            0.0, CLIP_STATE,
        )
        base = np.empty((N_ASPECTS, N_BASE_PER_ASPECT), dtype=np.float64)
        base[:, :len(EPSILONS)] = self.two_state
        base[:, len(EPSILONS)] = np.maximum(
            self.up_state, self.down_state)
        base[:, len(EPSILONS) + 1] = self.up_state - self.down_state
        output = np.empty(N_EXTRA, dtype=np.float64)
        output[:N_ASPECTS * N_BASE_PER_ASPECT] = base.ravel()
        output[N_ASPECTS * N_BASE_PER_ASPECT:] = _summaries(
            self.two_state, self.up_state, self.down_state)
        return output

    def step(self, value):
        if not self.owns_inner:
            raise RuntimeError("use step_from_inner with a shared detector")
        self.inner.step(value)
        return self._from_current_inner()

    def step_from_inner(self):
        return self._from_current_inner()


if __name__ == "__main__":
    rng = np.random.default_rng(20260813)
    worst = 0.0
    for trial in range(10):
        n_hist = int(rng.integers(1000, 3000))
        steps = int(rng.integers(10, 700))
        ar = float(rng.uniform(-0.85, 0.85))
        noise = rng.standard_t(
            df=float(rng.uniform(3.0, 12.0)), size=n_hist + steps)
        values = np.empty(n_hist + steps, dtype=np.float64)
        values[0] = noise[0]
        for index in range(1, values.size):
            values[index] = ar * values[index - 1] + noise[index]
        history = values[:n_hist].copy()
        online = values[n_hist:].copy()
        if trial % 3 == 0 and steps > 30:
            tau = int(rng.integers(5, steps - 5))
            online[tau:] += float(rng.uniform(-0.8, 0.8))
        elif trial % 3 == 1 and steps > 30:
            tau = int(rng.integers(5, steps - 5))
            online[tau:] *= float(rng.uniform(1.2, 1.8))
        elif trial % 3 == 2 and steps > 30:
            tau = int(rng.integers(5, steps - 5))
            for index in range(tau + 1, steps):
                online[index] += 0.55 * online[index - 1]

        offline = extra_features(history, online)
        detector = StreamDetector55(history)
        streamed = np.vstack([detector.step(value) for value in online])
        shared_inner = d3.StreamDetector3(history)
        shared = StreamDetector55(history, inner=shared_inner)
        shared_rows = []
        for value in online:
            shared_inner.step(value)
            shared_rows.append(shared.step_from_inner())
        shared_streamed = np.vstack(shared_rows)
        error = float(np.max(np.abs(offline - streamed)))
        shared_error = float(np.max(np.abs(offline - shared_streamed)))
        worst = max(worst, error, shared_error)
        assert offline.shape == (steps, N_EXTRA)
        assert error < 1e-9, (trial, error)
        assert shared_error < 1e-9, (trial, shared_error)

    detector = StreamDetector55(rng.normal(size=3000))
    started = time.perf_counter()
    for value in rng.normal(size=1000):
        detector.step(value)
    elapsed = time.perf_counter() - started
    print(f"detectors55 OK: {N_EXTRA} features, worst equality "
          f"{worst:.3e}, {elapsed * 1e6 / 1000:.0f} us/step")
