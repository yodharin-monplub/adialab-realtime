"""Small causal score-history corrections for Model 10.

Every transform keeps the current raw logit with coefficient exactly one and
adds only a bounded or convex history correction.  The batch implementation
processes each series chronologically, while :class:`StreamingCausalTransform`
provides the identical one-row-at-a-time contract needed by Crunch inference.

The returned values are finite ranking margins, not calibrated probabilities.
Use :func:`score_to_probability` only when a consumer requires values in
``[0, 1]``; this monotone conversion does not change TSAUC.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import math
from typing import Iterable, Iterator, Literal

import numpy as np


TransformKind = Literal[
    "raw",
    "logit_ema_trend",
    "rolling_max_premium",
    "decayed_peak_premium",
    "prior_ew_innovation",
]


def _number_tag(value: float | int) -> str:
    """Return a stable filename-safe representation of a numeric setting."""

    return format(float(value), ".12g").replace("-", "m").replace(".", "p")


@dataclass(frozen=True)
class TransformSpec:
    """Serializable contract for one causal transform candidate."""

    kind: TransformKind
    correction_weight: float = 0.0
    half_life: float | None = None
    window: int | None = None
    decay_factor: float | None = None
    minimum_scale: float | None = None
    innovation_clip: float | None = None

    @property
    def family(self) -> str:
        """Return the stable family label used by nested-selection reports."""

        self.validate()
        return self.kind

    @property
    def name(self) -> str:
        """Return a deterministic unique name suitable for reports and paths."""

        self.validate()
        if self.kind == "raw":
            return "raw"
        weight = _number_tag(self.correction_weight)
        if self.kind == "logit_ema_trend":
            return f"ema_trend_h{_number_tag(self.half_life)}_w{weight}"
        if self.kind == "rolling_max_premium":
            return f"rolling_max_n{self.window}_w{weight}"
        if self.kind == "decayed_peak_premium":
            return f"decayed_peak_f{_number_tag(self.decay_factor)}_w{weight}"
        return (
            f"prior_innovation_h{_number_tag(self.half_life)}_w{weight}"
            f"_s{_number_tag(self.minimum_scale)}"
            f"_c{_number_tag(self.innovation_clip)}"
        )

    def to_dict(self) -> dict[str, float | int | str | None]:
        """Return a JSON-compatible representation of this frozen contract."""

        self.validate()
        return asdict(self)

    def validate(self) -> None:
        """Reject ambiguous, non-finite, or unsafe transform settings."""

        supported = {
            "raw",
            "logit_ema_trend",
            "rolling_max_premium",
            "decayed_peak_premium",
            "prior_ew_innovation",
        }
        if self.kind not in supported:
            raise ValueError(f"unsupported causal transform kind: {self.kind!r}")
        if not math.isfinite(self.correction_weight):
            raise ValueError("correction_weight must be finite")
        if abs(self.correction_weight) > 1.0:
            raise ValueError("abs(correction_weight) must not exceed one")

        if self.kind == "raw":
            if (
                self.correction_weight != 0.0
                or self.half_life is not None
                or self.window is not None
                or self.decay_factor is not None
                or self.minimum_scale is not None
                or self.innovation_clip is not None
            ):
                raise ValueError("raw transform cannot contain correction settings")
            return

        if self.correction_weight == 0.0:
            raise ValueError("non-raw transforms require a non-zero correction_weight")
        if self.kind == "rolling_max_premium":
            if self.correction_weight < 0.0:
                raise ValueError("rolling maximum premium weight must be positive")
            if not isinstance(self.window, int) or isinstance(self.window, bool):
                raise ValueError("rolling maximum window must be an integer")
            if self.window < 2:
                raise ValueError("rolling maximum window must be at least two")
            if any(
                value is not None
                for value in (
                    self.half_life,
                    self.decay_factor,
                    self.minimum_scale,
                    self.innovation_clip,
                )
            ):
                raise ValueError("rolling maximum accepts only window and weight")
            return

        if self.kind == "decayed_peak_premium":
            if self.correction_weight < 0.0:
                raise ValueError("decayed peak premium weight must be positive")
            if self.decay_factor is None or not math.isfinite(self.decay_factor):
                raise ValueError("decay_factor must be finite")
            if not 0.0 < self.decay_factor <= 1.0:
                raise ValueError("decay_factor must be in (0, 1]")
            if any(
                value is not None
                for value in (
                    self.half_life,
                    self.window,
                    self.minimum_scale,
                    self.innovation_clip,
                )
            ):
                raise ValueError("decayed peak accepts only decay factor and weight")
            return

        if self.window is not None or self.decay_factor is not None:
            raise ValueError(f"{self.kind} does not accept window or decay_factor")
        if self.half_life is None or not math.isfinite(self.half_life):
            raise ValueError("half_life must be finite")
        if self.half_life <= 0.0:
            raise ValueError("half_life must be positive")

        if self.kind == "logit_ema_trend":
            if self.minimum_scale is not None or self.innovation_clip is not None:
                raise ValueError("EMA trend accepts only half_life and weight")
            return

        if self.minimum_scale is None or not math.isfinite(self.minimum_scale):
            raise ValueError("minimum_scale must be finite")
        if self.minimum_scale <= 0.0:
            raise ValueError("minimum_scale must be positive")
        if self.innovation_clip is None or not math.isfinite(self.innovation_clip):
            raise ValueError("innovation_clip must be finite")
        if self.innovation_clip <= 0.0:
            raise ValueError("innovation_clip must be positive")


def safe_logit(
    probability: Iterable[float] | np.ndarray,
    epsilon: float = 1e-7,
) -> np.ndarray:
    """Convert finite probabilities to bounded, finite log odds."""

    values = np.asarray(probability, dtype=np.float64)
    if not math.isfinite(epsilon) or not 0.0 < epsilon < 0.5:
        raise ValueError("epsilon must be finite and strictly between zero and 0.5")
    if not np.all(np.isfinite(values)):
        raise ValueError("probability contains NaN or infinity")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("probability must be in [0, 1]")
    clipped = np.clip(values, epsilon, 1.0 - epsilon)
    return np.log(clipped) - np.log1p(-clipped)


def _safe_logit_scalar(probability: float, epsilon: float = 1e-7) -> float:
    """Allocation-free scalar twin of :func:`safe_logit` for inference."""

    value = float(probability)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("probability must be finite and in [0, 1]")
    clipped = min(max(value, epsilon), 1.0 - epsilon)
    return math.log(clipped) - math.log1p(-clipped)


def score_to_probability(score: Iterable[float] | np.ndarray) -> np.ndarray:
    """Map finite ranking margins monotonically into ``[0, 1]``."""

    values = np.asarray(score, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("score contains NaN or infinity")
    output = np.empty_like(values)
    positive = values >= 0.0
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    output[~positive] = exponential / (1.0 + exponential)
    return output


def validate_causal_order(
    probability: Iterable[float] | np.ndarray,
    series_position: Iterable[int] | np.ndarray,
    time_online: Iterable[int] | np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Validate aligned probabilities and contiguous chronological series.

    Series identifiers must be integer and nondecreasing, which guarantees
    that a series cannot reappear after its streaming state has been reset.
    When supplied, ``time_online`` must be integer and strictly increase within
    each series. Gaps are allowed so the same function can check row subsets.
    """

    values = np.asarray(probability, dtype=np.float64)
    owner = np.asarray(series_position)
    if values.ndim != 1 or owner.ndim != 1 or values.shape != owner.shape:
        raise ValueError("probability and series_position must be same-length 1-D arrays")
    if not np.issubdtype(owner.dtype, np.integer):
        raise ValueError("series_position must contain integers")
    safe_logit(values)
    owner = owner.astype(np.int64, copy=False)
    if np.any(owner < 0):
        raise ValueError("series_position must be non-negative")
    if owner.size and np.any(owner[1:] < owner[:-1]):
        raise ValueError("series_position must be nondecreasing and contiguous")

    if time_online is None:
        return values, owner, None
    online = np.asarray(time_online)
    if online.ndim != 1 or online.shape != values.shape:
        raise ValueError("time_online must be a same-length 1-D array")
    if not np.issubdtype(online.dtype, np.integer):
        raise ValueError("time_online must contain integers")
    online = online.astype(np.int64, copy=False)
    if np.any(online < 0):
        raise ValueError("time_online must be non-negative")
    same_series = owner[1:] == owner[:-1]
    if np.any(online[1:][same_series] <= online[:-1][same_series]):
        raise ValueError("time_online must strictly increase within each series")
    return values, owner, online


def _ema_alpha(half_life: float) -> float:
    """Return the EW update weight corresponding to a half-life in rows."""

    return -math.expm1(math.log(0.5) / half_life)


class StreamingCausalTransform:
    """One-series streaming implementation of a :class:`TransformSpec`."""

    def __init__(self, spec: TransformSpec) -> None:
        spec.validate()
        self.spec = spec
        self._alpha = (
            _ema_alpha(float(spec.half_life))
            if spec.half_life is not None
            else None
        )
        self.reset()

    def reset(self) -> None:
        """Clear all history before processing another series."""

        self._initialized = False
        self._ema = 0.0
        self._variance = 0.0
        self._peak = 0.0
        self._index = -1
        self._last_time: int | None = None
        self._maximum_queue: deque[tuple[int, float]] = deque()

    def step(self, probability: float, time_online: int | None = None) -> float:
        """Consume one revealed probability and return its causal rank margin."""

        margin = _safe_logit_scalar(probability)
        if time_online is not None:
            if not isinstance(time_online, (int, np.integer)):
                raise ValueError("time_online must be an integer")
            current_time = int(time_online)
            if current_time < 0:
                raise ValueError("time_online must be non-negative")
            if self._last_time is not None and current_time <= self._last_time:
                raise ValueError("time_online must strictly increase within a series")
            self._last_time = current_time

        kind = self.spec.kind
        if kind == "raw":
            return margin

        weight = self.spec.correction_weight
        if kind == "logit_ema_trend":
            if not self._initialized:
                self._ema = margin
                self._initialized = True
            else:
                alpha = float(self._alpha)
                self._ema += alpha * (margin - self._ema)
            return margin + weight * (margin - self._ema)

        if kind == "rolling_max_premium":
            self._index += 1
            window = int(self.spec.window)
            expired = self._index - window
            while self._maximum_queue and self._maximum_queue[0][0] <= expired:
                self._maximum_queue.popleft()
            while self._maximum_queue and self._maximum_queue[-1][1] <= margin:
                self._maximum_queue.pop()
            self._maximum_queue.append((self._index, margin))
            maximum = self._maximum_queue[0][1]
            return margin + weight * (maximum - margin)

        if kind == "decayed_peak_premium":
            if not self._initialized:
                self._peak = float(probability)
                self._initialized = True
            else:
                self._peak = max(
                    float(probability),
                    float(self.spec.decay_factor) * self._peak,
                )
            peak_margin = _safe_logit_scalar(self._peak)
            return margin + weight * (peak_margin - margin)

        if not self._initialized:
            self._ema = margin
            self._variance = 0.0
            self._initialized = True
            return margin
        alpha = float(self._alpha)
        delta = margin - self._ema
        scale = max(math.sqrt(max(self._variance, 0.0)), float(self.spec.minimum_scale))
        limit = float(self.spec.innovation_clip)
        innovation = min(max(delta / scale, -limit), limit)
        output = margin + weight * innovation
        self._ema += alpha * delta
        self._variance = (1.0 - alpha) * (
            self._variance + alpha * delta * delta
        )
        return output

    def step_probability(
        self,
        probability: float,
        time_online: int | None = None,
    ) -> float:
        """Consume one row and return the margin mapped into ``[0, 1]``."""

        score = self.step(probability, time_online)
        if score >= 0.0:
            return 1.0 / (1.0 + math.exp(-score))
        exponential = math.exp(score)
        return exponential / (1.0 + exponential)


def apply_causal_transform(
    probability: Iterable[float] | np.ndarray,
    series_position: Iterable[int] | np.ndarray,
    spec: TransformSpec,
    time_online: Iterable[int] | np.ndarray | None = None,
) -> np.ndarray:
    """Apply one transform offline with an exact state reset per series."""

    spec.validate()
    values, owner, online = validate_causal_order(
        probability, series_position, time_online
    )
    output = np.empty(values.size, dtype=np.float64)
    if values.size == 0:
        return output

    transformer = StreamingCausalTransform(spec)
    previous_owner = owner[0]
    for index, (value, current_owner) in enumerate(zip(values, owner, strict=True)):
        if current_owner != previous_owner:
            transformer.reset()
            previous_owner = current_owner
        current_time = None if online is None else int(online[index])
        output[index] = transformer.step(float(value), current_time)
    if not np.all(np.isfinite(output)):
        raise AssertionError("causal transform produced a non-finite score")
    return output


def iter_causal_transform_scores(
    probability: Iterable[float] | np.ndarray,
    series_position: Iterable[int] | np.ndarray,
    specs: Iterable[TransformSpec],
    time_online: Iterable[int] | np.ndarray | None = None,
) -> Iterator[tuple[TransformSpec, np.ndarray]]:
    """Yield candidate score arrays one at a time to bound grid-search memory."""

    values, owner, online = validate_causal_order(
        probability, series_position, time_online
    )
    seen: set[str] = set()
    for spec in specs:
        if spec.name in seen:
            raise ValueError(f"duplicate transform specification: {spec.name}")
        seen.add(spec.name)
        yield spec, apply_causal_transform(values, owner, spec, online)


def _time_slice_layout(
    owner: np.ndarray,
    online: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Build compact row indexes for vectorized chronological state updates."""

    if owner.size == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, empty, empty, 0
    new_series = np.r_[True, owner[1:] != owner[:-1]]
    dense_series = np.cumsum(new_series, dtype=np.int64) - 1
    order = np.argsort(online, kind="stable")
    ordered_time = online[order]
    starts = np.flatnonzero(np.r_[True, ordered_time[1:] != ordered_time[:-1]])
    stops = np.r_[starts[1:], order.size]
    return dense_series, order, starts, stops, int(dense_series[-1]) + 1


def _ema_trend_basis(
    margin: np.ndarray,
    dense_series: np.ndarray,
    order: np.ndarray,
    starts: np.ndarray,
    stops: np.ndarray,
    series_count: int,
    half_life: float,
) -> np.ndarray:
    """Return ``current_logit - current_inclusive_EMA`` for every row."""

    alpha = _ema_alpha(half_life)
    state = np.zeros(series_count, dtype=np.float64)
    initialized = np.zeros(series_count, dtype=bool)
    basis = np.empty(margin.size, dtype=np.float64)
    for left, right in zip(starts, stops, strict=True):
        positions = order[left:right]
        active = dense_series[positions]
        first = ~initialized[active]
        if np.any(first):
            state[active[first]] = margin[positions[first]]
            initialized[active[first]] = True
        later = ~first
        if np.any(later):
            later_series = active[later]
            state[later_series] += alpha * (
                margin[positions[later]] - state[later_series]
            )
        basis[positions] = margin[positions] - state[active]
    return basis


def _prior_innovation_basis(
    margin: np.ndarray,
    dense_series: np.ndarray,
    order: np.ndarray,
    starts: np.ndarray,
    stops: np.ndarray,
    series_count: int,
    half_life: float,
    minimum_scale: float,
    innovation_clip: float,
) -> np.ndarray:
    """Return clipped innovations against the state available at ``t-1``."""

    alpha = _ema_alpha(half_life)
    mean = np.zeros(series_count, dtype=np.float64)
    variance = np.zeros(series_count, dtype=np.float64)
    initialized = np.zeros(series_count, dtype=bool)
    basis = np.empty(margin.size, dtype=np.float64)
    for left, right in zip(starts, stops, strict=True):
        positions = order[left:right]
        active = dense_series[positions]
        first = ~initialized[active]
        if np.any(first):
            first_series = active[first]
            mean[first_series] = margin[positions[first]]
            variance[first_series] = 0.0
            initialized[first_series] = True
            basis[positions[first]] = 0.0
        later = ~first
        if np.any(later):
            later_positions = positions[later]
            later_series = active[later]
            delta = margin[later_positions] - mean[later_series]
            scale = np.maximum(
                np.sqrt(np.maximum(variance[later_series], 0.0)),
                minimum_scale,
            )
            basis[later_positions] = np.clip(
                delta / scale,
                -innovation_clip,
                innovation_clip,
            )
            mean[later_series] += alpha * delta
            variance[later_series] = (1.0 - alpha) * (
                variance[later_series] + alpha * delta * delta
            )
    return basis


def _rolling_max_basis(
    margin: np.ndarray,
    dense_series: np.ndarray,
    order: np.ndarray,
    starts: np.ndarray,
    stops: np.ndarray,
    series_count: int,
    window: int,
) -> np.ndarray:
    """Return finite-window max minus current logit using a per-series ring."""

    ring = np.full((series_count, window), -np.inf, dtype=np.float64)
    count = np.zeros(series_count, dtype=np.int64)
    basis = np.empty(margin.size, dtype=np.float64)
    for left, right in zip(starts, stops, strict=True):
        positions = order[left:right]
        active = dense_series[positions]
        slot = count[active] % window
        ring[active, slot] = margin[positions]
        count[active] += 1
        basis[positions] = np.max(ring[active], axis=1) - margin[positions]
    return basis


def _decayed_peak_basis(
    probability: np.ndarray,
    margin: np.ndarray,
    dense_series: np.ndarray,
    order: np.ndarray,
    starts: np.ndarray,
    stops: np.ndarray,
    series_count: int,
    decay_factor: float,
) -> np.ndarray:
    """Return logit(decayed probability peak) minus the current logit."""

    peak = np.zeros(series_count, dtype=np.float64)
    initialized = np.zeros(series_count, dtype=bool)
    basis = np.empty(margin.size, dtype=np.float64)
    for left, right in zip(starts, stops, strict=True):
        positions = order[left:right]
        active = dense_series[positions]
        current = probability[positions]
        first = ~initialized[active]
        if np.any(first):
            peak[active[first]] = current[first]
            initialized[active[first]] = True
        later = ~first
        if np.any(later):
            later_series = active[later]
            peak[later_series] = np.maximum(
                current[later], decay_factor * peak[later_series]
            )
        basis[positions] = safe_logit(peak[active]) - margin[positions]
    return basis


def apply_transform_grid(
    probability: Iterable[float] | np.ndarray,
    time_online: Iterable[int] | np.ndarray,
    series_position: Iterable[int] | np.ndarray,
    specs: Iterable[TransformSpec],
) -> Iterator[tuple[TransformSpec, np.ndarray]]:
    """Yield a full transform grid with vectorized time-slice state updates.

    The input order is preserved. Consecutive candidates sharing a family and
    structural parameter reuse one correction-basis array, which is why the
    frozen configuration groups weights inside each half-life or window. At
    most the raw margin, one basis, and the currently yielded ``float32``
    candidate are retained by this generator; callers should score and release
    each output before requesting the next one.
    """

    values, owner, online = validate_causal_order(
        probability, series_position, time_online
    )
    if online is None:  # pragma: no cover - required argument guarantees this.
        raise AssertionError("time_online validation unexpectedly returned None")
    margin = safe_logit(values)
    dense, order, starts, stops, series_count = _time_slice_layout(owner, online)
    seen: set[str] = set()
    cached_key: tuple[object, ...] | None = None
    cached_basis: np.ndarray | None = None

    for spec in specs:
        spec.validate()
        if spec.name in seen:
            raise ValueError(f"duplicate transform specification: {spec.name}")
        seen.add(spec.name)
        if spec.kind == "raw":
            yield spec, margin.astype(np.float32)
            continue

        if spec.kind == "logit_ema_trend":
            key = (spec.kind, float(spec.half_life))
            if key != cached_key:
                cached_basis = _ema_trend_basis(
                    margin,
                    dense,
                    order,
                    starts,
                    stops,
                    series_count,
                    float(spec.half_life),
                )
                cached_key = key
        elif spec.kind == "rolling_max_premium":
            key = (spec.kind, int(spec.window))
            if key != cached_key:
                cached_basis = _rolling_max_basis(
                    margin,
                    dense,
                    order,
                    starts,
                    stops,
                    series_count,
                    int(spec.window),
                )
                cached_key = key
        elif spec.kind == "decayed_peak_premium":
            key = (spec.kind, float(spec.decay_factor))
            if key != cached_key:
                cached_basis = _decayed_peak_basis(
                    values,
                    margin,
                    dense,
                    order,
                    starts,
                    stops,
                    series_count,
                    float(spec.decay_factor),
                )
                cached_key = key
        else:
            key = (
                spec.kind,
                float(spec.half_life),
                float(spec.minimum_scale),
                float(spec.innovation_clip),
            )
            if key != cached_key:
                cached_basis = _prior_innovation_basis(
                    margin,
                    dense,
                    order,
                    starts,
                    stops,
                    series_count,
                    float(spec.half_life),
                    float(spec.minimum_scale),
                    float(spec.innovation_clip),
                )
                cached_key = key
        if cached_basis is None:  # pragma: no cover - every family sets it.
            raise AssertionError("transform family did not build its correction basis")
        output = margin + spec.correction_weight * cached_basis
        if not np.all(np.isfinite(output)):
            raise AssertionError(f"non-finite grid output for {spec.name}")
        yield spec, output.astype(np.float32)


def apply_transform(
    score: Iterable[float] | np.ndarray,
    time_online: Iterable[int] | np.ndarray,
    series_position: Iterable[int] | np.ndarray,
    spec: TransformSpec,
) -> np.ndarray:
    """Apply ``spec`` with the Model 10 pipeline's stable positional API.

    ``score`` contains raw probabilities despite its deliberately generic
    name. Returning ``float32`` keeps a 4.2-million-row candidate near 16 MiB.
    No labels or cross-series information enter this transformation.
    """

    return apply_causal_transform(
        score,
        series_position,
        spec,
        time_online,
    ).astype(np.float32)


def make_streaming_transform(spec: TransformSpec) -> StreamingCausalTransform:
    """Construct fresh one-series streaming state for ``spec``."""

    return StreamingCausalTransform(spec)


__all__ = [
    "StreamingCausalTransform",
    "TransformKind",
    "TransformSpec",
    "apply_causal_transform",
    "apply_transform",
    "apply_transform_grid",
    "iter_causal_transform_scores",
    "make_streaming_transform",
    "safe_logit",
    "score_to_probability",
    "validate_causal_order",
]
