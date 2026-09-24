"""Memory-conscious loader for labeled Structural Break real-time parquets.

The public files use a MultiIndex ``(id, time)``.  Period 1 is the complete,
guaranteed-clean historical reference; period 2 is the online stream.  The
loader converts that representation into contiguous per-series slices and
constructs the per-step target from ``tau_index``.  It never reads the
password-protected local test archive.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SeriesRecord:
    """One labeled training series in the same form supplied by Crunch."""

    dataset_id: int
    historical: np.ndarray
    online: np.ndarray
    online_time: np.ndarray
    tau_index: int

    @property
    def target(self) -> np.ndarray:
        if self.tau_index < 0:
            return np.zeros(self.online.size, dtype=np.uint8)
        return (np.arange(self.online.size) >= self.tau_index).astype(np.uint8)


class TrainingData:
    """Load a labeled train/test split and expose zero-copy series views."""

    def __init__(
        self,
        train_directory: str | Path = "train",
        *,
        split: str = "train",
    ) -> None:
        if split not in {"train", "test"}:
            raise ValueError("split must be 'train' or 'test'")
        directory = Path(train_directory)
        x_path = directory / f"X_{split}.parquet"
        index_path = directory / f"y_{split}_index.parquet"
        if not x_path.exists() or not index_path.exists():
            raise FileNotFoundError(
                f"expected {x_path} and {index_path}; run from the project root"
            )

        frame = pd.read_parquet(x_path, columns=["value", "period"])
        if frame.index.names != ["id", "time"]:
            raise ValueError(f"unexpected X index names: {frame.index.names}")
        if not frame.index.is_monotonic_increasing:
            frame = frame.sort_index()

        self.values = frame["value"].to_numpy(dtype=np.float64, copy=False)
        self.period = frame["period"].to_numpy(dtype=np.int8, copy=False)
        self.row_id = frame.index.get_level_values("id").to_numpy(dtype=np.int64)
        self.absolute_time = frame.index.get_level_values("time").to_numpy(
            dtype=np.int64
        )
        changes = np.flatnonzero(self.row_id[1:] != self.row_id[:-1]) + 1
        self.boundaries = np.r_[0, changes, self.row_id.size].astype(np.int64)
        self.series_ids = self.row_id[self.boundaries[:-1]].astype(np.int64)

        tau = pd.read_parquet(index_path).sort_index()
        if set(tau.columns) != {"tau_index", "tau"}:
            raise ValueError(f"unexpected y_train_index columns: {tau.columns}")
        missing = np.setdiff1d(self.series_ids, tau.index.to_numpy())
        if missing.size:
            raise ValueError(f"tau metadata missing {missing.size} series")
        self.tau_index = tau.loc[self.series_ids, "tau_index"].to_numpy(
            dtype=np.int64
        )

        # Locate the unique 1 -> 2 boundary for each series once.  The data
        # currently has all historical rows first; validating that assumption
        # makes accidental future leakage fail loudly.
        self.online_starts = np.empty(self.series_ids.size, dtype=np.int64)
        self.online_lengths = np.empty(self.series_ids.size, dtype=np.int64)
        self.historical_lengths = np.empty(self.series_ids.size, dtype=np.int64)
        for position, (left, right) in enumerate(
            zip(self.boundaries[:-1], self.boundaries[1:])
        ):
            local_period = self.period[left:right]
            online = np.flatnonzero(local_period == 2)
            if online.size == 0:
                raise ValueError(f"series {self.series_ids[position]} has no period 2")
            online_offset = int(online[0])
            if not np.all(local_period[:online_offset] == 1) or not np.all(
                local_period[online_offset:] == 2
            ):
                raise ValueError(
                    f"series {self.series_ids[position]} is not period-ordered"
                )
            self.online_starts[position] = left + online_offset
            self.historical_lengths[position] = online_offset
            self.online_lengths[position] = right - left - online_offset

        expected_rows = int(self.online_lengths.sum())
        y_path = directory / f"y_{split}.parquet"
        if y_path.exists():
            # This check guards both an off-by-one interpretation of tau and a
            # stale index file.  y itself is not retained in memory.
            supplied = pd.read_parquet(y_path, columns=["target"])
            if len(supplied) != expected_rows:
                raise ValueError("y_train row count does not match online periods")
            supplied_y = supplied["target"].to_numpy(dtype=np.uint8)
            generated_y = np.concatenate([record.target for record in self])
            if not np.array_equal(supplied_y, generated_y):
                raise ValueError("tau_index-derived labels disagree with y_train")

    def __len__(self) -> int:
        return int(self.series_ids.size)

    @property
    def online_row_count(self) -> int:
        return int(self.online_lengths.sum())

    def __iter__(self) -> Iterator[SeriesRecord]:
        for position in range(len(self)):
            yield self.record(position)

    def record(self, position: int) -> SeriesRecord:
        left = int(self.boundaries[position])
        split = int(self.online_starts[position])
        right = int(self.boundaries[position + 1])
        return SeriesRecord(
            dataset_id=int(self.series_ids[position]),
            historical=self.values[left:split],
            online=self.values[split:right],
            online_time=self.absolute_time[split:right],
            tau_index=int(self.tau_index[position]),
        )

    def series_table(self) -> pd.DataFrame:
        """Return one row per series for diagnostics and fold construction."""

        return pd.DataFrame(
            {
                "id": self.series_ids,
                "historical_length": self.historical_lengths,
                "online_length": self.online_lengths,
                "tau_index": self.tau_index,
                "has_break": (self.tau_index >= 0).astype(np.uint8),
            }
        ).set_index("id")
