"""Local verification only. Does not upload or start cloud jobs."""
import contextlib
import io
import json
from pathlib import Path
import sys
import time
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "submissions/submission29"
sys.path.insert(0, str(BUNDLE))
from main import infer, INFER_PARALLELISM, train


def references():
    frame = pd.read_parquet(BUNDLE / "resources/X_test.reduced.parquet").sort_index()
    return [(key, group.loc[group.period == 1, "value"].to_numpy(dtype=float),
             group.loc[group.period == 2, "value"].to_numpy(dtype=float))
            for key, group in frame.groupby(level="id")]


def execute(records):
    consumed = 0
    def online(values):
        nonlocal consumed
        for value in values:
            consumed += 1
            yield value
    def datasets():
        for key, history, values in records:
            if key is None:
                yield history, online(values)
            else:
                yield key, history, online(values)
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        generator = infer(datasets(), BUNDLE / "resources")
        assert next(generator) is None
        assert consumed == 0
        predictions = 0
        for score in generator:
            predictions += 1
            assert consumed == predictions, "Online iterator consumed ahead of prediction"
            assert score == 0.5
    reports = [json.loads(line.removeprefix("OVERLAP_FINAL "))
               for line in log.getvalue().splitlines() if line.startswith("OVERLAP_FINAL ")]
    assert len(reports) == 1
    assert consumed == predictions
    return reports[0]


class ProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.refs = references()

    def test_all_100_reference_series(self):
        self.assertEqual(INFER_PARALLELISM, 1)
        train(iter(()), BUNDLE / "resources")
        report = execute(self.refs)
        for mode in report["modes"].values():
            self.assertEqual(mode["yes_cloud_series"], 100)
            self.assertEqual(mode["reference_found_percentage"], 100)
        self.assertTrue(report["complete"])
        self.assertFalse(report["expected_cloud_count_reached"])

    def test_online_changes_lengths_duplicates_and_ids(self):
        key, history, values = self.refs[0]
        changed = values.copy()
        changed[-1] += 1
        report = execute([(9999999, history, values), (None, history, values),
                          (key, history, changed), (key, history, values[:-1])])
        mode = report["modes"]["exact"]
        self.assertEqual(mode["yes_cloud_series"], 2)
        self.assertEqual(mode["yes_percentage"], 50)
        self.assertEqual(mode["reference_series_found"], 1)
        self.assertEqual(mode["history_only_yes_cloud_series"], 4)

    def test_precision_is_reported_separately(self):
        key, history, values = self.refs[0]
        changed = history.copy()
        i = int(np.argmax(np.abs(changed)))
        changed[i] = np.nextafter(changed[i], np.inf)
        report = execute([(key, changed, values)])
        self.assertEqual(report["modes"]["exact"]["yes_cloud_series"], 0)
        self.assertEqual(report["modes"]["public_float32_precision"]["yes_cloud_series"], 1)

    def test_empty(self):
        report = execute([])
        self.assertEqual(report["completed_cloud_series"], 0)
        self.assertIsNone(report["modes"]["exact"]["yes_percentage"])

    def test_partial_stream(self):
        key, history, values = self.refs[0]
        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            generator = infer(iter([(key, history, iter(values))]), BUNDLE / "resources")
            self.assertIsNone(next(generator))
            self.assertEqual(next(generator), 0.5)
            generator.close()
        report = json.loads(next(line.removeprefix("OVERLAP_FINAL ")
                                for line in log.getvalue().splitlines()
                                if line.startswith("OVERLAP_FINAL ")))
        self.assertFalse(report["complete"])
        self.assertEqual(report["completed_cloud_series"], 0)

    def test_10000_series_simulation(self):
        _, history, values = self.refs[0]
        negative_history = history.copy()
        negative_history[0] += 123.0
        def records():
            yield from self.refs
            for i in range(9900):
                yield i, negative_history, values
        started = time.monotonic()
        report = execute(records())
        report["simulation_seconds"] = time.monotonic() - started
        report["source"] = "LOCAL SIMULATION: 100 public references plus 9900 deliberate nonmatches"
        self.assertTrue(report["expected_cloud_count_reached"])
        for mode in report["modes"].values():
            self.assertEqual(mode["yes_cloud_series"], 100)
            self.assertEqual(mode["yes_percentage"], 1)
            self.assertEqual(mode["reference_series_found"], 100)
            self.assertEqual(mode["reference_found_percentage"], 100)
        (Path(__file__).parent / "local_validation.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    unittest.main(verbosity=2)
