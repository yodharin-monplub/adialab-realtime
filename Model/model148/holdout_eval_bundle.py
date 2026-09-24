"""Sealed-holdout evaluation of a BUILT bundle, exactly as it ships (its own SeriesPredictor).

Usage: python models/model148/holdout_eval_bundle.py submission34 solution148 [--workers 6] [--limit N]
Resumable (checkpoint every 100 series).  Compares with Submission31 and with the best earlier
candidate (Model147 BAG3 = Submission33) on the same 1,667 sealed series.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import importlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
_state = {}


def init_worker(bundle_name, module_name):
    bundle = ROOT / "submissions" / bundle_name
    sys.path.insert(0, str(bundle))
    module = importlib.import_module(module_name)
    _state["module"] = module
    _state["bundle"] = module.Bundle(bundle / "resources")
    _state["frame"] = pd.read_parquet(ROOT / "train/X_train.parquet")


def run_series(sid):
    series = _state["frame"].loc[sid]
    history = series.loc[series.period == 1, "value"].to_numpy(dtype=np.float32)
    online = series.loc[series.period == 2, "value"].to_numpy(dtype=np.float32)
    predictor = _state["module"].SeriesPredictor(_state["bundle"], history)
    return int(sid), np.asarray([predictor.step(v) for v in online], dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle"); ap.add_argument("module")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    sys.path.insert(0, str(ROOT))
    ref = np.load(ROOT / "models/model123/holdout_predictions.npz")
    ids = ref["ids"][:a.limit] if a.limit else ref["ids"]
    partial = HERE / f"holdout_partial_{a.bundle}.npz"
    final = {}
    if partial.exists() and not a.limit:
        saved = np.load(partial); final = {int(k): saved[k] for k in saved.files}
        print(f"resuming: {len(final)} series already done", flush=True)
    todo = [int(s) for s in ids.tolist() if int(s) not in final]
    started = time.perf_counter(); done = len(final)
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context("spawn"), initializer=init_worker,
                             initargs=(a.bundle, a.module)) as pool:
        for sid, f in pool.map(run_series, todo, chunksize=4):
            final[sid] = f; done += 1
            if done % 100 == 0:
                print(f"{done}/{len(ids)} series, {time.perf_counter() - started:.0f}s", flush=True)
                if not a.limit:
                    np.savez(partial, **{str(k): v for k, v in final.items()})
    from tsauc import tsauc
    cand = np.concatenate([final[int(s)] for s in ids]); n = len(cand)
    y, t = ref["target"][:n], ref["step"][:n]
    s31 = np.load(ROOT / "models/model124/holdout_predictions31.npz")["final"][:n]
    bag3 = np.load(ROOT / "models/model147/holdout_predictions_BAG3.npz")["final"][:n]
    result = {"bundle": a.bundle, "series": int(len(ids)), "rows": int(n),
              "submission31_final": tsauc(y, s31, t), "submission33_final": tsauc(y, bag3, t),
              "candidate_final": tsauc(y, cand, t), "seconds": time.perf_counter() - started}
    result["gain_vs_sub31"] = result["candidate_final"] - result["submission31_final"]
    result["gain_vs_sub33"] = result["candidate_final"] - result["submission33_final"]
    suffix = "_smoke" if a.limit else ""
    np.savez(HERE / f"holdout_predictions_{a.bundle}{suffix}.npz", final=cand, ids=ids)
    (HERE / f"holdout_results_{a.bundle}{suffix}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
