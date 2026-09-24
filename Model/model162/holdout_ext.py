"""Model162 step 2: does the 2025 external data help once it gets ORACLE-TEACHER soft targets?
In memory (no model files written): full-dev voters exactly as the FB1 + soft-synthetic recipe (Model152 FULL
recipe, oracle-teacher soft targets on real rows, splices and re-cuts - Model158), plus the external rows pooled
into the non-real block (40% total mass, as Model145 SX40) with soft targets = 0.5*y + 0.5*mean of the five
4-fold oracle teachers (none saw any external series).  Scored on the sealed holdout from Model152's components.
Reference (same recipe without external, measured the same way): student 0.635119, final 0.635908.
Usage: python models/model162/holdout_ext.py [EXT_FRACTION]     (fraction of external series used, default 1.0)
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(1, str(ROOT / "models/model152"))

import numpy as np  # noqa: E402
import lightgbm as lgb  # noqa: E402

import models.model147.run_model147 as m147  # noqa: E402
from models.model137.run_model137 import synthetic_block  # noqa: E402
from models.model152.run_full import all_dev_rows  # noqa: E402
from models.model20.run_model20 import reference_weights, splice_weights  # noqa: E402
from tsauc import tsauc  # noqa: E402


def main():
    frac = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    _, protocol, layout, real, pool, _ = m147.m111.load_everything()
    rec = dict(np.load(m147.HERE / "data/recuts.npz")); rec["split"] = np.load(m147.HERE / "data/recuts_split.npy")
    ext = dict(np.load(ROOT / "models/model145/data/external.npz"))
    if frac < 1.0:
        keep_series = np.random.default_rng(162).choice(np.unique(ext["series"]), int(frac * len(np.unique(ext["series"]))), replace=False)
        m = np.isin(ext["series"], keep_series); ext = {k: v[m] for k, v in ext.items()}
    else:
        m = np.ones(len(ext["target"]), bool)
    ext_t = np.load(ROOT / "models/model162/data/external_teacher.npy")[m]
    union = protocol["contract"]["union_names"]
    col0 = np.asarray([union.index(n) for n in protocol["contract"]["names"][0]])
    T = {j: lgb.Booster(model_file=str(ROOT / f"models/model152/runs/full_teacherB{j}.txt")) for j in range(5)}
    xt = np.hstack([ext["X_union"][:, col0], ext_t])
    ep = np.mean([T[j].predict(xt) for j in range(5)], axis=0); xt = None
    print(f"external rows {len(ep):,}: teacher TS-AUC {tsauc(ext['target'], ep, ext['time_online']):.4f}", flush=True)
    rt = np.hstack([rec["X_union"][:, col0], np.load(ROOT / "models/model147/data/recuts_teacher.npy"),
                    np.load(ROOT / "models/model158/data/recut_oracle.npy")])
    rp = np.empty(len(rt))
    for j in range(5):
        mk = rec["fold"] == j; rp[mk] = T[j].predict(rt[mk])
    rt = None
    ft = np.load(ROOT / "models/model152/runs/full_teacherB.npz")
    split_real = np.load(m147.D144 / "data_white/real.npy", mmap_mode="r")
    boosters = {}
    for v in range(5):
        seed = protocol["contract"]["recipes"][v]["seed"]
        names = protocol["contract"]["names"][v]; columns = np.asarray([union.index(n) for n in names])
        fit = all_dev_rows(layout, seed); y, t = layout.target[fit], layout.time_online[fit]
        dev = np.flatnonzero(layout.row_fold >= 0)
        w = reference_weights(y, t, layout.target[dev], layout.time_online[dev])
        soft = ft["prob"][np.searchsorted(ft["rows"], fit)]
        sx, sy, st = synthetic_block(protocol, layout, pool, v, names)
        su = np.load(ROOT / f"models/model155/data/splice_union_fold{v}.npy")
        sp = T[v].predict(np.hstack([su[:, col0], np.load(ROOT / f"models/model158/data/splice_teacher_fold{v}.npy")])); su = None
        ex = np.concatenate((sx, rec["X_union"][:, columns], ext["X_union"][:, columns]))
        ey = np.r_[sy, rec["target"], ext["target"]].astype(np.int8)
        et = np.r_[st, rec["time_online"], ext["time_online"]].astype(t.dtype)
        rw, sw, _ = splice_weights(w, y, t, ey, et, 0.4)
        target = np.r_[0.5 * y + 0.5 * soft, 0.5 * ey + 0.5 * np.r_[sp, rp, ep]].astype(np.float32)
        x = np.concatenate((real[fit][:, columns], ex)); ex = None
        extra = np.concatenate([np.asarray(split_real[fit]), np.load(m147.D144 / f"data_white/syn_fold{v}.npy"), rec["split"], ext["X_split"]])
        keep = np.load(m147.D144 / f"runs_white/E_fold{v}.npz")["columns"]
        x = np.concatenate((x, extra), axis=1)[:, keep]; extra = None
        model = lgb.LGBMRegressor(objective="cross_entropy", random_state=seed, n_jobs=8, verbosity=-1, deterministic=True,
                                  force_col_wise=True, **m147.m111.m84.CONFIG["arms"]["expressive"])
        model.fit(x, target, sample_weight=np.r_[rw, sw]); x = None
        boosters[v] = model.booster_; print("voter", v, "trained", flush=True)
    import offline_holdout as oh
    from split_tests import NAMES
    index = oh.x_columns(); X = np.load(oh.H / "X.npy", mmap_mode="r"); n = X.shape[0]
    ref = np.load(ROOT / "models/model123/holdout_predictions.npz"); yh, th = ref["target"], ref["step"]
    bc = oh.bundle_constants(); const = json.loads((ROOT / "models/model112/constants_m147.json").read_text())
    chain, gru = np.load(oh.H / "chain.npy"), np.load(oh.H / "gru.npy")
    z = np.zeros(n)
    for v in range(5):
        nm = list(protocol["contract"]["names"][v]) + list(NAMES)
        keep = np.load(m147.D144 / f"runs_white/E_fold{v}.npz")["columns"]; idx = np.asarray([index[nm[c]] for c in keep])
        p = np.empty(n)
        for lo in range(0, n, 200_000):
            p[lo:lo + 200_000] = boosters[v].predict(np.asarray(X[lo:lo + 200_000][:, idx]), num_threads=8)
        p = np.clip(p.astype(np.float32).astype(np.float64), 1e-7, 1 - 1e-7); z += np.log(p) - np.log1p(-p)
    z = z * 0.2; pe = np.clip(1 / (1 + np.exp(-z)), 1e-6, 1 - 1e-6); d = np.log(pe) - np.log1p(-pe)
    f = 1 / (1 + np.exp(-((chain - bc["mu_chain"]) / bc["sd_chain"] + bc["weight"] * (d - const["mu_dist"]) / const["sd_dist"]
                          + bc["weight_gru"] * (gru - bc["mu_gru"]) / bc["sd_gru"])))
    r, fn = tsauc(yh, d, th), tsauc(yh, f, th)
    print(f"HOLDOUT +external(frac {frac}): raw {r:.6f} final {fn:.6f} | vs reference raw {r - 0.635119:+.6f} final {fn - 0.635908:+.6f}", flush=True)


if __name__ == "__main__":
    main()
