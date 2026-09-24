"""Model153: score full-refit GRUs on the sealed holdout offline (Model152 saved components).

GRU inputs are rebuilt from models/model152/holdout/X.npy exactly as Submission31's Bundle.gru_input does
([61 whitening, 40 detector columns, log1p(t)], standardized with the bundle constants, clip, float16).
  check                  deployed 15 GRUs (numpy, from the bundle) on the first N series vs saved gru.npy
  score TAG STUDENT CFG... final = chain + 0.5 z(student) + 0.5 z(mean logit of the given full GRUs),
                         constants from the bundle / BAG3 (nothing tuned on the holdout).
                         STUDENT = a models/model152/holdout_student_{STUDENT}.npz tag (e.g. BAGF3, BAG3)
                         CFG = runs/{CFG}_weights.npz stems, e.g. h96l3_s0 h192_s0 h128_s0
Usage: python models/model153/offline_gru.py check | score TAG STUDENT CFG [CFG ...]
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
H = ROOT / "models/model152/holdout"
BUNDLE = ROOT / "submissions/submission31"
sys.path.insert(0, str(BUNDLE)); sys.path.insert(1, str(ROOT)); sys.path.insert(2, str(ROOT / "models/model144"))
LAYERS = {"h192": 2, "h96l3": 3, "h128": 2}


def load_bundle():
    import solution124
    return solution124.Bundle(BUNDLE / "resources")


def gru_inputs(bundle, rows=None):
    """(n, 102) float32 GRU inputs for holdout rows (all rows if None)."""
    sys.path.insert(0, str(ROOT / "models/model152"))
    from offline_holdout import x_columns
    index = x_columns()
    contract = json.loads((ROOT / "models/model124/data/input_contract.json").read_text())
    names = contract["names"]; assert names[-1] == "log1p_t" and len(names) == 102
    cols = np.asarray([index[n] for n in names[:-1]])
    X = np.load(H / "X.npy", mmap_mode="r")
    step = np.load(ROOT / "models/model123/holdout_predictions.npz")["step"]
    rows = np.arange(len(step)) if rows is None else rows
    out = np.empty((len(rows), 102), np.float32)
    for lo in range(0, len(rows), 200_000):
        r = rows[lo:lo + 200_000]
        raw = np.concatenate((np.asarray(X[r][:, cols]), np.log1p(step[r].astype(np.float64)).astype(np.float32)[:, None]), axis=1)
        np.nan_to_num(raw, copy=False, nan=0.0, posinf=1e6, neginf=-1e6)
        z = np.clip((raw - bundle.gru_mu) / bundle.gru_sd, -bundle.gru_clip, bundle.gru_clip)
        out[lo:lo + len(r)] = z.astype(np.float16).astype(np.float32)
    return out


def series_bounds():
    step = np.load(ROOT / "models/model123/holdout_predictions.npz")["step"]
    starts = np.flatnonzero(step == 0)
    return starts, np.r_[starts[1:], len(step)]


def torch_logits(weight_file, layers, Z, starts, ends):
    import torch
    from models.model124.train_gru import GRUStudent
    w = dict(np.load(weight_file))
    hidden = w["gru.weight_hh_l0"].shape[1]
    model = GRUStudent(Z.shape[1], hidden, layers)
    model.load_state_dict({k: torch.from_numpy(v) for k, v in w.items()}); model.eval()
    out = np.empty(len(Z), np.float32)
    order = np.argsort(ends - starts)
    with torch.no_grad():
        for i in range(0, len(order), 64):
            idx = order[i:i + 64]
            T = int((ends[idx] - starts[idx]).max())
            xb = np.zeros((len(idx), T, Z.shape[1]), np.float32)
            for j, s in enumerate(idx):
                xb[j, :ends[s] - starts[s]] = Z[starts[s]:ends[s]]
            lo = model(torch.from_numpy(xb)).numpy()
            for j, s in enumerate(idx):
                out[starts[s]:ends[s]] = lo[j, :ends[s] - starts[s]]
    return out


def main():
    args = sys.argv[1:]
    bundle = load_bundle()
    starts, ends = series_bounds()
    if args[0] == "check":
        n = 20
        rows = np.arange(starts[0], ends[n - 1])
        Z = gru_inputs(bundle, rows)
        saved = np.load(H / "gru.npy")[rows]
        ours = np.zeros(len(rows), np.float64)
        for gru in bundle.grus:
            for a, b in zip(starts[:n], ends[:n]):
                st = gru.initial_state()
                ours[a:b] += [gru.step(Z[i], st) for i in range(a, b)]
        ours /= len(bundle.grus)
        print(f"{n} series, {len(rows):,} rows: max |mean GRU logit - saved| {np.abs(ours - saved).max():.2e}")
        return
    from tsauc import tsauc
    tag, student, cfgs = args[1], args[2], args[3:]
    Z = gru_inputs(bundle)
    logits = [torch_logits(HERE / f"runs/{c}_weights.npz", LAYERS[c.split("_s")[0]], Z, starts, ends) for c in cfgs]
    g = np.mean(logits, axis=0)
    np.save(HERE / f"holdout_gru_{tag}.npy", g)
    ref = np.load(ROOT / "models/model123/holdout_predictions.npz"); y, t = ref["target"], ref["step"]
    chain, gru_dep = np.load(H / "chain.npy"), np.load(H / "gru.npy")
    st = np.load(ROOT / f"models/model152/holdout_student_{student}.npz"); assert len(st["rows"]) == len(y)
    const = json.loads((ROOT / "models/model112/constants_m147_bag3.json").read_text())
    zs = bundle.weight * (st["dist"] - const["mu_dist"]) / const["sd_dist"]
    zc = (chain - bundle.mu_chain) / bundle.sd_chain
    fin = lambda gg: 1 / (1 + np.exp(-(zc + zs + bundle.weight_gru * (gg - bundle.mu_gru) / bundle.sd_gru)))  # noqa: E731
    res = dict(tag=tag, student=student, configs=cfgs,
               gru_raw_deployed=tsauc(y, gru_dep, t), gru_raw_full=tsauc(y, g, t),
               final_deployed_gru=tsauc(y, fin(gru_dep), t), final_full_gru=tsauc(y, fin(g), t),
               sub33_bag3=tsauc(y, np.load(ROOT / "models/model147/holdout_predictions_BAG3.npz")["final"], t))
    res["gain_full_gru"] = res["final_full_gru"] - res["final_deployed_gru"]
    res["gain_vs_sub33"] = res["final_full_gru"] - res["sub33_bag3"]
    np.savez(HERE / f"holdout_final_{tag}.npz", final=fin(g).astype(np.float32), rows=np.arange(len(y)), dist=st["dist"])
    (HERE / f"holdout_{tag}.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
