"""Stage-3 grid summary: 8 transforms x 130 test/window combinations -> data/summary.csv + stdout.

Per transform: significant counts at .01/.05/.10, best test, and the transform's mean TS-AUC over
the 20 best identity tests (a 'how much does the transform help the strong tests' number).
Per test (window 0 = full prefix): best transform.
"""
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ORDER = ["identity", "abs", "square", "sqrt", "log", "diff", "rankgauss", "ranklaplace"]


def main():
    frames = {t: pd.read_csv(HERE / f"data/results_{t}.csv") for t in ORDER if (HERE / f"data/results_{t}.csv").exists()}
    grid = pd.concat(frames.values(), ignore_index=True)
    grid.to_csv(HERE / "data/summary.csv", index=False)
    ident = frames["identity"].sort_values("tsauc", ascending=False)
    top20 = [(r.test, int(r.window)) for r in ident.head(20).itertuples()]
    print("transform    sig.01 sig.05 sig.10  best test (window)      tsauc   se    mean(top-20 identity tests)")
    for t, d in frames.items():
        b = d.sort_values("tsauc", ascending=False).iloc[0]
        m = d.set_index(["test", "window"]).reindex(top20)["tsauc"].mean()
        print(f"{t:12s} {int((d.p < .01).sum()):5d} {int((d.p < .05).sum()):6d} {int((d.p < .10).sum()):6d}  "
              f"{b.test:20s} ({int(b.window):3d})  {b.tsauc:.4f} {b.se:.4f}   {m:.4f}")
    print("\nbest transform per test family (window 0 = whole online prefix):")
    w0 = grid[grid.window == 0]
    best = w0.loc[w0.groupby("test")["tsauc"].idxmax()].sort_values("tsauc", ascending=False)
    for r in best.head(25).itertuples():
        base = w0[(w0.test == r.test) & (w0["transform"] == "identity")]["tsauc"]
        base = float(base.iloc[0]) if len(base) else float("nan")
        print(f"  {r.test:22s} {r.transform:12s} {r.tsauc:.4f} (identity {base:.4f}, se {r.se:.4f})")
    n_all = grid.groupby("transform").size()
    print("\ncombinations significant at .01 on ALL transforms:",
          int((grid[grid.p < .01].groupby(["test", "window"]).size() == len(frames)).sum()), "of", int(n_all.max()))


if __name__ == "__main__":
    main()
