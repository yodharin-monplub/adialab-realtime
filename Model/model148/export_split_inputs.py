"""Model148: GRU student with the 27 whitened split-test features (Model144) added to its inputs.

Takes Model124's export (102 standardized inputs per step, soft targets, folds) and appends the 27
split-test columns of the same rows, standardized with label-free global constants and clipped to
+/-10 exactly like the other inputs.  Output: models/model148/data/sequences_split.npz + contract json.
Train with the unchanged trainer:  python models/model124/train_gru.py --data <this file> --fold k ...
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "data"


def main():
    OUT.mkdir(exist_ok=True)
    base = np.load(ROOT / "models/model124/data/sequences.npz")
    split = np.load(ROOT / "models/model144/data_white/real.npy").astype(np.float32)
    assert len(split) == len(base["target"])
    mu, sd = split.mean(0), split.std(0) + 1e-6
    z = np.clip((split - mu) / sd, -10, 10).astype(np.float16)
    X = np.concatenate((base["X"], z), axis=1)
    np.savez(OUT / "sequences_split.npz", X=X, soft=base["soft"], target=base["target"], time_online=base["time_online"],
             row_fold=base["row_fold"], series_position=base["series_position"], series_ids=base["series_ids"])
    contract = json.loads((ROOT / "models/model124/data/input_contract.json").read_text())
    from sys import path
    path.insert(0, str(ROOT))
    from models.model147.deployment.split_tests_numpy import NAMES
    contract["split_names"] = list(NAMES); contract["split_mu"] = mu.tolist(); contract["split_sd"] = sd.tolist()
    (OUT / "input_contract_split.json").write_text(json.dumps(contract, indent=2))
    print("saved", X.shape)


if __name__ == "__main__":
    main()
