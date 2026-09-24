"""Summarize matched inference benchmarks and input-precision sensitivity."""
from pathlib import Path
import json
import sys
import numpy as np
import pyarrow.dataset as ds

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "models/model17/deployment_validation"
sys.path.insert(0, str(ROOT))
from tsauc import tsauc


def main():
    reports = {name: json.loads((OUTPUT / f"{name}_w4_n24.json").read_text())
               for name in ("model16", "model17", "model16_full_precision")}
    with np.load(OUTPUT / "model16_w4_n24.npz") as p:
        ids, lengths, cloud = p["id"], p["lengths"], p["prediction"]
    with np.load(OUTPUT / "model16_full_precision_w4_n24.npz") as p:
        np.testing.assert_array_equal(ids, p["id"])
        np.testing.assert_array_equal(lengths, p["lengths"])
        local = p["prediction"]
    with np.load(ROOT / "models/artifacts/core_features.metadata.npz") as p:
        all_ids = p["series_ids"]
    with np.load(ROOT / "models/model5/artifacts/split_assignments.npz") as p:
        allowed = all_ids[p["inner_series_fold"] >= 0]
    assert np.isin(ids, allowed).all(), "Non-development IDs in precision audit"
    labels = ds.dataset(ROOT / "train/y_train_index.parquet", format="parquet").to_table(
        columns=["id", "tau_index"], filter=ds.field("id").isin(ids.tolist())).to_pydict()
    tau = dict(zip(labels["id"], labels["tau_index"]))
    assert set(tau) == set(ids.tolist())
    steps = np.concatenate([np.arange(n) for n in lengths])
    target = np.concatenate([((np.arange(n) >= tau[i]) & (tau[i] >= 0)).astype(np.uint8) for i, n in zip(ids, lengths)])
    absolute = np.abs(cloud.astype(float) - local.astype(float))
    full_score, cloud_score = tsauc(target, local, steps), tsauc(target, cloud, steps)
    precision = dict(series=len(ids), observations=len(cloud), mean_absolute_probability_change=float(absolute.mean()),
                     maximum_absolute_probability_change=float(absolute.max()), fraction_changed=float(np.mean(absolute > 0)),
                     sample_tsauc_float64_inputs=full_score, sample_tsauc_float32_inputs=cloud_score,
                     sample_tsauc_delta=cloud_score-full_score,
                     caveat="Same fitted deployment on a small development sample; these are NOT CV, independent validation, or cloud scores.")
    ratio = reports["model17"]["wall_seconds"] / reports["model16"]["wall_seconds"]
    cloud_reference = json.loads((OUTPUT / "model16_cloud_reference.json").read_text())
    assert cloud_reference["success"] and cloud_reference["cpu"] == 16384 and cloud_reference["memory"] == 65536
    old_cloud_seconds = cloud_reference["duration"]
    estimated_hours = old_cloud_seconds * ratio / 3600
    output = dict(precision_audit=precision, matched_elapsed_ratio_model17_over_model16=ratio,
                  previous_actual_cloud_seconds=old_cloud_seconds, nominal_model17_cloud_hours=estimated_hours,
                  nominal_12_worker_rss_gib=12*reports["model17"]["peak_worker_rss_mib"]/1024,
                  independent_validation_complete=False, no_outer_labels_loaded=True, no_cloud_run_started=True)
    (OUTPUT / "audit.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
