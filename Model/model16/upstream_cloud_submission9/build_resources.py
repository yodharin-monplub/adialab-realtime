#!/usr/bin/env python3
"""Export the timeout-safe Model 6 winner into this push-ready bundle.

This is an offline packaging tool, not part of Crunch inference.  It reads only
the already-fitted, train-derived Model 5/6 artifacts named by the frozen
contract.  OOF arrays, holdout predictions, labels, caches, and rejected models
are never copied. The retained detector boosters are merged for one native
LightGBM traversal and the random forest is converted losslessly to a compact,
pickle-free NumPy resource.

The deepest ExtraTrees component was removed because it dominated runtime for
only a 0.000045 development-TSAUC gain.  The retained six-component recipe was
fixed from nested development CV before the revised sealed holdout was scored.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np


BUNDLE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = BUNDLE_ROOT.parents[2]
RUN_ROOT = PROJECT_ROOT / "models/model6/runs/tsroc0p64_909b8f5c6c"
SCREEN_ROOT = PROJECT_ROOT / "models/model5/artifacts/screen"
FROZEN_PATH = RUN_ROOT / "frozen_contract.json"
EXPECTED_CONTRACT_HASH = (
    "92f5511cd65d6432bd35bcb2d8928fdd4cf2c1cea421ebc7299063db96ef7742"
)
DEPLOYMENT_PROFILE = "timeout_safe_v1"
EXCLUDED_COMPONENT = "core__feature__extra_trees__fisher_256__extra_deep"
EXPECTED_EXCLUDED_TUNING_TSROC = 0.6267246934597523
TIMEOUT_SAFE_TUNING_TSROC = 0.626679557852601
TIMEOUT_SAFE_HOLDOUT_TSROC = 0.6261435067543629
TIMEOUT_SAFE_HOLDOUT_SE = 0.011365359028780312
TIMEOUT_SAFE_HOLDOUT_INTERVAL = [0.6038788555511624, 0.6496036221514382]

CORE_LOGIT_COMPONENT = "core__stack__logit_mean"
CANDIDATE_EXTRA_COMPONENT = (
    "candidate__feature__extra_trees__gain_512__extra_sparse_columns"
)
CORE_XGBOOST_COMPONENT = "core__feature__xgboost__fisher_256__xgb_depth3"
CANDIDATE_META_COMPONENT = "candidate__stack__meta_lightgbm_gbdt"
CORE_META_COMPONENT = "core__stack__meta_linear_l2"
CANDIDATE_RF_COMPONENT = (
    "candidate__feature__random_forest__fisher_512__rf_shallow"
)


FEATURE_SOURCES = {
    CANDIDATE_EXTRA_COMPONENT: {
        "kind": "forest",
        "family": "candidate",
        "source": "candidate/extra_trees__gain_512__extra_sparse_columns",
    },
    CORE_XGBOOST_COMPONENT: {
        "kind": "xgboost",
        "family": "core",
        "source": "core/xgboost__fisher_256__xgb_depth3",
    },
    CANDIDATE_RF_COMPONENT: {
        "kind": "forest",
        "family": "candidate",
        "source": "candidate/random_forest__fisher_512__rf_shallow",
    },
}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _export_forest_bag(
    source: Path,
    selected_names: list[str],
    path: Path,
) -> None:
    """Merge five sklearn forests and map every split to the union row."""

    name_to_union = {name: index for index, name in enumerate(selected_names)}
    roots: list[int] = []
    group_offsets = [0]
    children_left: list[int] = []
    children_right: list[int] = []
    feature: list[int] = []
    threshold: list[float] = []
    positive_probability: list[float] = []
    max_depth = 0

    for fold in range(5):
        with np.load(source / f"fold_{fold}.columns.npz", allow_pickle=False) as payload:
            names = [str(value) for value in payload["names"]]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate forest feature names in fold {fold}")
        try:
            local_to_union = np.asarray(
                [name_to_union[name] for name in names], dtype=np.int32
            )
        except KeyError as error:
            raise ValueError(
                f"forest fold {fold} references unknown feature {error.args[0]!r}"
            ) from error
        fitted = joblib.load(source / f"fold_{fold}.joblib")
        model = fitted.model
        if int(model.n_features_in_) != len(names):
            raise ValueError(f"forest model/column mismatch in fold {fold}")
        if not np.array_equal(np.asarray(model.classes_), np.asarray([0, 1])):
            raise ValueError("compact forest requires classes ordered as [0, 1]")
        estimators = list(model.estimators_)
        if not estimators:
            raise ValueError(f"fitted forest fold {fold} contains no trees")
        for estimator in estimators:
            tree = estimator.tree_
            base = len(feature)
            count = int(tree.node_count)
            roots.append(base)
            left = np.asarray(tree.children_left, dtype=np.int64)
            right = np.asarray(tree.children_right, dtype=np.int64)
            children_left.extend(np.where(left >= 0, left + base, -1).tolist())
            children_right.extend(np.where(right >= 0, right + base, -1).tolist())
            local_feature = np.asarray(tree.feature, dtype=np.int64)
            split = local_feature >= 0
            mapped = np.full(count, -1, dtype=np.int32)
            mapped[split] = local_to_union[local_feature[split]]
            feature.extend(mapped.tolist())
            local_threshold = np.zeros(count, dtype=np.float64)
            local_threshold[split] = np.asarray(tree.threshold, dtype=np.float64)[split]
            threshold.extend(local_threshold.tolist())
            values = np.asarray(tree.value[:, 0, :], dtype=np.float64)
            if values.shape != (count, 2):
                raise ValueError("compact forest requires a binary classifier")
            denominator = values.sum(axis=1)
            probability = np.divide(
                values[:, 1],
                denominator,
                out=np.zeros(count, dtype=np.float64),
                where=denominator > 0.0,
            )
            positive_probability.extend(probability.tolist())
            max_depth = max(max_depth, int(tree.max_depth))
        group_offsets.append(len(roots))

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        format_version=np.asarray(1, dtype=np.int32),
        model_type=np.asarray("forest_bag"),
        feature_count=np.asarray(len(selected_names), dtype=np.int32),
        max_depth=np.asarray(max_depth, dtype=np.int32),
        voter_count=np.asarray(5, dtype=np.int32),
        roots=np.asarray(roots, dtype=np.int32),
        group_offsets=np.asarray(group_offsets, dtype=np.int32),
        children_left=np.asarray(children_left, dtype=np.int32),
        children_right=np.asarray(children_right, dtype=np.int32),
        feature=np.asarray(feature, dtype=np.int32),
        threshold=np.asarray(threshold, dtype=np.float64),
        positive_probability=np.asarray(positive_probability, dtype=np.float64),
    )


def _feature_reports(frozen: dict[str, Any]) -> dict[str, dict[str, Any]]:
    reports: dict[str, dict[str, Any]] = {}
    for family, block in frozen["feature_finalists"].items():
        for component in block["components"]:
            reports[f"{family}__feature__{component['component']}"] = component
    return reports


def _detector_reports(frozen: dict[str, Any]) -> dict[str, dict[str, Any]]:
    reports: dict[str, dict[str, Any]] = {}
    for family, block in frozen["detector_stack"].items():
        for component in block["components"]:
            reports[f"{family}__stack__{component['component']}"] = component
    return reports


def _append_lightgbm_tree(
    node: dict[str, Any],
    local_to_union: np.ndarray,
    children_left: list[int],
    children_right: list[int],
    feature: list[int],
    threshold: list[float],
    leaf_value: list[float],
    *,
    depth: int = 0,
) -> tuple[int, int]:
    """Append one numerical LightGBM tree and return its root/max depth."""

    index = len(feature)
    children_left.append(-1)
    children_right.append(-1)
    feature.append(-1)
    threshold.append(0.0)
    leaf_value.append(0.0)
    if "split_index" not in node:
        value = float(node["leaf_value"])
        if not np.isfinite(value):
            raise ValueError("LightGBM detector tree contains a non-finite leaf")
        leaf_value[index] = value
        return index, depth

    if node.get("decision_type") != "<=" or node.get("missing_type") != "None":
        raise ValueError("compact detector runtime supports finite <= splits only")
    local_feature = int(node["split_feature"])
    if local_feature < 0 or local_feature >= local_to_union.size:
        raise ValueError("LightGBM split feature is outside its bank contract")
    feature[index] = int(local_to_union[local_feature])
    threshold[index] = float(node["threshold"])
    left, left_depth = _append_lightgbm_tree(
        node["left_child"],
        local_to_union,
        children_left,
        children_right,
        feature,
        threshold,
        leaf_value,
        depth=depth + 1,
    )
    right, right_depth = _append_lightgbm_tree(
        node["right_child"],
        local_to_union,
        children_left,
        children_right,
        feature,
        threshold,
        leaf_value,
        depth=depth + 1,
    )
    children_left[index] = left
    children_right[index] = right
    return index, max(left_depth, right_depth)


def _remapped_lightgbm_trees(
    path: Path,
    local_to_union: np.ndarray,
    first_tree_index: int,
) -> tuple[list[str], list[np.ndarray], str]:
    """Read one booster and remap every textual split to union columns."""

    text = path.read_text(encoding="utf-8")
    first = text.index("Tree=")
    end = text.index("end of trees", first)
    body = text[first:end]
    starts = [match.start() for match in re.finditer(r"(?m)^Tree=\d+\n", body)]
    if not starts:
        raise ValueError(f"LightGBM model contains no tree sections: {path}")
    starts.append(len(body))
    sections: list[str] = []
    leaves: list[np.ndarray] = []
    for local_tree, (left, right) in enumerate(zip(starts[:-1], starts[1:])):
        lines = body[left:right].splitlines(keepends=True)
        if not lines or not lines[0].startswith("Tree="):
            raise ValueError(f"malformed LightGBM tree section: {path}")
        lines[0] = f"Tree={first_tree_index + local_tree}\n"
        num_leaves: int | None = None
        leaf_values: np.ndarray | None = None
        for index, line in enumerate(lines):
            if line.startswith("num_leaves="):
                num_leaves = int(line.split("=", 1)[1])
            elif line.startswith("split_feature="):
                local = [int(value) for value in line.split("=", 1)[1].split()]
                mapped = [int(local_to_union[value]) for value in local]
                lines[index] = "split_feature=" + " ".join(map(str, mapped)) + "\n"
            elif line.startswith("leaf_value="):
                leaf_values = np.asarray(
                    [float(value) for value in line.split("=", 1)[1].split()],
                    dtype=np.float64,
                )
        if num_leaves is None or leaf_values is None or leaf_values.size != num_leaves:
            raise ValueError(f"malformed LightGBM leaf values: {path}")
        section = "".join(lines)
        if not section.endswith("\n\n"):
            raise ValueError(f"LightGBM tree section lost its separator: {path}")
        sections.append(section)
        leaves.append(leaf_values)

    parameters = text.index("parameters:", end)
    parameter_footer = text[parameters:]
    return sections, leaves, parameter_footer


def _tree_depth(node: dict[str, Any]) -> int:
    if "split_index" not in node:
        return 0
    return 1 + max(_tree_depth(node["left_child"]), _tree_depth(node["right_child"]))


def _export_detector_ensemble(
    staging: Path,
    frozen: dict[str, Any],
    selected_names: list[str],
    detector_banks: dict[str, list[str]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Merge all bank voters for native C++ leaf traversal.

    A single LightGBM call traverses every source tree. ``pred_leaf`` returns
    the chosen local leaf for each tree; the companion NumPy lookup then
    reconstructs every original voter margin exactly. This is materially
    faster than advancing ten thousand tree roots in Python on every yield.
    """

    name_to_union = {name: index for index, name in enumerate(selected_names)}
    if len(name_to_union) != len(selected_names):
        raise ValueError("union detector feature names are duplicated")
    tree_sections: list[str] = []
    leaf_offsets: list[int] = []
    leaf_values: list[float] = []
    group_offsets = [0]
    bank_keys: list[list[str]] = []
    family_output: dict[str, Any] = {}
    max_depth = 0

    for family, order in detector_banks.items():
        family_output[family] = {"bank_order": order}
        for bank in order:
            bank_keys.append([family, bank])
            source = SCREEN_ROOT / family / bank
            source_manifest = json.loads(
                (source / "manifest.json").read_text(encoding="utf-8")
            )
            names = [str(value) for value in source_manifest["feature_names"]]
            if len(names) != len(set(names)):
                raise ValueError(f"duplicate source feature names for {family}/{bank}")
            try:
                local_to_union = np.asarray(
                    [name_to_union[name] for name in names], dtype=np.int32
                )
            except KeyError as error:
                raise ValueError(
                    f"{family}/{bank} references unknown feature {error.args[0]!r}"
                ) from error

            for fold in range(5):
                booster = lgb.Booster(model_file=str(source / f"fold_{fold}.txt"))
                if booster.num_feature() != len(names):
                    raise ValueError(f"model/column mismatch for {family}/{bank}/fold{fold}")
                dumped = booster.dump_model()
                if (
                    dumped.get("objective") != "binary sigmoid:1"
                    or dumped.get("average_output") is not False
                ):
                    raise ValueError("unexpected detector LightGBM objective")
                group_start = len(tree_sections)
                sections, leaves, current_footer = _remapped_lightgbm_trees(
                    source / f"fold_{fold}.txt",
                    local_to_union,
                    len(tree_sections),
                )
                if len(sections) != len(dumped["tree_info"]):
                    raise ValueError("LightGBM text/dump tree counts differ")
                if not tree_sections:
                    parameter_footer = current_footer
                tree_sections.extend(sections)
                for tree, values in zip(dumped["tree_info"], leaves):
                    leaf_offsets.append(len(leaf_values))
                    leaf_values.extend(values.tolist())
                    max_depth = max(max_depth, _tree_depth(tree["tree_structure"]))
                if len(tree_sections) == group_start:
                    raise ValueError(f"empty LightGBM voter for {family}/{bank}/fold{fold}")
                group_offsets.append(len(tree_sections))

    model_relative = Path("detectors/native_ensemble.txt")
    lookup_relative = Path("detectors/native_leaf_lookup.npz")
    model_path = staging / model_relative
    lookup_path = staging / lookup_relative
    model_path.parent.mkdir(parents=True, exist_ok=True)
    tree_sizes = [len(section.encode("utf-8")) for section in tree_sections]
    header = "\n".join(
        (
            "tree",
            "version=v4",
            "num_class=1",
            "num_tree_per_iteration=1",
            "label_index=0",
            f"max_feature_idx={len(selected_names) - 1}",
            "objective=binary sigmoid:1",
            "feature_names=" + " ".join(selected_names),
            "feature_infos=" + " ".join(["none"] * len(selected_names)),
            "tree_sizes=" + " ".join(map(str, tree_sizes)),
            "",
        )
    ) + "\n"
    footer = "end of trees\n\nfeature_importances:\n\n" + parameter_footer
    model_path.write_text(header + "".join(tree_sections) + footer, encoding="utf-8")
    np.savez_compressed(
        lookup_path,
        format_version=np.asarray(1, dtype=np.int32),
        model_type=np.asarray("native_lightgbm_detector_ensemble"),
        feature_count=np.asarray(len(selected_names), dtype=np.int32),
        voters_per_bank=np.asarray(5, dtype=np.int32),
        bank_count=np.asarray(len(bank_keys), dtype=np.int32),
        group_offsets=np.asarray(group_offsets, dtype=np.int32),
        leaf_offsets=np.asarray(leaf_offsets, dtype=np.int32),
        leaf_values=np.asarray(leaf_values, dtype=np.float64),
    )
    specification = {
        "kind": "native_lightgbm_detector_ensemble",
        "model_path": model_relative.as_posix(),
        "lookup_path": lookup_relative.as_posix(),
        "feature_names": selected_names,
        "bank_keys": bank_keys,
        "voters_per_bank": 5,
        "trees": len(tree_sections),
        "leaves": len(leaf_values),
        "max_depth": max_depth,
    }
    return family_output, specification


def _export_feature_components(
    staging: Path, frozen: dict[str, Any], selected_names: list[str]
) -> list[dict[str, Any]]:
    reports = _feature_reports(frozen)
    available = set(selected_names)
    output = []
    for name, specification in FEATURE_SOURCES.items():
        source = RUN_ROOT / "final_cv" / specification["source"]
        report = reports[name]
        common = {
            "name": name,
            "kind": specification["kind"],
            "family": specification["family"],
            "feature_contract": report["feature_contract"],
            "hyperparameters": report["hyperparameters"],
            "transform": report["postprocess"]["deployment_transform_spec"],
        }
        relative_directory = Path("feature_models") / name
        if specification["kind"] == "forest":
            relative = relative_directory / "five_voter_bag.npz"
            _export_forest_bag(source, selected_names, staging / relative)
            output.append(
                {
                    **common,
                    "path": relative.as_posix(),
                    "feature_names": selected_names,
                    "voter_count": 5,
                }
            )
            continue

        voters = []
        for fold in range(5):
            with np.load(
                source / f"fold_{fold}.columns.npz", allow_pickle=False
            ) as payload:
                names = [str(value) for value in payload["names"]]
            if len(names) != len(set(names)) or not set(names) <= available:
                raise ValueError(f"invalid frozen feature names for {name}/fold{fold}")
            fitted = joblib.load(source / f"fold_{fold}.joblib")
            relative = relative_directory / f"fold_{fold}.ubj"
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            fitted.model.save_model(destination)
            if int(fitted.model.n_features_in_) != len(names):
                raise ValueError(f"model/column mismatch for {name}/fold{fold}")
            voters.append({"path": relative.as_posix(), "feature_names": names})
        output.append({**common, "voters": voters})
    return output


def _export_meta_models(staging: Path) -> dict[str, Any]:
    core_source = (
        RUN_ROOT
        / "holdout_predictions/detector_meta/core/linear_l2/deployment.joblib"
    )
    core = joblib.load(core_source)
    core_relative = Path("meta_models/core_linear_l2.npz")
    core_path = staging / core_relative
    core_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        core_path,
        format_version=np.asarray(1, dtype=np.int32),
        model_type=np.asarray("linear_logistic"),
        mean=np.asarray(core.scaler.mean_, dtype=np.float64),
        scale=np.asarray(core.scaler.scale_, dtype=np.float64),
        coefficient=np.asarray(core.model.coef_[0], dtype=np.float64),
        intercept=np.asarray(core.model.intercept_[0], dtype=np.float64),
        clip=np.asarray(25.0, dtype=np.float64),
    )

    candidate_source = (
        RUN_ROOT
        / "holdout_predictions/detector_meta/candidate/lightgbm_gbdt/deployment.joblib"
    )
    candidate = joblib.load(candidate_source)
    candidate_relative = Path("meta_models/candidate_lightgbm_gbdt.txt")
    candidate_path = staging / candidate_relative
    candidate.model.save_model(
        candidate_path, num_iteration=int(candidate.best_iteration)
    )
    return {
        "core_linear_l2": {
            "kind": "linear_logistic",
            "path": core_relative.as_posix(),
        },
        "candidate_lightgbm_gbdt": {
            "kind": "lightgbm",
            "path": candidate_relative.as_posix(),
            "best_iteration": int(candidate.best_iteration),
        },
    }


def _timeout_safe_recipe(frozen: dict[str, Any]) -> dict[str, Any]:
    """Return the frozen global recipe without its negligible slow final step."""

    source = frozen["global_ensemble"]["deployment_recipe"]
    if source["start_component"] != CORE_LOGIT_COMPONENT:
        raise ValueError("the frozen global start component changed")
    steps = [dict(step) for step in source["steps"]]
    if not steps or steps[-1]["component"] != EXCLUDED_COMPONENT:
        raise ValueError("the expected timeout-heavy final component changed")
    excluded = steps.pop()
    if not np.isclose(
        float(excluded["tuning_tsroc_after_step"]),
        EXPECTED_EXCLUDED_TUNING_TSROC,
    ):
        raise ValueError("the frozen excluded-component score changed")
    if not steps or not np.isclose(
        float(steps[-1]["tuning_tsroc_after_step"]), TIMEOUT_SAFE_TUNING_TSROC
    ):
        raise ValueError("the timeout-safe recipe score changed")
    return {"start_component": CORE_LOGIT_COMPONENT, "steps": steps}


def _component_specs(frozen: dict[str, Any]) -> list[dict[str, Any]]:
    detector = _detector_reports(frozen)
    features = _feature_reports(frozen)
    definitions = {
        CORE_LOGIT_COMPONENT: {
            "kind": "bank_logit_mean",
            "family": "core",
            "transform": detector[CORE_LOGIT_COMPONENT]["postprocess"][
                "deployment_transform_spec"
            ],
        },
        CANDIDATE_EXTRA_COMPONENT: {
            "kind": "feature_bag",
            "transform": features[CANDIDATE_EXTRA_COMPONENT]["postprocess"][
                "deployment_transform_spec"
            ],
        },
        CORE_XGBOOST_COMPONENT: {
            "kind": "feature_bag",
            "transform": features[CORE_XGBOOST_COMPONENT]["postprocess"][
                "deployment_transform_spec"
            ],
        },
        CANDIDATE_META_COMPONENT: {
            "kind": "lightgbm_meta",
            "family": "candidate",
            "transform": detector[CANDIDATE_META_COMPONENT]["postprocess"][
                "deployment_transform_spec"
            ],
        },
        CORE_META_COMPONENT: {
            "kind": "linear_meta",
            "family": "core",
            "transform": detector[CORE_META_COMPONENT]["postprocess"][
                "deployment_transform_spec"
            ],
        },
        CANDIDATE_RF_COMPONENT: {
            "kind": "feature_bag",
            "transform": features[CANDIDATE_RF_COMPONENT]["postprocess"][
                "deployment_transform_spec"
            ],
        },
    }
    recipe = _timeout_safe_recipe(frozen)
    order = [recipe["start_component"]] + [
        step["component"] for step in recipe["steps"]
    ]
    if set(order) != set(definitions) or len(order) != len(definitions):
        raise ValueError("frozen global recipe differs from Submission 6 components")
    return [{"name": name, **definitions[name]} for name in order]


def _write_bundle_manifest() -> None:
    entries = []
    for path in sorted(BUNDLE_ROOT.rglob("*")):
        if not path.is_file() or path.name == "bundle_manifest.json":
            continue
        relative = path.relative_to(BUNDLE_ROOT)
        if relative.parts[0] in {".crunchdao", "data", "prediction"}:
            continue
        if "__pycache__" in relative.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        entries.append(
            {
                "path": relative.as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    _write_json(
        BUNDLE_ROOT / "bundle_manifest.json",
        {
            "submission": "submission_6",
            "frozen_contract_hash": EXPECTED_CONTRACT_HASH,
            "deployment_profile": DEPLOYMENT_PROFILE,
            "files": entries,
            "total_bytes": sum(item["bytes"] for item in entries),
        },
    )


def build(destination_path: str | Path | None = None) -> Path:
    # Joblib records repository-qualified model wrapper classes.  Make their
    # modules importable regardless of the caller's current working directory.
    project_root_text = str(PROJECT_ROOT)
    if project_root_text not in sys.path:
        sys.path.insert(0, project_root_text)

    default_destination = BUNDLE_ROOT / "resources"
    destination = (
        default_destination if destination_path is None else Path(destination_path)
    )
    if destination.exists():
        raise FileExistsError(
            f"refusing to overwrite existing resources: {destination}"
        )
    frozen = json.loads(FROZEN_PATH.read_text(encoding="utf-8"))
    if frozen.get("contract_hash") != EXPECTED_CONTRACT_HASH:
        raise ValueError("completed Model 6 frozen contract hash changed")

    _timeout_safe_recipe(frozen)
    core_banks = [str(value) for value in frozen["detector_stack"]["core"]["banks"]]
    candidate_banks = [
        str(value) for value in frozen["detector_stack"]["candidate"]["banks"]
    ]
    union_banks = core_banks + [bank for bank in candidate_banks if bank not in core_banks]
    # Import from the exact production copy so packaging fails if copied
    # detector code and selected model names are no longer compatible.
    try:
        from .feature_pipeline import feature_names
    except ImportError:
        from feature_pipeline import feature_names

    selected_names = feature_names(union_banks)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(
        prefix=".resources_build_", dir=destination.parent
    ) as raw:
        staging = Path(raw)
        detector_families, detector_ensemble = _export_detector_ensemble(
            staging,
            frozen,
            selected_names,
            {"core": core_banks, "candidate": candidate_banks},
        )
        feature_components = _export_feature_components(
            staging, frozen, selected_names
        )
        meta_models = _export_meta_models(staging)
        components = _component_specs(frozen)
        global_recipe = _timeout_safe_recipe(frozen)
        resource_hashes = {
            path.relative_to(staging).as_posix(): _sha256(path)
            for path in sorted(staging.rglob("*"))
            if path.is_file()
        }
        manifest = {
            "format_version": 2,
            "submission": "submission_6",
            "experiment_version": frozen["experiment_version"],
            "frozen_contract_hash": EXPECTED_CONTRACT_HASH,
            "pretrained_models_bundled": True,
            "training_source": "updated public train/ folder only",
            "competition_test_data_used": False,
            "holdout_tsroc": TIMEOUT_SAFE_HOLDOUT_TSROC,
            "holdout_bootstrap_standard_error": TIMEOUT_SAFE_HOLDOUT_SE,
            "union_banks": union_banks,
            "detector_families": detector_families,
            "detector_ensemble": detector_ensemble,
            "feature_components": feature_components,
            "meta_models": meta_models,
            "components": components,
            "global_recipe": {
                "start_component": global_recipe["start_component"],
                "steps": global_recipe["steps"],
            },
            "global_transform": frozen["global_ensemble"]["postprocess"][
                "deployment_transform_spec"
            ],
            "five_fold_voters_bagged": True,
            "file_sha256": resource_hashes,
            "deployment_profile": DEPLOYMENT_PROFILE,
            "excluded_timeout_component": EXCLUDED_COMPONENT,
            "tuning_tsroc": TIMEOUT_SAFE_TUNING_TSROC,
            "holdout_bootstrap_95_interval": TIMEOUT_SAFE_HOLDOUT_INTERVAL,
        }
        _write_json(staging / "manifest.json", manifest)
        os.replace(staging, destination)

    if destination == default_destination:
        _write_bundle_manifest()
    print(f"built {destination}")
    print(f"resource files: {len(resource_hashes):,}")
    print(
        f"resource size: {sum(path.stat().st_size for path in destination.rglob('*') if path.is_file()) / 1024**2:.1f} MiB"
    )
    return destination


if __name__ == "__main__":
    build()
