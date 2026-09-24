"""Fast Model 9 protocol tests; no learner or feature cache is built."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import nbformat
import numpy as np

from models.model9.config import MODEL9_VERSION, make_model9_config
from models.model9 import model9_pipeline as model9
from models.model9 import pairwise_stack as pairwise


PROJECT = Path(__file__).resolve().parents[1]


def _expect_value_error(function, *args, **kwargs) -> None:
    try:
        function(*args, **kwargs)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def _synthetic_role_context() -> SimpleNamespace:
    """Return five development folds plus two explicitly sealed series."""

    lengths = np.asarray([2, 3, 4, 5, 2, 4, 3, 5, 4, 2, 6, 3], dtype=np.int64)
    owner = np.repeat(np.arange(lengths.size, dtype=np.int64), lengths)
    series_fold = np.asarray([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, -1, -1])
    row_fold = series_fold[owner]
    development_mask = row_fold >= 0
    split = SimpleNamespace(
        row_inner_fold=row_fold,
        development_rows=np.flatnonzero(development_mask),
        development_row_mask=development_mask,
        holdout_row_mask=~development_mask,
    )
    data = SimpleNamespace(metadata={"series_position": owner})
    config = SimpleNamespace(folds=5, seed=91_009)
    return SimpleNamespace(config=config, source=SimpleNamespace(split=split, data=data))


def test_model9_configuration_freezes_full_data_experiment() -> None:
    config = make_model9_config(PROJECT)
    config.validate()
    assert config.protocol()["experiment_version"] == MODEL9_VERSION
    assert config.folds == 5
    assert config.architectures == (
        "candidate_all_gbdt",
        "candidate_split512_gbdt",
        "core_fisher512_gbdt",
        "joint_fisher768_gbdt",
    )
    assert config.scale_train_rows == tuple(sorted(set(config.scale_train_rows)))
    assert len(config.scale_train_rows) >= 2
    assert config.blend_weights[0] == 0.0
    assert config.blend_weights[-1] == 1.0
    assert config.artifact_directory.startswith("models/model9/runs/fulldata_")


def test_model9_complete_series_cap_is_deterministic_and_never_splits() -> None:
    owner = np.repeat(np.arange(8, dtype=np.int64), [2, 5, 3, 4, 6, 2, 7, 3])
    eligible = np.arange(owner.size, dtype=np.int64)
    first = model9.complete_series_cap(eligible, owner, maximum_rows=13, seed=812)
    second = model9.complete_series_cap(eligible, owner, maximum_rows=13, seed=812)
    assert np.array_equal(first, second)
    assert 0 < first.size <= 13
    selected_series = np.unique(owner[first])
    for series in selected_series:
        assert np.array_equal(first[owner[first] == series], eligible[owner == series])

    # A non-binding cap preserves the exact eligible order and membership.
    reversed_rows = eligible[::-1]
    uncapped = model9.complete_series_cap(
        reversed_rows, owner, maximum_rows=owner.size, seed=99
    )
    assert np.array_equal(uncapped, reversed_rows)


def test_model9_nested_row_cap_is_deterministic_and_nested() -> None:
    rows = np.arange(100, 300, dtype=np.int64)
    small = model9.nested_row_cap(rows, maximum_rows=40, seed=712)
    large = model9.nested_row_cap(rows, maximum_rows=125, seed=712)
    repeated = model9.nested_row_cap(rows, maximum_rows=40, seed=712)
    assert np.array_equal(small, repeated)
    assert small.size == 40 and large.size == 125
    assert set(small).issubset(set(large))
    assert np.all(np.diff(small) > 0) and np.all(np.diff(large) > 0)


def test_model9_fold_roles_are_series_disjoint_and_exclude_outer_holdout() -> None:
    context = _synthetic_role_context()
    owner = context.source.data.metadata["series_position"]
    prohibited = context.source.split.holdout_row_mask
    for held in range(context.config.folds):
        fit, stop, score, stop_fold = model9.role_rows(
            context, held, train_cap=12, stop_cap=5
        )
        assert stop_fold == (held + 1) % context.config.folds
        all_rows = np.concatenate((fit, stop, score))
        assert not np.any(prohibited[all_rows])
        role_series = [set(owner[rows]) for rows in (fit, stop, score)]
        assert not (
            role_series[0] & role_series[1]
            or role_series[0] & role_series[2]
            or role_series[1] & role_series[2]
        )
        # The score fold is never subsampled. Fit/stop row caps operate only
        # after complete series have been assigned to disjoint roles.
        for series in np.unique(owner[score]):
            expected = np.flatnonzero(
                (owner == series) & ~context.source.split.holdout_row_mask
            )
            assert np.array_equal(score[owner[score] == series], expected)
        row_fold = context.source.split.row_inner_fold
        assert np.all(row_fold[score] == held)
        assert np.all(row_fold[stop] == stop_fold)
        assert np.all((row_fold[fit] != held) & (row_fold[fit] != stop_fold))

        larger_fit, larger_stop, larger_score, _ = model9.role_rows(
            context, held, train_cap=24, stop_cap=10
        )
        assert set(fit).issubset(set(larger_fit))
        assert set(stop).issubset(set(larger_stop))
        assert np.array_equal(score, larger_score)


def test_model9_nested_selection_does_not_use_held_fold_score() -> None:
    fold = np.repeat(np.arange(5, dtype=np.int64), 2)
    context = SimpleNamespace(
        config=SimpleNamespace(folds=5, uncertainty_penalty=0.0),
        source=SimpleNamespace(
            split=SimpleNamespace(
                development_rows=np.arange(fold.size, dtype=np.int64),
                row_inner_fold=fold,
            )
        ),
    )
    predictions = {
        "held_only": np.full(fold.size, 0.2, dtype=np.float32),
        "stable": np.full(fold.size, 0.8, dtype=np.float32),
    }
    scores = {
        id(predictions["held_only"]): [1.0, 0.4, 0.4, 0.4, 0.4],
        id(predictions["stable"]): [0.0, 0.6, 0.6, 0.6, 0.6],
    }
    original_scores = model9._fold_scores
    original_report = model9._prediction_report
    try:
        model9._fold_scores = lambda _context, value: scores[id(value)]
        model9._prediction_report = lambda _context, _value: {"tested": True}
        selected, report = model9._nested_select(context, predictions)
    finally:
        model9._fold_scores = original_scores
        model9._prediction_report = original_report

    # The held-only candidate wins fold 0 if fold 0 is leaked into selection;
    # excluding fold 0 correctly chooses the stable candidate instead.
    assert report["held_choices"][0]["selected"] == "stable"
    assert np.all(selected[fold == 0] == predictions["stable"][fold == 0])
    for choice in report["held_choices"]:
        assert choice["held_fold"] in range(5)


def test_model9_architecture_feature_counts_are_fixed_and_unique() -> None:
    core_columns = np.arange(0, 800, dtype=np.int32)
    candidate_columns = np.arange(1_000, 2_415, dtype=np.int32)
    source = SimpleNamespace(
        family_columns={"core": core_columns, "candidate": candidate_columns},
        family_names={
            "core": [f"core_{index}" for index in range(core_columns.size)],
            "candidate": [
                f"candidate_{index}" for index in range(candidate_columns.size)
            ],
        },
    )
    context = SimpleNamespace(source=source)

    def fake_columns(_source, family, _fold, contract):
        count = int(contract["count"])
        return (
            _source.family_columns[family][:count].copy(),
            _source.family_names[family][:count].copy(),
        )

    original = model9.model6._columns_for_contract
    model9.model6._columns_for_contract = fake_columns
    try:
        expected = {
            "candidate_all_gbdt": 1_415,
            "candidate_split512_gbdt": 512,
            "core_fisher512_gbdt": 512,
            "joint_fisher768_gbdt": 768,
        }
        for architecture, count in expected.items():
            columns, names = model9.architecture_columns(context, architecture, fold=2)
            assert columns.shape == (count,)
            assert len(names) == count
            assert len(np.unique(columns)) == count
            assert len(set(names)) == count
    finally:
        model9.model6._columns_for_contract = original

    _expect_value_error(model9.architecture_columns, context, "unknown", 0)


def test_model9_meta_features_are_aligned_finite_and_named() -> None:
    probabilities = {
        "zeta": np.asarray([0.0, 0.2, 0.8, 1.0, np.nan]),
        "alpha": np.asarray([0.7, 0.4, 0.6, 0.3, 0.5]),
        "middle": np.asarray([0.4, 0.9, 0.1, 0.5, 0.2]),
    }
    online_time = np.asarray([0, 1, 7, 63, 1_024], dtype=np.int64)
    matrix, names = model9.meta_features(probabilities, online_time)
    assert matrix.shape == (online_time.size, 2 * len(probabilities) + 5)
    assert matrix.dtype == np.float32
    assert matrix.flags.c_contiguous
    assert np.all(np.isfinite(matrix))
    assert len(names) == matrix.shape[1]
    assert len(set(names)) == len(names)
    assert names[:3] == ["logit__alpha", "logit__middle", "logit__zeta"]
    assert np.all(matrix[0, 7:10] == 0.0)  # time zero has zero age interactions

    _expect_value_error(
        model9.meta_features,
        {"a": np.ones(3), "b": np.ones(2)},
        np.arange(3),
    )


def test_model9_pairwise_sampling_is_exact_time_matched_and_deterministic() -> None:
    # Every scoreable time has at least one positive and one negative.  Row 11
    # is deliberately ineligible, so this also checks that the sampler honors
    # its explicit population mask rather than sampling from all input rows.
    target = np.asarray([0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0], dtype=np.uint8)
    time_online = np.asarray([0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 3, 3])
    eligible = np.ones(target.size, dtype=bool)
    eligible[11] = False

    first = pairwise.sample_time_matched_pairs(
        target, time_online, eligible, pair_count=2_000, seed=71_003
    )
    second = pairwise.sample_time_matched_pairs(
        target, time_online, eligible, pair_count=2_000, seed=71_003
    )
    positive, negative, detail = first
    assert np.array_equal(positive, second[0])
    assert np.array_equal(negative, second[1])
    assert detail == second[2]
    assert np.all(time_online[positive] == time_online[negative])
    assert np.all(target[positive] == 1)
    assert np.all(target[negative] == 0)
    assert np.all(eligible[positive]) and np.all(eligible[negative])
    assert detail["scoreable_time_strata"] == 4
    assert detail["pairs"] == 2_000


def test_model9_pairwise_sampling_excludes_the_held_fold() -> None:
    # The same time strata occur in every fold.  A boolean mask is therefore
    # the only thing preventing rows from the scored fold entering meta-fit.
    row_fold = np.repeat(np.arange(5, dtype=np.int8), 8)
    time_online = np.tile(np.repeat(np.arange(4), 2), 5)
    target = np.tile(np.asarray([0, 1] * 4, dtype=np.uint8), 5)
    held_fold = 3
    eligible = row_fold != held_fold
    positive, negative, _ = pairwise.sample_time_matched_pairs(
        target, time_online, eligible, pair_count=4_000, seed=83_117
    )
    assert not np.any(row_fold[positive] == held_fold)
    assert not np.any(row_fold[negative] == held_fold)
    assert set(np.unique(row_fold[np.concatenate((positive, negative))])) == {
        0,
        1,
        2,
        4,
    }


def test_model9_pairwise_fit_and_prediction_are_finite() -> None:
    rng = np.random.default_rng(19_907)
    rows = 240
    columns = 7
    target = np.tile(np.asarray([0, 1], dtype=np.uint8), rows // 2)
    time_online = np.repeat(np.arange(rows // 4), 4)
    # Include informative, noisy, constant, and highly correlated columns to
    # exercise scaling and regularization on a small deterministic matrix.
    signal = (2.0 * target.astype(np.float32) - 1.0)[:, None]
    noise = rng.normal(0.0, 0.7, size=(rows, columns - 1)).astype(np.float32)
    logits = np.column_stack(
        (
            signal[:, 0] + noise[:, 0],
            0.5 * signal[:, 0] + noise[:, 1],
            noise[:, 2:],
            np.ones(rows, dtype=np.float32),
        )
    ).astype(np.float32)
    assert logits.shape == (rows, columns)

    eligible = np.ones(rows, dtype=bool)
    positive, negative, _ = pairwise.sample_time_matched_pairs(
        target, time_online, eligible, pair_count=1_500, seed=6_401
    )
    config = pairwise.make_config(PROJECT)
    scale, coefficient, detail = pairwise.fit_pairwise_model(
        logits, positive, negative, config
    )
    prediction = pairwise.predict_margin(
        logits,
        np.arange(rows, dtype=np.int64),
        scale,
        coefficient,
        chunk_rows=37,
    )
    assert scale.shape == coefficient.shape == (columns,)
    assert prediction.shape == (rows,)
    assert np.all(np.isfinite(scale))
    assert np.all(np.isfinite(coefficient))
    assert np.all(np.isfinite(prediction))
    assert np.all(scale >= config.minimum_pair_scale)
    assert detail["solver_iterations"] >= 1
    assert np.mean(prediction[target == 1]) > np.mean(prediction[target == 0])


def test_model9_pairwise_rejects_outer_holdout_paths_before_loading() -> None:
    # The repository uses a custom no-fixture test runner, so standard-library
    # temporary directories keep this path-safety test self-contained.
    with TemporaryDirectory() as temporary:
        root = Path(temporary).resolve()
        safe = root / "models" / "development" / "oof.npy"
        safe.parent.mkdir(parents=True)
        safe.write_bytes(b"development-only")
        assert pairwise._development_only_path(safe, root) == (
            "models/development/oof.npy"
        )

        prohibited = root / "models" / "outer_holdout" / "prediction.npy"
        prohibited.parent.mkdir(parents=True)
        prohibited.write_bytes(b"must-not-be-opened")
        _expect_value_error(pairwise._development_only_path, prohibited, root)

        outside = root.parent / f"{root.name}-outside-development.npy"
        outside.write_bytes(b"outside")
        try:
            _expect_value_error(pairwise._development_only_path, outside, root)
        finally:
            outside.unlink()


def test_model9_pairwise_public_api_is_present() -> None:
    public_functions = (
        "make_config",
        "config_for_model9_context",
        "load_stack_data",
        "sample_time_matched_pairs",
        "fit_pairwise_model",
        "predict_margin",
        "run_pairwise_stack",
        "run_for_model9_context",
        "run",
        "status",
        "main",
    )
    for name in public_functions:
        assert callable(getattr(pairwise, name, None)), name
    assert pairwise.PAIRWISE_STACK_VERSION.startswith("model9-pairwise-")
    assert len(pairwise.DEPLOYED_FEATURE_MODELS) == 3


def test_model9_notebook_is_valid_when_present() -> None:
    notebook_path = PROJECT / "models/model9/model9.ipynb"
    if not notebook_path.exists():
        return
    notebook = nbformat.read(notebook_path, as_version=4)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert len(code_cells) >= 4
    source = "\n".join(cell.source for cell in code_cells)
    assert "run_scale_cv" in source
    assert "run_promoted_cv" in source
    assert "run_meta_stack" in source or "run_all" in source
    for index, cell in enumerate(code_cells):
        compile(cell.source, f"model9.ipynb cell {index}", "exec")
