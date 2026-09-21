"""Protocol invariants and evaluator checks using synthetic observations only."""

import importlib.util
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from identity_invariant_fas.evaluation import identity_probe as probe
from identity_invariant_fas.evaluation.identity_leakage import cross_validated_identity_probe
from identity_invariant_fas.evaluation.identity_probe import (
    ProbeObservation,
    ProbeSplit,
    evaluate_closed_set_identity_probe,
    make_closed_set_probe_splits,
    validate_closed_set_probe_split,
)


def observations(n_groups=4):
    return [
        ProbeObservation(
            sample_id=f"{identity}/video-{group}/frame-{frame}",
            identity_id=identity,
            group_id=f"{identity}/video-{group}",
            session_id=f"{identity}/session-{group // 2}",
        )
        for identity in ("person-a", "person-b")
        for group in range(n_groups)
        for frame in range(3)
    ]


def test_video_disjoint_closed_set_and_no_frame_leakage():
    rows = observations()
    splits = make_closed_set_probe_splits(rows, n_splits=4, seed=19)
    evaluated = []
    for split in splits:
        train, evaluation = set(split.train_indices), set(split.eval_indices)
        assert train.isdisjoint(evaluation)
        assert train | evaluation == set(range(len(rows)))
        assert {rows[i].identity_id for i in train} == {rows[i].identity_id for i in evaluation}
        assert {rows[i].group_id for i in train}.isdisjoint({rows[i].group_id for i in evaluation})
        for group in {row.group_id for row in rows}:
            frames = {i for i, row in enumerate(rows) if row.group_id == group}
            assert frames <= train or frames <= evaluation
        validate_closed_set_probe_split(rows, split)
        evaluated.extend(evaluation)
    assert sorted(evaluated) == list(range(len(rows)))


def test_determinism_seed_control_and_row_reordering():
    rows = observations(n_groups=8)
    first = make_closed_set_probe_splits(rows, n_splits=4, seed=19)
    assert first == make_closed_set_probe_splits(rows, n_splits=4, seed=19)
    assert first != make_closed_set_probe_splits(rows, n_splits=4, seed=20)
    reordered = list(reversed(rows))
    second = make_closed_set_probe_splits(reordered, n_splits=4, seed=19)
    assert [{rows[i].sample_id for i in s.eval_indices} for s in first] == [
        {reordered[i].sample_id for i in s.eval_indices} for s in second
    ]


def test_group_overlap_rejected():
    rows = observations()
    split = ProbeSplit(tuple(range(0, len(rows), 2)), tuple(range(1, len(rows), 2)))
    with pytest.raises(ValueError, match="group_id overlap"):
        validate_closed_set_probe_split(rows, split)


def test_unseen_evaluation_identity_rejected():
    rows = observations()
    midpoint = len(rows) // 2
    split = ProbeSplit(tuple(range(midpoint)), tuple(range(midpoint, len(rows))))
    with pytest.raises(ValueError, match="Unseen evaluation identities"):
        validate_closed_set_probe_split(rows, split)


def test_training_identity_missing_from_evaluation_rejected():
    rows = observations(n_groups=2)
    split = ProbeSplit(tuple(range(3, len(rows))), (0, 1, 2))
    with pytest.raises(ValueError, match="Training identities absent from evaluation"):
        validate_closed_set_probe_split(rows, split)


def test_insufficient_independent_groups_despite_many_frames():
    rows = [replace(row, group_id=row.identity_id) for row in observations()]
    with pytest.raises(ValueError, match="separable group_id"):
        make_closed_set_probe_splits(rows, n_splits=2, seed=1)


def test_session_disjoint_splits():
    rows = observations()
    for split in make_closed_set_probe_splits(rows, n_splits=2, seed=7, session_disjoint=True):
        train_sessions = {rows[i].session_id for i in split.train_indices}
        eval_sessions = {rows[i].session_id for i in split.eval_indices}
        assert train_sessions.isdisjoint(eval_sessions)
        validate_closed_set_probe_split(rows, split, session_disjoint=True)


def test_shared_sessions_across_identities_stay_together():
    rows = [replace(row, session_id=row.session_id.split("/")[-1]) for row in observations(8)]
    splits = make_closed_set_probe_splits(rows, n_splits=4, seed=7, session_disjoint=True)
    assert len(splits) == 4
    for split in splits:
        validate_closed_set_probe_split(rows, split, session_disjoint=True)


def test_shared_groups_across_identities_stay_together():
    rows = [replace(row, group_id=row.group_id.split("/")[-1]) for row in observations()]
    for split in make_closed_set_probe_splits(rows, n_splits=4, seed=7):
        validate_closed_set_probe_split(rows, split)


def test_session_overlap_only_rejected_when_requested():
    rows = observations(n_groups=2)
    train = tuple(i for i, row in enumerate(rows) if row.group_id.endswith("0"))
    evaluation = tuple(i for i, row in enumerate(rows) if row.group_id.endswith("1"))
    split = ProbeSplit(train, evaluation)
    validate_closed_set_probe_split(rows, split)
    with pytest.raises(ValueError, match="session_id overlap"):
        validate_closed_set_probe_split(rows, split, session_disjoint=True)


@pytest.mark.parametrize("mode", ["missing", "partial", "single_session", "inconsistent"])
def test_insufficient_session_metadata(mode):
    rows = observations()
    if mode == "missing":
        rows = [replace(row, session_id=None) for row in rows]
    elif mode == "partial":
        rows[0] = replace(rows[0], session_id=None)
    elif mode == "single_session":
        rows = [replace(row, session_id="one-session") for row in rows]
    else:
        rows[0] = replace(rows[0], session_id="other-session")
    with pytest.raises(ValueError, match="session"):
        make_closed_set_probe_splits(rows, seed=1, n_splits=2, session_disjoint=True)


def test_duplicate_sample_id_rejected_even_if_group_is_changed():
    rows = observations()
    rows.append(replace(rows[0], group_id="another-group"))
    with pytest.raises(ValueError, match="Duplicate sample_id"):
        make_closed_set_probe_splits(rows, seed=1, n_splits=2)


@pytest.mark.parametrize("mode", ["overlap", "missing", "duplicate", "out_of_bounds", "noninteger", "empty"])
def test_invalid_sample_partitions(mode):
    rows = observations()
    valid = make_closed_set_probe_splits(rows, seed=1, n_splits=2)[0]
    train, evaluation = valid.train_indices, valid.eval_indices
    if mode == "overlap":
        train += (evaluation[0],)
    elif mode == "missing":
        train = train[1:]
    elif mode == "duplicate":
        train += (train[0],)
    elif mode == "out_of_bounds":
        train += (len(rows),)
    elif mode == "noninteger":
        train = (0.0,) + train[1:]
    else:
        train = ()
    with pytest.raises(ValueError):
        validate_closed_set_probe_split(rows, ProbeSplit(train, evaluation))


@pytest.mark.parametrize("field,value", [
    ("group_id", ""), ("identity_id", " "), ("sample_id", " padded"), ("session_id", ""),
])
def test_invalid_metadata(field, value):
    rows = observations()
    rows[0] = replace(rows[0], **{field: value})
    with pytest.raises(ValueError, match=field):
        make_closed_set_probe_splits(rows, seed=1, n_splits=2)


@pytest.mark.parametrize("kwargs", [{"seed": None}, {"seed": -1}, {"seed": True}, {"seed": 1, "n_splits": 1}])
def test_explicit_valid_seed_and_fold_count_required(kwargs):
    with pytest.raises(ValueError):
        make_closed_set_probe_splits(observations(), **kwargs)


def synthetic_features(rows):
    return np.asarray([[float(row.identity_id == "person-a"), i / len(rows)] for i, row in enumerate(rows)])


def test_evaluation_is_deterministic_and_reports_protocol_membership():
    rows = observations()
    features = synthetic_features(rows)
    result = evaluate_closed_set_identity_probe(features, rows, seed=3, n_splits=2)
    assert result == evaluate_closed_set_identity_probe(features, rows, seed=3, n_splits=2)
    assert result["n_identities"] == 2
    assert result["uniform_chance_accuracy"] == 0.5
    assert result["grouping"] == "group"
    for fold in result["folds"]:
        assert 0 <= fold["accuracy"] <= 1
        assert 0 <= fold["macro_f1"] <= 1
        assert set(fold["train_sample_ids"]).isdisjoint(fold["eval_sample_ids"])
        assert set(fold["train_group_ids"]).isdisjoint(fold["eval_group_ids"])
        assert fold["train_identity_counts"].keys() == fold["eval_identity_counts"].keys()


def test_scaler_is_fit_only_on_probe_training_features(monkeypatch):
    rows = observations()
    features = synthetic_features(rows)
    split = make_closed_set_probe_splits(rows, seed=3, n_splits=2)[0]
    observed = []
    original = probe.StandardScaler.fit

    def record_fit(self, values, *args, **kwargs):
        observed.append(values.copy())
        return original(self, values, *args, **kwargs)

    monkeypatch.setattr(probe.StandardScaler, "fit", record_fit)
    result = evaluate_closed_set_identity_probe(features, rows, seed=3, splits=[split])
    assert len(observed) == 1
    np.testing.assert_array_equal(observed[0], features[list(split.train_indices)])
    assert result["accuracy_std"] is None


def test_all_protocols_validated_before_any_fit(monkeypatch):
    rows = observations()
    valid = make_closed_set_probe_splits(rows, seed=3, n_splits=2)[0]
    invalid = ProbeSplit(tuple(range(0, len(rows), 2)), tuple(range(1, len(rows), 2)))

    def forbid_fit(*args, **kwargs):
        pytest.fail("A classifier was fitted before all folds were validated")

    monkeypatch.setattr(probe.Pipeline, "fit", forbid_fit)
    with pytest.raises(ValueError, match="group_id overlap"):
        evaluate_closed_set_identity_probe(synthetic_features(rows), rows, seed=3, splits=[valid, invalid])


@pytest.mark.parametrize("mode", ["length", "rank", "nan", "zero_width"])
def test_invalid_features(mode):
    rows = observations()
    features = synthetic_features(rows)
    if mode == "length":
        features = features[:-1]
    elif mode == "rank":
        features = features[:, 0]
    elif mode == "nan":
        features[0, 0] = np.nan
    else:
        features = features[:, :0]
    with pytest.raises(ValueError, match="features"):
        evaluate_closed_set_identity_probe(features, rows, seed=1, n_splits=2)


def test_legacy_api_warns_and_preserves_result_schema():
    rows = observations()
    with pytest.warns(FutureWarning, match="legacy/exploratory"):
        result = cross_validated_identity_probe(
            synthetic_features(rows), [r.identity_id for r in rows], n_splits=2
        )
    assert result["n_subjects"] == 2
    assert len(result["folds"]) == 2
    assert "accuracy_mean" in result


def test_nuaa_cli_requires_explicit_legacy_opt_in(monkeypatch, capsys):
    path = Path(__file__).parents[1] / "experiments/identity_leakage_nuaa.py"
    spec = importlib.util.spec_from_file_location("legacy_nuaa_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr("sys.argv", [str(path), "--model", "paper_ecnn", "--checkpoint", "unused", "--output", "unused"])
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 2
    assert "--allow-legacy-sample-cv" in capsys.readouterr().err
