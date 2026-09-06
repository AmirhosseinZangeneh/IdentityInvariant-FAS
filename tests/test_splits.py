from identity_invariant_fas.data.manifest import FASSample
from identity_invariant_fas.data.splits import subject_kfold_split, subject_train_val_split


def make_samples():
    samples = []
    for subject in [f"{index:04d}" for index in range(10)]:
        samples.append(FASSample("x.jpg", 0, subject, "NUAA"))
        samples.append(FASSample("y.jpg", 1, subject, "NUAA"))
    return samples


def test_outer_folds_are_subject_disjoint():
    for fold in subject_kfold_split(make_samples(), n_folds=5, seed=42):
        assert fold["train_subjects"].isdisjoint(fold["test_subjects"])


def test_validation_is_subject_disjoint():
    train, val = subject_train_val_split(make_samples(), val_ratio=0.2, seed=42)
    train_subjects = {sample.subject for sample in train}
    val_subjects = {sample.subject for sample in val}
    assert train_subjects.isdisjoint(val_subjects)
