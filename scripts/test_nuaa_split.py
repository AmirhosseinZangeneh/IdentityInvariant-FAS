from identity_invariant_fas.data.nuaa import load_nuaa_samples
from identity_invariant_fas.data.splits import (
    assign_subject_ids,
    subject_kfold_split,
)


samples = load_nuaa_samples(
    "datasets/NUAA",
    partition="all",
)

samples = assign_subject_ids(samples)


print("Total samples:", len(samples))

subjects = sorted(
    {
        sample.subject
        for sample in samples
    }
)

print("Total subjects:", len(subjects))
print("Subjects:", subjects)


folds = subject_kfold_split(
    samples,
    n_folds=5,
    seed=42,
)


for fold in folds:

    train_subjects = fold["train_subjects"]
    test_subjects = fold["test_subjects"]

    overlap = (
        train_subjects &
        test_subjects
    )

    print("\nFold:", fold["fold"])
    print(
        "Train subjects:",
        sorted(train_subjects)
    )
    print(
        "Test subjects:",
        sorted(test_subjects)
    )
    print(
        "Overlap:",
        overlap
    )

    print(
        "Train samples:",
        len(fold["train"])
    )

    print(
        "Test samples:",
        len(fold["test"])
    )