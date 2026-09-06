from pathlib import Path
from collections import Counter


DATA_ROOT = Path("datasets/NUAA")


def parse_protocol(file_path):
    samples = []

    with open(file_path, "r") as f:
        for line in f:
            if not line.strip():
                continue

            parts = line.split()

            image_path = parts[0]
            subject = image_path.split("\\")[0]

            samples.append(
                {
                    "path": image_path,
                    "subject": subject
                }
            )

    return samples


def summarize(name, samples):

    subjects = sorted(
        set(
            item["subject"]
            for item in samples
        )
    )

    print(f"\n{name}")
    print("-" * 40)

    print(
        "Samples:",
        len(samples)
    )

    print(
        "Subjects:",
        len(subjects)
    )

    print(
        subjects
    )

    return set(subjects)


def main():

    files = {

        "client_train":
            DATA_ROOT / "client_train_face.txt",

        "client_test":
            DATA_ROOT / "client_test_face.txt",

        "imposter_train":
            DATA_ROOT / "imposter_train_face.txt",

        "imposter_test":
            DATA_ROOT / "imposter_test_face.txt",
    }


    data = {}

    for name, path in files.items():

        data[name] = parse_protocol(path)


    client_train_subjects = summarize(
        "Client Train",
        data["client_train"]
    )

    client_test_subjects = summarize(
        "Client Test",
        data["client_test"]
    )

    imposter_train_subjects = summarize(
        "Imposter Train",
        data["imposter_train"]
    )

    imposter_test_subjects = summarize(
        "Imposter Test",
        data["imposter_test"]
    )


    print("\nIdentity Overlap Analysis")
    print("=" * 40)


    print(
        "Client train/test overlap:",
        client_train_subjects &
        client_test_subjects
    )


    print(
        "Imposter train/test overlap:",
        imposter_train_subjects &
        imposter_test_subjects
    )


    all_train = (
        client_train_subjects |
        imposter_train_subjects
    )

    all_test = (
        client_test_subjects |
        imposter_test_subjects
    )


    print(
        "Overall train/test overlap:",
        all_train & all_test
    )


if __name__ == "__main__":
    main()