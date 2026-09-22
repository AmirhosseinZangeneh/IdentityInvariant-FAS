"""Legacy MSU indexing: unnormalized IDs and filename-stem groups.

Use msu_protocol and msu_preprocessing for canonical, locked provenance.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

from ..constants import ATTACK_LABEL, BONA_FIDE_LABEL
from .manifest import FASSample


_CLIENT_PATTERN = re.compile(r"client(?P<subject>\d{3})", re.IGNORECASE)


def read_subject_list(path: str | Path) -> set[str]:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    return {token.strip() for token in re.split(r"[\s,]+", text) if token.strip()}


def index_msu_videos(root: str | Path) -> list[FASSample]:
    """Legacy compatibility API; not the canonical preprocessing path."""
    warnings.warn(
        "Legacy MSU loader uses unnormalized subject IDs and video stems. "
        "Use load_msu_protocol and msu_preprocessing for canonical provenance.",
        FutureWarning, stacklevel=2,
    )
    root = Path(root)
    train_subjects = read_subject_list(root / "train_sub_list.txt")
    test_subjects = read_subject_list(root / "test_sub_list.txt")

    overlap = train_subjects & test_subjects
    if overlap:
        raise ValueError(f"MSU train/test subject overlap detected: {sorted(overlap)}")

    samples: list[FASSample] = []
    for class_name, label in (("real", BONA_FIDE_LABEL), ("attack", ATTACK_LABEL)):
        directory = root / "scene01" / class_name
        for video_path in sorted(directory.glob("*")):
            if video_path.suffix.lower() not in {".mp4", ".mov"}:
                continue

            match = _CLIENT_PATTERN.search(video_path.stem)
            if match is None:
                raise ValueError(f"Cannot parse client ID from: {video_path.name}")
            subject = match.group("subject")

            if subject in train_subjects:
                split = "train"
            elif subject in test_subjects:
                split = "test"
            else:
                raise ValueError(f"Subject {subject} is absent from official MSU split lists.")

            tokens = video_path.stem.split("_")
            camera = "android" if "android" in tokens else ("laptop" if "laptop" in tokens else "")
            attack_type = ""
            if label == ATTACK_LABEL:
                for candidate in ("ipad_video", "iphone_video", "printed_photo"):
                    if candidate in video_path.stem:
                        attack_type = candidate
                        break

            samples.append(
                FASSample(
                    path=str(video_path),
                    label=label,
                    subject=subject,
                    dataset="MSU-MFSD",
                    split=split,
                    video_id=video_path.stem,
                    attack_type=attack_type,
                    camera=camera,
                )
            )
    return samples
