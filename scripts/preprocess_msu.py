"""Extract uniformly sampled face crops from original MSU-MFSD videos."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from identity_invariant_fas.data.manifest import FASSample, write_manifest
from identity_invariant_fas.data.msu import index_msu_videos


def read_face_file(path: Path) -> dict[int, tuple[int, int, int, int]]:
    """Read PittPatt frame-level face rectangles: frame,left,top,right,bottom,..."""
    faces: dict[int, tuple[int, int, int, int]] = {}
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            parts = [token for token in raw.replace(",", " ").split() if token]
            if len(parts) < 5:
                continue
            try:
                frame = int(float(parts[0]))
                left, top, right, bottom = map(lambda x: int(float(x)), parts[1:5])
            except ValueError:
                continue
            faces[frame] = (left, top, right, bottom)
    return faces


def expand_box(box, width: int, height: int, margin: float):
    left, top, right, bottom = box
    box_width = right - left
    box_height = bottom - top
    x_margin = int(round(box_width * margin))
    y_margin = int(round(box_height * margin))
    return (
        max(0, left - x_margin),
        max(0, top - y_margin),
        min(width, right + x_margin),
        min(height, bottom + y_margin),
    )


def process_video(sample, output_root: Path, frames_per_video: int, margin: float):
    video_path = Path(sample.path)
    face_path = video_path.with_suffix(".face")
    if not face_path.exists():
        raise FileNotFoundError(f"Missing face annotation: {face_path}")

    faces = read_face_file(face_path)
    valid_frames = sorted(faces)
    if not valid_frames:
        return []

    count = min(frames_per_video, len(valid_frames))
    positions = np.linspace(0, len(valid_frames) - 1, count, dtype=int)
    selected_frames = {valid_frames[position] for position in positions}

    capture = cv2.VideoCapture(str(video_path))
    if hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO"):
        capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)

    output_samples = []
    frame_index = 0

    while True:
        success, frame = capture.read()
        if not success:
            break

        if frame_index in selected_frames:
            height, width = frame.shape[:2]
            left, top, right, bottom = expand_box(
                faces[frame_index], width, height, margin
            )
            crop = frame[top:bottom, left:right]
            if crop.size:
                class_name = "real" if sample.label == 0 else "attack"
                directory = output_root / sample.split / class_name / sample.subject / sample.video_id
                directory.mkdir(parents=True, exist_ok=True)
                image_path = directory / f"frame_{frame_index:06d}.jpg"
                cv2.imwrite(str(image_path), crop)

                output_samples.append(
                    FASSample(
                        path=str(image_path),
                        label=sample.label,
                        subject=sample.subject,
                        dataset=sample.dataset,
                        split=sample.split,
                        video_id=sample.video_id,
                        attack_type=sample.attack_type,
                        camera=sample.camera,
                    )
                )
        frame_index += 1

    capture.release()
    return output_samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Original MSU-MFSD root.")
    parser.add_argument("--output-root", default="data_processed/MSU-MFSD")
    parser.add_argument("--frames-per-video", type=int, default=30)
    parser.add_argument("--margin", type=float, default=0.0)
    args = parser.parse_args()

    videos = index_msu_videos(args.root)
    output_root = Path(args.output_root)
    extracted = []

    for index, sample in enumerate(videos, start=1):
        extracted.extend(
            process_video(sample, output_root, args.frames_per_video, args.margin)
        )
        print(f"[{index}/{len(videos)}] {sample.video_id}")

    write_manifest(extracted, output_root / "manifest.csv")
    print(f"Extracted {len(extracted)} face crops.")


if __name__ == "__main__":
    main()
