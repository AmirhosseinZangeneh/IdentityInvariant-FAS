"""Centralized image transforms for fair model comparison."""

from __future__ import annotations

from typing import Callable, Any

from torchvision import transforms


def build_train_transform(
    image_size: int = 160,
    horizontal_flip: bool = False,
) -> Callable[[Any], Any]:

    operations: list[Callable[[Any], Any]] = [
        transforms.Resize(
            (image_size, image_size)
        )
    ]

    if horizontal_flip:

        operations.append(
            transforms.RandomHorizontalFlip()
        )

    operations.append(
        transforms.ToTensor()
    )

    return transforms.Compose(
        operations
    )


def build_eval_transform(
    image_size: int = 160,
) -> Callable[[Any], Any]:

    return transforms.Compose(
        [
            transforms.Resize(
                (image_size, image_size)
            ),
            transforms.ToTensor(),
        ]
    )