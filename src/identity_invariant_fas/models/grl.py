"""Gradient reversal for adversarial representation learning."""

from __future__ import annotations

import math
import torch
from torch import Tensor, nn


class _GradientScaleFunction(torch.autograd.Function):
    """Identity forward, explicitly signed encoder-gradient scale backward."""

    @staticmethod
    def forward(ctx, x: Tensor, coefficient: float) -> Tensor:
        ctx.coefficient = float(coefficient)
        return x.view_as(x)

    @staticmethod
    def backward(
        ctx,
        *grad_outputs: Tensor,
    ) -> tuple[Tensor, None]:

        grad_output = grad_outputs[0]

        return (
            ctx.coefficient * grad_output,
            None,
        )


class GradientScaleLayer(nn.Module):
    """Control only gradients upstream of this layer, including zero or positive."""

    def __init__(self, scale: float = 0.0) -> None:
        super().__init__()
        self.set_scale(scale)

    def set_scale(self, scale: float) -> None:
        if not math.isfinite(scale):
            raise ValueError("gradient scale must be finite")
        self.scale = float(scale)

    def forward(self, x: Tensor) -> Tensor:
        return _GradientScaleFunction.apply(x, self.scale)


class GradientReversalLayer(nn.Module):
    """Reverse incoming gradients by ``-coefficient`` during backpropagation."""

    def __init__(self, coefficient: float = 1.0, *, lambda_: float | None = None) -> None:
        super().__init__()
        if lambda_ is not None:
            coefficient = lambda_
        if coefficient < 0:
            raise ValueError("GRL coefficient must be non-negative.")
        self.coefficient = float(coefficient)

    @property
    def lambda_(self) -> float:
        """Compatibility alias for earlier checkpoints and experiment code."""
        return self.coefficient

    @lambda_.setter
    def lambda_(self, value: float) -> None:
        self.set_coefficient(value)

    def set_coefficient(self, coefficient: float) -> None:
        if coefficient < 0:
            raise ValueError("GRL coefficient must be non-negative.")
        self.coefficient = float(coefficient)

    def forward(self, x: Tensor) -> Tensor:
        return _GradientScaleFunction.apply(x, -self.coefficient)
