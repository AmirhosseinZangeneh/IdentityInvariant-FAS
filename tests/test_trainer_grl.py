"""Synthetic schedule tests: no datasets, model fitting or experiment execution."""
import copy
import importlib.util
import math
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch
import yaml
from torch import nn

from identity_invariant_fas.training.engine import GRLConfiguration, Trainer
from identity_invariant_fas.training.losses import LossOutput


def load_experiment(name):
    path = Path(__file__).resolve().parents[1] / "experiments" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"grl_test_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build_optimizer_and_trainer = load_experiment("_common").build_optimizer_and_trainer
build_sweep_configs = load_experiment("grl_sweep").build_sweep_configs


class StubGRL(nn.Module):
    def __init__(self, target=0.05):
        super().__init__()
        self.parameter = nn.Parameter(torch.zeros(1))
        self.grl_lambda = target

    def set_grl_lambda(self, value):
        self.grl_lambda = value

    def forward(self, images):
        return torch.zeros(len(images), 2)


def trainer(target=0.05, **kwargs):
    return Trainer(StubGRL(target), Mock(), "cpu", grl_target_lambda=target,
                   **({"total_epochs": 30, "warmup_epochs": 5} | kwargs))


@pytest.mark.parametrize("target", [0.01, 0.05, 0.10])
def test_fixed_history_and_mutable_model_regression(target):
    t = trainer(target)
    history = []
    for epoch in range(1, 31):
        # Old code ignored the configured target and installed the same schedule.
        t.model.grl_lambda = 123.0
        t._update_grl_lambda(epoch)
        assert t.model.grl_lambda == t.current_grl_lambda
        history.append(t.current_grl_lambda)
    assert history == [0.0] * 5 + [target] * 25
    with pytest.raises(FrozenInstanceError):
        t.grl_config.target = 9


def test_public_factory_uses_actual_epoch_budget():
    config = {"model": {"grl_lambda": 0.01},
              "training": {"epochs": 7, "warmup_epochs": 5, "grl_schedule": "fixed"}}
    _, t = build_optimizer_and_trainer(StubGRL(), config, "cpu")
    assert t.total_epochs == 7
    config["model"]["grl_lambda"] = 9
    assert [t.grl_config.coefficient(e) for e in range(1, 8)] == [0] * 5 + [0.01] * 2


def test_epoch_budget_and_grl_target_cannot_silently_default():
    with pytest.raises(TypeError):
        Trainer(StubGRL(), Mock(), "cpu", grl_target_lambda=0.05)
    with pytest.raises(ValueError, match="explicit"):
        Trainer(StubGRL(), Mock(), "cpu", total_epochs=30)
    with pytest.raises(KeyError):
        build_optimizer_and_trainer(StubGRL(), {"model": {}, "training": {}}, "cpu")


@pytest.mark.parametrize("mode", ["dann", "progressive"])
def test_progressive_is_scaled_and_uses_active_epoch_budget(mode):
    small = trainer(0.01, grl_schedule=mode, total_epochs=7)
    large = trainer(0.10, grl_schedule=mode, total_epochs=7)
    a = [small.grl_config.coefficient(e) for e in range(1, 8)]
    b = [large.grl_config.coefficient(e) for e in range(1, 8)]
    assert a[:5] == [0] * 5
    assert 0 < a[5] < a[6] < 0.01
    assert a[5] == pytest.approx(0.01 * (2 / (1 + math.exp(-5)) - 1))
    assert b == pytest.approx([10 * x for x in a])


@pytest.mark.parametrize("target", [-1, float("nan"), float("inf"), -float("inf")])
def test_invalid_targets(target):
    with pytest.raises(ValueError, match="target"):
        trainer(target)


@pytest.mark.parametrize("mode", ["unknown", "", "Fixed", None])
def test_invalid_modes(mode):
    with pytest.raises(ValueError, match="schedule"):
        trainer(grl_schedule=mode)


@pytest.mark.parametrize("kwargs", [
    {"total_epochs": 0}, {"total_epochs": -1}, {"total_epochs": 1.5},
    {"total_epochs": True}, {"warmup_epochs": -1}, {"warmup_epochs": 1.5},
    {"warmup_epochs": True}, {"warmup_epochs": 30}, {"warmup_epochs": 31},
])
def test_invalid_epoch_configuration(kwargs):
    with pytest.raises(ValueError):
        trainer(**kwargs)


@pytest.mark.parametrize("epoch", [0, -1, 31, 1.5, True])
def test_public_epoch_range(epoch):
    with pytest.raises(ValueError, match="epoch"):
        trainer()._update_grl_lambda(epoch)


def test_no_warmup_single_epoch_and_zero_target():
    assert trainer(warmup_epochs=0, total_epochs=1).grl_config.coefficient(1) == 0.05
    assert trainer(0, grl_schedule="dann").grl_config.coefficient(30) == 0


def test_non_grl_model_unaffected():
    model = nn.Linear(1, 2)
    before = copy.deepcopy(model.state_dict())
    t = Trainer(model, Mock(), "cpu", total_epochs=3, warmup_epochs=5)
    for epoch in range(1, 4):
        t._update_grl_lambda(epoch)
        assert t.current_grl_lambda == 0
    assert t.grl_config is None
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())


@pytest.mark.parametrize("epoch,expected", [(5, 0.0), (6, 0.05)])
def test_epoch_result_reports_effective_value_without_optimization(epoch, expected):
    t = trainer()
    # Exercise aggregation only. Neither backward nor optimizer performs work.
    loss = Mock()
    loss.item.return_value = 1.0
    t.loss_fn = Mock(return_value=LossOutput(total=loss, spoof=loss, subject=None))
    before = t.model.parameter.detach().clone()
    result = t.train_epoch([{"image": torch.zeros(2, 1), "label": torch.zeros(2, dtype=torch.long)}], epoch)
    assert result.grl_lambda == expected == t.model.grl_lambda
    assert torch.equal(before, t.model.parameter)
    t.optimizer.step.assert_called_once()
    loss.backward.assert_called_once()


def test_real_sweep_config_varies_only_target_and_output_and_has_distinct_histories():
    path = Path(__file__).resolve().parents[1] / "configs/grl_sweep.yaml"
    source = yaml.safe_load(path.read_text())
    original = copy.deepcopy(source)
    runs = build_sweep_configs(source)
    assert source == original
    assert [r["model"]["grl_lambda"] for r in runs] == [0.01, 0.05, 0.10]
    normalized, histories = [], []
    for run in runs:
        _, t = build_optimizer_and_trainer(StubGRL(), run, "cpu")
        assert t.grl_config.schedule == "fixed"
        histories.append(tuple(t.grl_config.coefficient(e) for e in range(1, t.total_epochs + 1)))
        row = copy.deepcopy(run)
        del row["model"]["grl_lambda"]
        del row["training"]["output_dir"]
        normalized.append(row)
    assert normalized[0] == normalized[1] == normalized[2]
    assert len(set(histories)) == 3


def test_sweep_rejects_duplicate_values():
    with pytest.raises(ValueError, match="distinct"):
        build_sweep_configs({"grl_values": [0.05, 0.05]})
