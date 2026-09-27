"""Synthetic forward/autograd checks only: no fitting, datasets or experiment runs."""

import copy
import importlib.util
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch
from torch import nn
import torch.nn.functional as F
import yaml

from identity_invariant_fas.models import ControlledIdentityECNN, PaperECNNClassifier
from identity_invariant_fas.models.controlled_identity_ecnn import IDENTITY_MODES
from identity_invariant_fas.training.engine import GRLConfiguration, Trainer
from identity_invariant_fas.training.losses import LossOutput, MultiTaskFASLoss


def load_experiment(name):
    path = Path(__file__).resolve().parents[1] / "experiments" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"controlled_test_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


common = load_experiment("_common")
build_configs = load_experiment("controlled_identity_ablation").build_ablation_configs


class TinyEncoder(nn.Module):
    feature_dim = 4

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 4)

    def forward(self, x):
        return self.linear(x).tanh()


@pytest.fixture
def tiny(monkeypatch):
    monkeypatch.setattr(
        "identity_invariant_fas.models.controlled_identity_ecnn.PaperECNN", TinyEncoder
    )


def matched_models():
    models = []
    for mode in IDENTITY_MODES:
        torch.manual_seed(19)
        models.append(ControlledIdentityECNN(3, identity_mode=mode).double())
    return models


def make_trainer(model, **overrides):
    options = dict(total_epochs=7, warmup_epochs=5, identity_target_lambda=0.05,
                   identity_mode=model.identity_mode, subject_weight=0.1)
    return Trainer(model, Mock(), "cpu", **(options | overrides))


X = torch.tensor([[0.2, -0.7, 1.1], [1.0, 0.3, -0.5], [-0.2, 0.9, 0.1]], dtype=torch.double)
Y = torch.tensor([0, 1, 0])
SUBJECTS = torch.tensor([0, 1, 2])


def flattened(grads):
    return torch.cat([g.flatten() for g in grads])


@pytest.mark.parametrize("epoch,scale", [(1, 0), (5, 0), (6, 0.05), (7, 0.05)])
def test_actual_identity_encoder_and_head_gradients(tiny, epoch, scale):
    _, positive, adversarial = matched_models()
    ordinary = copy.deepcopy(positive)
    ordinary.set_identity_encoder_scale(1.0)
    params = lambda m: list(m.encoder.parameters()) + list(m.subject_classifier.parameters())
    n_encoder = len(list(ordinary.encoder.parameters()))
    # g includes the same beta as each arm; the only difference must be routing.
    g = torch.autograd.grad(0.1 * F.cross_entropy(ordinary(X)[1], SUBJECTS), params(ordinary))
    assert flattened(g[:n_encoder]).norm() > 0
    assert flattened(g[n_encoder:]).norm() > 0
    results = []
    for model, sign in [(positive, 1), (adversarial, -1)]:
        make_trainer(model)._update_grl_lambda(epoch)
        spoof, subject = model(X)
        loss = MultiTaskFASLoss(0.1, identity_mode=model.identity_mode)(spoof, Y, subject, SUBJECTS)
        grads = torch.autograd.grad(0.1 * loss.subject, params(model))
        for actual, reference in zip(grads[:n_encoder], g[:n_encoder]):
            torch.testing.assert_close(actual, sign * scale * reference, rtol=1e-12, atol=1e-14)
        for actual, reference in zip(grads[n_encoder:], g[n_encoder:]):
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        results.append(flattened(grads[:n_encoder]))
    torch.testing.assert_close(results[0], -results[1], rtol=0, atol=0)


def test_spoof_only_ignores_labels_and_has_no_subject_optimization(tiny):
    model = matched_models()[0]
    spoof, subject = model(X)
    assert subject is None
    loss_fn = MultiTaskFASLoss(0.1, identity_mode="spoof_only")
    # Even invalid subject labels do not enter the explicitly disabled objective.
    loss = loss_fn(spoof, Y, torch.ones(3, 3), torch.tensor([-99, 100, 7]))
    assert loss.subject is None and loss.total is loss.spoof
    grads = torch.autograd.grad(loss.total, list(model.parameters()), allow_unused=True)
    by_id = {id(p): g for p, g in zip(model.parameters(), grads)}
    assert all(by_id[id(p)] is None for p in model.subject_classifier.parameters())
    with pytest.raises(ValueError, match="contradicts"):
        loss_fn(spoof, Y, subject_loss_enabled=True)


def test_spoof_only_subject_head_unchanged_after_real_adamw_step(tiny):
    torch.manual_seed(19)
    model = ControlledIdentityECNN(3, identity_mode="spoof_only").double()
    subject_before = {
        name: parameter.detach().clone()
        for name, parameter in model.subject_classifier.named_parameters()
    }
    spoof_before = model.spoof_classifier.weight.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.1)
    optimizer_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
    assert all(id(p) in optimizer_ids for p in model.subject_classifier.parameters())

    optimizer.zero_grad(set_to_none=True)
    spoof, subject = model(X)
    loss = MultiTaskFASLoss(0.1, identity_mode="spoof_only")(spoof, Y, subject, SUBJECTS)
    assert loss.subject is None and loss.total is loss.spoof
    loss.total.backward()
    assert all(p.grad is None for p in model.subject_classifier.parameters())
    assert torch.count_nonzero(model.spoof_classifier.weight.grad) > 0
    optimizer.step()

    for name, parameter in model.subject_classifier.named_parameters():
        assert parameter.grad is None
        assert torch.equal(parameter.detach(), subject_before[name])
    assert not torch.equal(model.spoof_classifier.weight.detach(), spoof_before)


def test_spoof_forward_and_spoof_gradients_identical(tiny):
    values, gradients = [], []
    for model in matched_models():
        make_trainer(model)._update_grl_lambda(6)
        spoof, _, features = model(X, return_feature=True)
        values.append((spoof, features))
        params = list(model.encoder.parameters()) + list(model.spoof_classifier.parameters())
        gradients.append(flattened(torch.autograd.grad(F.cross_entropy(spoof, Y), params)))
    for i in (1, 2):
        for actual, expected in zip(values[i], values[0]):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(gradients[i], gradients[0], rtol=0, atol=0)


def test_real_architecture_state_capacity_initialization_and_forward():
    models = matched_models()
    state = models[0].state_dict()
    counts = [sum(p.numel() for p in m.parameters()) for m in models]
    assert len(set(counts)) == 1
    images = torch.randn(2, 3, 32, 32, dtype=torch.double)
    outputs = []
    for model in models:
        assert model.state_dict().keys() == state.keys()
        assert all(torch.equal(v, state[k]) for k, v in model.state_dict().items())
        model.load_state_dict(state, strict=True)
        model.eval()
        with torch.no_grad():
            outputs.append(model(images, return_feature=True))
    for output in outputs[1:]:
        assert torch.equal(output[0], outputs[0][0])
        assert torch.equal(output[2], outputs[0][2])
    assert torch.equal(outputs[1][1], outputs[2][1])


@pytest.mark.parametrize("schedule", ["fixed", "progressive", "dann"])
@pytest.mark.parametrize("target", [0.0, 0.01, 0.05, 0.10])
def test_same_absolute_schedule_frozen_from_mutable_routing(tiny, schedule, target):
    models = matched_models()
    trainers = [make_trainer(m, grl_schedule=schedule, identity_target_lambda=target) for m in models]
    expected = GRLConfiguration(target, schedule, 5, 7)
    for epoch in range(1, 8):
        for t, sign in zip(trainers, [0, 1, -1]):
            t.model.identity_gradient.scale = 123  # Must not become the schedule target.
            t._update_grl_lambda(epoch)
            assert t.current_identity_encoder_scale == sign * expected.coefficient(epoch)
            assert t.model.identity_gradient.scale == t.current_identity_encoder_scale
            assert t.current_grl_lambda is None


@pytest.mark.parametrize("mode,sign", [("spoof_only", 0), ("identity_positive", 1), ("identity_adversarial", -1)])
@pytest.mark.parametrize("epoch", [5, 6])
def test_epoch_result_and_serialized_history_without_fitting(tiny, mode, sign, epoch):
    model = ControlledIdentityECNN(3, identity_mode=mode).double()
    trainer = make_trainer(model)
    scalar = Mock()
    scalar.item.return_value = 1.0
    trainer.loss_fn = Mock(subject_weight=0.1, return_value=LossOutput(scalar, scalar, None))
    before = copy.deepcopy(model.state_dict())
    result = trainer.train_epoch([{"image": X, "label": Y, "subject_label": SUBJECTS}], epoch)
    scale = sign * (0.05 if epoch > 5 else 0)
    assert result.grl_lambda is None
    assert result.identity_encoder_scale == scale
    assert result.history_record(epoch) == {
        "epoch": epoch, "identity_mode": mode, "identity_encoder_scale": scale,
        "subject_loss_enabled": mode != "spoof_only", "subject_weight": 0.1,
    }
    assert all(torch.equal(v, before[k]) for k, v in model.state_dict().items())
    assert trainer.loss_fn.call_args.kwargs["subject_loss_enabled"] == (mode != "spoof_only")


@pytest.mark.parametrize("mode", ["", "positive", "unknown", None])
def test_unknown_modes_rejected(mode):
    with pytest.raises(ValueError, match="identity_mode"):
        ControlledIdentityECNN(3, identity_mode=mode)


@pytest.mark.parametrize("overrides", [
    {"identity_mode": None}, {"identity_mode": "spoof_only"},
    {"identity_target_lambda": -0.05}, {"identity_target_lambda": float("nan")},
    {"identity_target_lambda": None}, {"grl_target_lambda": 0.05},
    {"grl_schedule": "unknown"}, {"subject_weight": 0},
    {"subject_weight": float("inf")}, {"warmup_epochs": 7},
])
def test_invalid_trainer_contracts(tiny, overrides):
    with pytest.raises(ValueError):
        make_trainer(matched_models()[1], **overrides)


@pytest.mark.parametrize("mode", IDENTITY_MODES[1:])
@pytest.mark.parametrize("labels", [None, torch.tensor([-1, 1, 2]), torch.tensor([0, 1, 9]),
                                  torch.tensor([0, 1]), torch.tensor([0.0, 1.0, 2.0])])
def test_identity_training_requires_all_subject_labels(tiny, mode, labels):
    model = ControlledIdentityECNN(3, identity_mode=mode).double()
    spoof, subject = model(X)
    with pytest.raises(ValueError):
        MultiTaskFASLoss(identity_mode=mode)(spoof, Y, subject, labels)


def test_identity_objective_cannot_be_disabled_or_lack_logits(tiny):
    model = matched_models()[1]
    spoof, subject = model(X)
    loss = MultiTaskFASLoss(identity_mode=model.identity_mode)
    with pytest.raises(ValueError):
        loss(spoof, Y, subject, SUBJECTS, subject_loss_enabled=False)
    with pytest.raises(ValueError):
        loss(spoof, Y, None, SUBJECTS)


def test_non_identity_baseline_legacy_loss_still_supported():
    model = PaperECNNClassifier()
    t = Trainer(model, Mock(), "cpu", total_epochs=3)
    t._update_grl_lambda(1)
    assert t.identity_mode is None and t.current_grl_lambda == 0
    loss = t.loss_fn(torch.zeros(3, 2), Y)
    assert loss.subject is None and loss.total is loss.spoof


def canonical_config():
    path = Path(__file__).resolve().parents[1] / "configs/controlled_identity_ablation.yaml"
    return yaml.safe_load(path.read_text())


def test_canonical_configs_only_vary_arm_destination_and_match_factory_initialization(tiny):
    source = canonical_config()
    before = copy.deepcopy(source)
    runs = build_configs(source)
    assert source == before
    assert [r["model"]["identity_mode"] for r in runs] == list(IDENTITY_MODES)
    normalized, states = [], []
    for run in runs:
        torch.manual_seed(run["training"]["seed"] + 1)
        model = common.build_model(run["model"]["name"], 3, run["model"])
        _, t = common.build_optimizer_and_trainer(model, run, "cpu")
        assert t.identity_mode == run["model"]["identity_mode"]
        assert t.total_epochs == run["training"]["epochs"]
        assert t.identity_config.target == run["model"]["identity_target_lambda"]
        assert t.loss_fn.subject_weight == run["training"]["subject_loss_weight"]
        states.append(model.state_dict())
        del run["model"]["identity_mode"]
        del run["training"]["output_dir"]
        normalized.append(run)
    assert normalized[0] == normalized[1] == normalized[2]
    for state in states[1:]:
        assert all(torch.equal(v, states[0][k]) for k, v in state.items())


@pytest.mark.parametrize("section,key,value", [
    ("model", "grl_lambda", -0.05), ("model", "identity_target_lambda", -0.05),
    ("model", "identity_mode", "identity_positive"), ("training", "subject_loss_weight", 0),
    ("training", "initialization_policy", "independent_seeds"),
    ("training", "optimizer", "sgd"), ("training", "model_selection", "test_accuracy"),
])
def test_invalid_canonical_contract(section, key, value):
    config = canonical_config()
    config[section][key] = value
    with pytest.raises(ValueError):
        build_configs(config)
