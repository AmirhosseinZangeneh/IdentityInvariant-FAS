import torch

from identity_invariant_fas.models import AblationECNN, IdentityInvariantECNN, PaperECNNClassifier


def test_baseline_shapes():
    model = PaperECNNClassifier().eval()
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        logits, features = model(x, return_feature=True)
    assert logits.shape == (2, 2)
    assert features.shape == (2, 256)


def test_identity_invariant_shapes():
    model = IdentityInvariantECNN(num_subjects=12).eval()
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        spoof_logits, subject_logits = model(x)
        features = model.extract_features(x)
    assert spoof_logits.shape == (2, 2)
    assert subject_logits.shape == (2, 12)
    assert features.shape == (2, 256)


def test_ablation_is_architecture_matched():
    model = AblationECNN(num_subjects=12).eval()
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        spoof_logits, subject_logits = model(x)
        features = model.extract_features(x)
    assert spoof_logits.shape == (2, 2)
    assert subject_logits.shape == (2, 12)
    assert features.shape == (2, 256)
