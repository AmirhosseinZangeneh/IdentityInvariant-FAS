import numpy as np

from identity_invariant_fas.evaluation.metrics import compute_apcer_bpcer, compute_acer


def test_pad_metric_label_convention():
    # Bona fide: 0,0 -> one false positive. Attack: 1,1 -> one false negative.
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1])

    apcer, bpcer = compute_apcer_bpcer(y_true, y_pred)
    assert apcer == 0.5
    assert bpcer == 0.5
    assert compute_acer(y_true, y_pred) == 0.5
