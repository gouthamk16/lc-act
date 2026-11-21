import pytest
import torch
from torch import nn

from lc_act.shared import normalize as _normalize
from lc_act.training import data_loader as _data
from lc_act.training import validation as _validation

CPU = torch.device("cpu")


class ConstPred(nn.Module):
    def __init__(self, value: torch.Tensor) -> None:
        super().__init__()
        self.value = value

    def forward(self, workspace, wrist, state, tasks):
        return self.value.expand(state.shape[0], -1, -1).to(state.device)


def test_evaluate_val_is_mean_l1():
    metrics = _validation.evaluate_val(
        ConstPred(torch.zeros(1, 16, 7)), [_batch(torch.ones(2, 16, 7))], CPU, 1, _stats()
    )
    assert metrics.l1 == pytest.approx(1.0)


def test_evaluate_val_raw_l1_uses_action_std():
    stats = _stats(action_std=torch.tensor([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 4.0]))
    metrics = _validation.evaluate_val(ConstPred(torch.zeros(1, 16, 7)), [_batch(torch.ones(1, 16, 7))], CPU, 1, stats)
    # z-scored L1 is 1; raw L1 is mean(|0 - std|) = (12 + 4) / 7
    assert metrics.raw_l1 == pytest.approx(16.0 / 7.0)


def test_evaluate_val_grip_acc_is_sign_match_in_raw_units():
    gt = torch.ones(1, 16, 7)
    pred = torch.ones(1, 16, 7)
    pred[..., -1] = -1.0
    assert _validation.evaluate_val(ConstPred(pred), [_batch(gt)], CPU, 1, _stats()).grip_acc == 0.0

    pred[..., -1] = 0.8
    assert _validation.evaluate_val(ConstPred(pred), [_batch(gt)], CPU, 1, _stats()).grip_acc == 1.0


def test_evaluate_val_breaks_down_by_suite():
    batch = _batch(torch.ones(2, 16, 7))
    batch.tasks = ["long", "object"]
    suites = {"long": "libero_10", "object": "libero_object"}
    metrics = _validation.evaluate_val(ConstPred(torch.zeros(1, 16, 7)), [batch], CPU, 1, _stats(), suites)
    assert metrics.per_suite == {"libero_10": 1.0, "libero_object": 1.0}


def _batch(action: torch.Tensor) -> _data.Batch:
    n = action.shape[0]
    return _data.Batch(torch.zeros(n, 3, 8, 8), torch.zeros(n, 3, 8, 8), torch.zeros(n, 8), action, ["a"] * n)


def _stats(action_std: torch.Tensor | None = None) -> _normalize.NormStats:
    return _normalize.NormStats(
        state_mean=torch.zeros(8),
        state_std=torch.ones(8),
        action_mean=torch.zeros(7),
        action_std=torch.ones(7) if action_std is None else action_std,
    )
