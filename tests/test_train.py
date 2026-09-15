import torch
from torch import nn

from lc_act.train import evaluate_val
from lc_act.types import Batch, NormalizeStats


class _ConstZero(nn.Module):
    def forward(self, workspace, wrist, state, tasks):
        return torch.zeros(state.shape[0], 16, 7, device=state.device)


class _ConstPred(nn.Module):
    def __init__(self, value: torch.Tensor) -> None:
        super().__init__()
        self.value = value

    def forward(self, workspace, wrist, state, tasks):
        return self.value.expand(state.shape[0], -1, -1).to(state.device)


def _batch(action: torch.Tensor) -> Batch:
    n = action.shape[0]
    return Batch(
        workspace=torch.zeros(n, 3, 8, 8),
        wrist=torch.zeros(n, 3, 8, 8),
        state=torch.zeros(n, 8),
        action=action,
        tasks=["a"] * n,
    )


def _identity_stats() -> NormalizeStats:
    return NormalizeStats(
        state_mean=torch.zeros(8),
        state_std=torch.ones(8),
        action_mean=torch.zeros(7),
        action_std=torch.ones(7),
    )


def test_evaluate_val_is_mean_l1():
    batch = _batch(torch.ones(2, 16, 7))
    metrics = evaluate_val(
        _ConstZero(), [batch], torch.device("cpu"), max_batches=1,
        stats=_identity_stats(),
    )
    assert abs(metrics.l1 - 1.0) < 1e-6


def test_evaluate_val_raw_l1_uses_action_std():
    stats = NormalizeStats(
        state_mean=torch.zeros(8),
        state_std=torch.ones(8),
        action_mean=torch.zeros(7),
        action_std=torch.tensor([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 4.0]),
    )
    batch = _batch(torch.ones(1, 16, 7))
    metrics = evaluate_val(
        _ConstZero(), [batch], torch.device("cpu"), max_batches=1, stats=stats,
    )
    # z-scored L1 is 1; raw L1 is mean(|0 - std|) = (12 + 4) / 7
    assert abs(metrics.raw_l1 - (16.0 / 7.0)) < 1e-5


def test_evaluate_val_grip_acc_is_sign_match_in_raw_units():
    stats = _identity_stats()
    gt = torch.ones(1, 16, 7)
    gt[..., -1] = 1.0
    pred = torch.ones(1, 16, 7)
    pred[..., -1] = -1.0
    metrics = evaluate_val(
        _ConstPred(pred), [_batch(gt)], torch.device("cpu"), max_batches=1, stats=stats,
    )
    assert metrics.grip_acc == 0.0
    pred[..., -1] = 0.8
    metrics = evaluate_val(
        _ConstPred(pred), [_batch(gt)], torch.device("cpu"), max_batches=1, stats=stats,
    )
    assert metrics.grip_acc == 1.0
