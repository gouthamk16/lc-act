import torch
from torch import nn

from lc_act.train import evaluate_val
from lc_act.types import Batch


class _ConstZero(nn.Module):
    def forward(self, workspace, wrist, state, tasks):
        return torch.zeros(state.shape[0], 16, 7, device=state.device)


def test_evaluate_val_is_mean_l1():
    batch = Batch(
        workspace=torch.zeros(2, 3, 8, 8),
        wrist=torch.zeros(2, 3, 8, 8),
        state=torch.zeros(2, 8),
        action=torch.ones(2, 16, 7),
        tasks=["a", "a"],
    )
    loss = evaluate_val(_ConstZero(), [batch], torch.device("cpu"), max_batches=1)
    assert abs(loss - 1.0) < 1e-6
