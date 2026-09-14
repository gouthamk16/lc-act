import torch

from lc_act.normalize import NormalizeStats


def test_apply_then_invert_action_is_identity():
    stats = NormalizeStats(
        state_mean=torch.zeros(8),
        state_std=torch.ones(8),
        action_mean=torch.tensor([1.0, 0, 0, 0, 0, 0, 0]),
        action_std=torch.tensor([2.0, 1, 1, 1, 1, 1, 1]),
    )
    raw = torch.zeros(2, 16, 7)
    raw[..., 0] = 5.0
    out = stats.invert_action(stats.apply_action(raw))
    assert torch.allclose(out, raw, atol=1e-5)


def test_std_is_clamped_away_from_zero():
    stats = NormalizeStats.from_tensors(
        state=torch.zeros(4, 8),
        action=torch.zeros(4, 16, 7),
    )
    assert torch.all(stats.state_std >= 1e-6)
    assert torch.all(stats.action_std >= 1e-6)
