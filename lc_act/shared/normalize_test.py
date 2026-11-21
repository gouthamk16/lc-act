import torch

from lc_act.shared import normalize as _normalize


def test_normalize_then_unnormalize_action_is_identity():
    stats = _normalize.NormStats(
        state_mean=torch.zeros(8),
        state_std=torch.ones(8),
        action_mean=torch.tensor([1.0, 0, 0, 0, 0, 0, 0]),
        action_std=torch.tensor([2.0, 1, 1, 1, 1, 1, 1]),
    )
    raw = torch.zeros(2, 16, 7)
    raw[..., 0] = 5.0

    out = stats.unnormalize_action(stats.normalize_action(raw))
    assert torch.allclose(out, raw, atol=1e-5)


def test_std_is_clamped_away_from_zero():
    stats = _normalize.NormStats.from_tensors(state=torch.zeros(4, 8), action=torch.zeros(4, 16, 7))
    assert torch.all(stats.state_std >= 1e-6)
    assert torch.all(stats.action_std >= 1e-6)


def test_dict_roundtrip_keeps_every_field():
    stats = _normalize.NormStats.from_tensors(torch.randn(4, 8), torch.randn(4, 16, 7))
    restored = _normalize.NormStats.from_dict(stats.to_dict())
    assert all(torch.equal(getattr(stats, key), getattr(restored, key)) for key in stats.to_dict())
