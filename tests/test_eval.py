import math

import torch

from lc_act.eval import temporal_ensemble


def test_temporal_ensemble_zero_decay_is_mean():
    candidates = [torch.tensor([0.0, 2.0]), torch.tensor([2.0, 4.0])]
    assert torch.allclose(temporal_ensemble(candidates, decay=0.0), torch.tensor([1.0, 3.0]))


def test_temporal_ensemble_weights_oldest_prediction_most():
    old, new = torch.tensor([1.0]), torch.tensor([0.0])
    w_new = math.exp(-0.5)
    expected = 1.0 / (1.0 + w_new)
    assert torch.allclose(temporal_ensemble([old, new], decay=0.5), torch.tensor([expected]))


def test_temporal_ensemble_single_candidate_passes_through():
    action = torch.tensor([0.3, -1.0])
    assert torch.equal(temporal_ensemble([action], decay=0.01), action)


def test_temporal_ensemble_equal_scales_match_decay_weighting():
    candidates = [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])]
    scales = [torch.full((2,), 0.3), torch.full((2,), 0.3)]
    assert torch.allclose(
        temporal_ensemble(candidates, 0.5, scales), temporal_ensemble(candidates, 0.5)
    )


def test_temporal_ensemble_downweights_unsure_candidate_per_dim():
    old, new = torch.tensor([1.0, 1.0]), torch.tensor([0.0, 0.0])
    unsure_old = [torch.tensor([math.log(9.0), 0.0]), torch.zeros(2)]
    result = temporal_ensemble([old, new], 0.0, unsure_old)
    assert torch.allclose(result, torch.tensor([0.1, 0.5]))
