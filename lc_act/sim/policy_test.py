import math

import torch

from lc_act.models import lc_act as _lc_act
from lc_act.sim import policy as _policy


def test_temporal_ensemble_zero_decay_is_mean():
    candidates = [torch.tensor([0.0, 2.0]), torch.tensor([2.0, 4.0])]
    assert torch.allclose(_policy.temporal_ensemble(candidates, decay=0.0), torch.tensor([1.0, 3.0]))


def test_temporal_ensemble_weights_oldest_prediction_most():
    old, new = torch.tensor([1.0]), torch.tensor([0.0])
    w_new = math.exp(-0.5)
    expected = 1.0 / (1.0 + w_new)
    assert torch.allclose(_policy.temporal_ensemble([old, new], decay=0.5), torch.tensor([expected]))


def test_temporal_ensemble_single_candidate_passes_through():
    action = torch.tensor([0.3, -1.0])
    assert torch.equal(_policy.temporal_ensemble([action], decay=0.01), action)


def test_temporal_ensemble_equal_scales_match_decay_weighting():
    candidates = [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])]
    scales = [torch.full((2,), 0.3), torch.full((2,), 0.3)]
    assert torch.allclose(
        _policy.temporal_ensemble(candidates, 0.5, scales), _policy.temporal_ensemble(candidates, 0.5)
    )


def test_temporal_ensemble_downweights_unsure_candidate_per_dim():
    old, new = torch.tensor([1.0, 1.0]), torch.tensor([0.0, 0.0])
    unsure_old = [torch.tensor([math.log(9.0), 0.0]), torch.zeros(2)]
    result = _policy.temporal_ensemble([old, new], 0.0, unsure_old)
    assert torch.allclose(result, torch.tensor([0.1, 0.5]))


def test_first_frame_fills_the_history_then_frames_shift_in():
    ensembler = _policy.ChunkEnsembler(_lc_act.LcActConfig(n_obs=2, obs_stride=2))
    workspace, _, state = ensembler.observe(_frame(0.0))
    assert workspace[:, 0].tolist() == [0.0, 0.0]
    assert state.shape == (2, 8)

    ensembler.observe(_frame(1.0))
    workspace, _, _ = ensembler.observe(_frame(2.0))
    assert workspace[:, 0].tolist() == [0.0, 2.0]


def test_act_ensembles_every_chunk_covering_the_step():
    ensembler = _policy.ChunkEnsembler(_lc_act.LcActConfig(), decay=0.0)
    assert torch.equal(ensembler.act(torch.zeros(3, 1)), torch.zeros(1))
    assert torch.allclose(ensembler.act(torch.ones(3, 1)), torch.tensor([0.5]))
    assert ensembler.step == 2


def _frame(value: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.full((2,), value), torch.full((2,), value), torch.full((8,), value)
