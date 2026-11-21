import pytest
import torch
from torch import nn

from lc_act.models import lc_act_test as _fakes
from lc_act.shared import normalize as _normalize
from lc_act.training import checkpoints as _checkpoints


def test_payload_roundtrip_keeps_epoch_step_optimizer():
    checkpoint = _checkpoints.Checkpoint(
        trainable={"w": torch.tensor(1.0)},
        stats=_stats(),
        horizon=16,
        tasks=["soup"],
        epoch=3,
        step=50,
        optimizer={"lr": 1e-4},
        scaler={"scale": 2.0},
        n_obs=2,
    )
    restored = _checkpoints.Checkpoint.from_payload(checkpoint.to_payload())
    assert (restored.epoch, restored.step, restored.n_obs) == (3, 50, 2)
    assert restored.optimizer == {"lr": 1e-4}
    assert restored.scaler == {"scale": 2.0}


def test_payload_without_later_fields_gets_their_old_defaults():
    payload = _checkpoints.Checkpoint(trainable={}, stats=_stats(), horizon=16, tasks=["soup"]).to_payload()
    for key in ("n_obs", "obs_stride", "n_decoder_layers"):
        del payload[key]
    config = _checkpoints.Checkpoint.from_payload(payload).model_config()
    assert (config.n_obs, config.obs_stride, config.num_decoder_layers) == (1, 1, 2)


def test_payload_without_stats_is_refused():
    with pytest.raises(RuntimeError, match="normalization stats"):
        _checkpoints.Checkpoint.from_payload({"trainable": {}, "horizon": 16, "tasks": []})


def test_load_trainable_rejects_missing_camera_embed():
    model = _fakes.make_model()
    weights = _checkpoints.trainable_state_dict(model)
    del weights["camera_embed"]
    with pytest.raises(RuntimeError, match="missing trainable"):
        _checkpoints.load_trainable(model, weights)


def test_trainable_state_dict_skips_text_keys():
    model = _fakes.make_model()
    model.text.dummy = nn.Parameter(torch.ones(4))
    keys = _checkpoints.trainable_state_dict(model)
    assert not any(key.startswith("text.") for key in keys)
    assert any(key.startswith("action_head.") for key in keys)


def _stats() -> _normalize.NormStats:
    return _normalize.NormStats.from_tensors(torch.zeros(4, 8), torch.zeros(4, 16, 7))
