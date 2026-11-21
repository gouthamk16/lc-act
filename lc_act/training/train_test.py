import pytest
import torch
import tyro

from lc_act.models import lc_act_test as _fakes
from lc_act.shared import normalize as _normalize
from lc_act.training import checkpoints as _checkpoints
from lc_act.training import config as _config
from lc_act.training import data_loader as _data
from lc_act.training import train as _train

CPU = torch.device("cpu")


def test_lr_warms_up_then_decays_to_zero():
    assert _train.lr_scale(0.0, 0.02) == 0.01
    assert _train.lr_scale(0.01, 0.02) == pytest.approx(0.5)
    assert _train.lr_scale(0.02, 0.02) == pytest.approx(1.0)
    assert _train.lr_scale(1.5, 0.02) == pytest.approx(0.0)


def test_max_hours_zero_means_no_deadline():
    config = tyro.cli(_config.TrainConfig, args=["--max-hours", "0"])
    assert _train._Clock(config, total_steps=1).deadline is None


def test_budget_seconds_arms_after_the_first_batch():
    clock = _train._Clock(tyro.cli(_config.TrainConfig, args=["--budget-seconds", "300"]), total_steps=1)
    assert clock.deadline is None
    assert clock.progress(5) == 0.0
    clock.arm()
    assert clock.deadline is not None


def test_cli_flags_keep_their_names():
    config = tyro.cli(
        _config.TrainConfig,
        args=["--budget-steps", "5", "--val-batches", "4", "--object-only", "--n-obs", "3", "--lr", "1e-3"],
    )
    assert (config.budget_steps, config.val_batches, config.n_obs, config.lr) == (5, 4, 3, 1e-3)
    assert config.object_only
    assert config.fixed_budget
    assert not tyro.cli(_config.TrainConfig, args=[]).object_only


def test_budget_steps_stops_at_that_step_across_epochs(tmp_path):
    frames = torch.zeros(2, 1, 8, 8, 3, dtype=torch.uint8)
    batch = _data.Batch(frames, frames, torch.zeros(2, 1, 8), torch.zeros(2, 16, 7), ["soup", "soup"])
    config = tyro.cli(_config.TrainConfig, args=["--budget-steps", "5", "--out", str(tmp_path)])

    _train.train_loop(config, _fakes.make_model(), [batch, batch], _stats(), ["soup"], CPU)

    assert _checkpoints.read_checkpoint(tmp_path / "last.pt").step == 5


def _stats() -> _normalize.NormStats:
    return _normalize.NormStats.from_tensors(torch.zeros(4, 8), torch.zeros(4, 16, 7))
