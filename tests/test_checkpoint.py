import pytest
import torch
from torch import nn

from lc_act.model import LcAct, load_trainable, trainable_state_dict
from lc_act.train import parse_args, training_deadline
from lc_act.types import CLIP_DIM, RESNET_DIM, Checkpoint, NormalizeStats


class FakeVision(nn.Module):
    def encode(self, images: torch.Tensor, size: int = 0, group: int = 1) -> torch.Tensor:
        return torch.zeros(images.shape[0], 4, RESNET_DIM, device=images.device)


class FakeText(nn.Module):
    def encode(self, texts: list[str]) -> torch.Tensor:
        return torch.zeros(len(texts), CLIP_DIM)


def _stats() -> NormalizeStats:
    state = torch.zeros(4, 8)
    action = torch.zeros(4, 16, 7)
    return NormalizeStats.from_tensors(state, action)


def test_checkpoint_roundtrip_keeps_epoch_step_optimizer():
    stats = _stats()
    checkpoint = Checkpoint(
        trainable={"w": torch.tensor(1.0)},
        stats=stats,
        horizon=16,
        tasks=["soup"],
        epoch=3,
        step=50,
        optimizer={"lr": 1e-4},
        scaler={"scale": 2.0},
        n_obs=2,
    )
    restored = Checkpoint.from_payload(checkpoint.to_payload())
    assert restored.epoch == 3
    assert restored.step == 50
    assert restored.optimizer == {"lr": 1e-4}
    assert restored.scaler == {"scale": 2.0}
    assert restored.n_obs == 2


def test_checkpoint_without_n_obs_is_single_frame():
    payload = Checkpoint(
        trainable={}, stats=_stats(), horizon=16, tasks=["soup"],
    ).to_payload()
    del payload["n_obs"]
    assert Checkpoint.from_payload(payload).n_obs == 1


def test_load_trainable_rejects_missing_camera_embed():
    model = LcAct(FakeVision(), FakeText())
    weights = trainable_state_dict(model)
    del weights["camera_embed"]
    with pytest.raises(RuntimeError, match="missing trainable"):
        load_trainable(model, weights)


def test_max_hours_zero_means_no_deadline():
    args = parse_args(["--max-hours", "0"])
    assert training_deadline(args.max_hours, now=10.0) is None


def test_budget_seconds_and_val_batches_parse():
    args = parse_args(["--budget-seconds", "300", "--val-batches", "4"])
    assert args.budget_seconds == 300.0
    assert args.val_batches == 4


def test_all_tasks_and_n_obs_parse():
    args = parse_args(["--all-tasks", "--n-obs", "2"])
    assert args.all_tasks
    assert args.n_obs == 2
