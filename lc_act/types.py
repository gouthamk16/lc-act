from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

STATE_DIM = 8
ACTION_DIM = 7
HORIZON = 16
TOKEN_DIM = 256
CLIP_DIM = 512
RESNET_DIM = 256
N_HEADS = 8
N_ENCODER_LAYERS = 3
N_DECODER_LAYERS = 2
FFN_DIM = 1024
IMAGE_SIZE = 256
VISION_SIZE = 128
DATASET_FPS = 10
ENV_FPS = 20


@dataclass
class NormalizeStats:
    state_mean: torch.Tensor
    state_std: torch.Tensor
    action_mean: torch.Tensor
    action_std: torch.Tensor

    def apply_state(self, state: torch.Tensor) -> torch.Tensor:
        return (state - self.state_mean) / self.state_std

    def apply_action(self, action: torch.Tensor) -> torch.Tensor:
        return (action - self.action_mean) / self.action_std

    def invert_action(self, action: torch.Tensor) -> torch.Tensor:
        return action * self.action_std + self.action_mean

    @classmethod
    def from_tensors(cls, state: torch.Tensor, action: torch.Tensor) -> "NormalizeStats":
        dims_state = tuple(range(state.ndim - 1))
        dims_action = tuple(range(action.ndim - 1))
        return cls(
            state_mean=state.mean(dim=dims_state),
            state_std=state.std(dim=dims_state, unbiased=False).clamp_min(1e-6),
            action_mean=action.mean(dim=dims_action),
            action_std=action.std(dim=dims_action, unbiased=False).clamp_min(1e-6),
        )


@dataclass
class Checkpoint:
    trainable: dict[str, torch.Tensor]
    stats: NormalizeStats
    horizon: int
    tasks: list[str]
    epoch: int = 0
    step: int = 0
    optimizer: dict[str, Any] | None = None
    scaler: dict[str, Any] | None = None
    n_obs: int = 1

    def to_payload(self) -> dict[str, Any]:
        return {
            "trainable": self.trainable,
            "stats": {
                "state_mean": self.stats.state_mean.cpu(),
                "state_std": self.stats.state_std.cpu(),
                "action_mean": self.stats.action_mean.cpu(),
                "action_std": self.stats.action_std.cpu(),
            },
            "horizon": self.horizon,
            "tasks": self.tasks,
            "epoch": self.epoch,
            "step": self.step,
            "optimizer": self.optimizer,
            "scaler": self.scaler,
            "n_obs": self.n_obs,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "Checkpoint":
        if "stats" not in payload:
            raise RuntimeError("checkpoint missing NormalizeStats; refuse to eval")
        raw_stats = payload["stats"]
        stats = NormalizeStats(
            state_mean=raw_stats["state_mean"],
            state_std=raw_stats["state_std"],
            action_mean=raw_stats["action_mean"],
            action_std=raw_stats["action_std"],
        )
        return cls(
            trainable=payload["trainable"],
            stats=stats,
            horizon=int(payload["horizon"]),
            tasks=list(payload["tasks"]),
            epoch=int(payload.get("epoch", 0)),
            step=int(payload.get("step", 0)),
            optimizer=payload.get("optimizer"),
            scaler=payload.get("scaler"),
            n_obs=int(payload.get("n_obs", 1)),
        )


@dataclass
class Batch:
    workspace: torch.Tensor
    wrist: torch.Tensor
    state: torch.Tensor
    action: torch.Tensor
    tasks: list[str]
