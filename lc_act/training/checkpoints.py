"""Checkpoints keep the trainable weights, the normalization stats and the config fields needed to rebuild the model.

The frozen CLIP text encoder is not saved; it is reloaded from the Hugging Face cache.
"""

import dataclasses
import pathlib
from typing import Any

import torch

from lc_act.models import lc_act as _lc_act
from lc_act.shared import normalize as _normalize

FROZEN_PREFIX = "text."


@dataclasses.dataclass
class Checkpoint:
    trainable: dict[str, torch.Tensor]
    stats: _normalize.NormStats
    horizon: int
    tasks: list[str]
    epoch: int = 0
    step: int = 0
    optimizer: dict[str, Any] | None = None
    scaler: dict[str, Any] | None = None
    # Defaults are what checkpoints from before each field existed were trained with.
    n_obs: int = 1
    obs_stride: int = 1
    n_decoder_layers: int = 2

    def model_config(self) -> _lc_act.LcActConfig:
        return _lc_act.LcActConfig(
            horizon=self.horizon,
            n_obs=self.n_obs,
            obs_stride=self.obs_stride,
            num_decoder_layers=self.n_decoder_layers,
        )

    def to_payload(self) -> dict[str, Any]:
        payload = {field.name: getattr(self, field.name) for field in dataclasses.fields(self)}
        payload["stats"] = self.stats.to_dict()
        return payload

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "Checkpoint":
        if "stats" not in payload:
            raise RuntimeError("checkpoint missing normalization stats; refuse to eval")
        names = {field.name for field in dataclasses.fields(cls)}
        fields = {key: value for key, value in payload.items() if key in names}
        fields["stats"] = _normalize.NormStats.from_dict(payload["stats"])
        return cls(**fields)


def trainable_state_dict(model: _lc_act.LcAct) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in model.state_dict().items() if not key.startswith(FROZEN_PREFIX)}


def load_trainable(model: _lc_act.LcAct, state_dict: dict[str, torch.Tensor]) -> _lc_act.LcAct:
    result = model.load_state_dict(state_dict, strict=False)
    missing = [key for key in result.missing_keys if not key.startswith(FROZEN_PREFIX)]
    if missing:
        raise RuntimeError(f"checkpoint missing trainable keys: {missing}")
    return model


def save_checkpoint(
    path: pathlib.Path,
    model: _lc_act.LcAct,
    stats: _normalize.NormStats,
    tasks: list[str],
    *,
    epoch: int,
    step: int,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
) -> None:
    """Write atomically, so a run killed mid-save keeps its previous checkpoint."""
    config = model.config
    checkpoint = Checkpoint(
        trainable=trainable_state_dict(model),
        stats=stats,
        horizon=config.horizon,
        tasks=tasks,
        epoch=epoch,
        step=step,
        optimizer=optimizer.state_dict(),
        scaler=scaler.state_dict(),
        n_obs=config.n_obs,
        obs_stride=config.obs_stride,
        n_decoder_layers=config.num_decoder_layers,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(checkpoint.to_payload(), tmp)
    tmp.replace(path)


def read_checkpoint(path: pathlib.Path) -> Checkpoint:
    return Checkpoint.from_payload(torch.load(path, map_location="cpu", weights_only=False))


def restore(checkpoint: Checkpoint, device: torch.device) -> tuple[_lc_act.LcAct, _normalize.NormStats]:
    """Rebuild the model; its ResNet weights come from the checkpoint, not ImageNet."""
    model = load_trainable(checkpoint.model_config().create(pretrained=False), checkpoint.trainable)
    return model.to(device), checkpoint.stats.to(device)


def load_checkpoint(path: pathlib.Path, device: torch.device) -> tuple[_lc_act.LcAct, _normalize.NormStats]:
    return restore(read_checkpoint(path), device)
