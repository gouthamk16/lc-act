"""Closed-loop inference: replan every env step and execute the ACT temporal ensemble of all chunks covering it."""

import collections

import numpy as np
import torch

from lc_act.models import lc_act as _lc_act
from lc_act.shared import normalize as _normalize

ENSEMBLE_DECAY = 0.01  # ACT's k: older predictions weigh slightly more

Frame = tuple[torch.Tensor, torch.Tensor, torch.Tensor]  # workspace, wrist, normalized state of one env step


class ChunkEnsembler:
    """One episode's inference state: the frame history the model sees and each pending chunk's vote per step."""

    def __init__(self, config: _lc_act.LcActConfig, decay: float = ENSEMBLE_DECAY) -> None:
        self.obs_stride = config.obs_stride
        self.decay = decay
        self.step = 0
        self._history: collections.deque[Frame] = collections.deque(maxlen=config.history_size)
        self._pending: collections.defaultdict[int, list[torch.Tensor]] = collections.defaultdict(list)
        self._scales: collections.defaultdict[int, list[torch.Tensor]] = collections.defaultdict(list)

    def observe(self, frame: Frame) -> Frame:
        """Add this step's frame; return the model's (n_obs, ...) inputs, oldest first.

        The episode's first frame fills the whole history, as the training sampler clamps to the episode start.
        """
        self._history.extend([frame] * (1 if self._history else self._history.maxlen))
        frames = list(self._history)[:: self.obs_stride]
        workspace, wrist, state = (torch.stack(parts) for parts in zip(*frames, strict=True))
        return workspace, wrist, state

    def act(self, chunk: torch.Tensor, log_scale: torch.Tensor | None = None) -> torch.Tensor:
        """Add this step's (horizon, 7) chunk as a vote for each step it covers; return this step's action.

        With `log_scale` (pass it every step or never), votes are also weighted by the model's confidence.
        """
        for offset, action in enumerate(chunk):
            self._pending[self.step + offset].append(action)
        if log_scale is not None:
            for offset, scale in enumerate(log_scale):
                self._scales[self.step + offset].append(scale)

        action = temporal_ensemble(self._pending.pop(self.step), self.decay, self._scales.pop(self.step, None))
        self.step += 1
        return action


class ModelEnsemble(torch.nn.Module):
    """Averages the chunks of independently trained models that share inputs and normalization."""

    def __init__(self, models: list[_lc_act.LcAct]) -> None:
        super().__init__()
        self.config = models[0].config
        if any((m.config.n_obs, m.config.obs_stride) != (self.config.n_obs, self.config.obs_stride) for m in models):
            raise ValueError("ensemble members need the same n_obs and obs_stride")
        self.members = torch.nn.ModuleList(models)

    def forward(self, *inputs, with_scale: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        outs = [member(*inputs, with_scale=True) for member in self.members]
        actions = torch.stack([a for a, _ in outs]).mean(0)
        if not with_scale:
            return actions
        return actions, torch.stack([s for _, s in outs]).mean(0)


def temporal_ensemble(
    candidates: list[torch.Tensor], decay: float, log_scales: list[torch.Tensor] | None = None
) -> torch.Tensor:
    """ACT temporal ensembling: candidates oldest first, weighted exp(-decay * i).

    With log_scales (the model's Laplace scale per candidate and dim), each weight is also divided by its scale,
    so a confident fresh prediction outvotes an unsure old one.
    """
    ages = torch.arange(len(candidates), dtype=torch.float32, device=candidates[0].device)
    weights = torch.exp(-decay * ages)[:, None].expand(-1, candidates[0].shape[-1])
    if log_scales is not None:
        weights = weights * torch.exp(-torch.stack(log_scales).float())
    weights = weights / weights.sum(0)
    return (torch.stack(candidates) * weights).sum(0)


def to_tensors(
    frame: tuple[np.ndarray, np.ndarray, np.ndarray], stats: _normalize.NormStats, device: torch.device
) -> Frame:
    """An observed (workspace, wrist, state) frame as uint8 images and the normalized state, on `device`."""
    workspace, wrist, state = frame
    return (
        torch.from_numpy(workspace).to(device),
        torch.from_numpy(wrist).to(device),
        stats.normalize_state(torch.from_numpy(state).to(device)),
    )
