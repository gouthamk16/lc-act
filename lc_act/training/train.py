"""Train LC-ACT on the LIBERO frame cache.

python -m lc_act.training.train --help
"""

import dataclasses
import math
import signal
import sys
import threading
import time

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import tyro

from lc_act.models import lc_act as _lc_act
from lc_act.shared import normalize as _normalize
from lc_act.training import checkpoints as _checkpoints
from lc_act.training import config as _config
from lc_act.training import data_loader as _data
from lc_act.training import validation as _validation

VAL_BATCH_SIZE = 64


def lr_scale(progress: float, warmup_frac: float) -> float:
    """Linear warmup over `warmup_frac` of the run (floored at 1%), then cosine decay to zero."""
    if progress < warmup_frac:
        return max(progress / warmup_frac, 0.01)
    rest = min(1.0, (progress - warmup_frac) / (1 - warmup_frac))
    return 0.5 * (1 + math.cos(math.pi * rest))


def train_step(
    model: _lc_act.LcAct,
    batch: _data.Batch,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    align_weight: float,
) -> float:
    """One mixed-precision step; returns the batch's z-scored action L1."""
    optimizer.zero_grad(set_to_none=True)
    device = batch.action.device
    with torch.autocast(device.type, enabled=device.type == "cuda"):
        memory = model.encode(batch.workspace, batch.wrist, batch.state, batch.tasks)
        prediction, log_scale = model.decode(memory, with_scale=True)
        error = (prediction - batch.action).abs()
        l1 = error.mean()

        # Laplace NLL on the detached residual: trains only the scale head.
        loss = l1 + (error.detach() * torch.exp(-log_scale) + log_scale).mean()
        if batch.teacher is not None:
            predicted = model.teacher_prediction(memory).float()
            loss = loss + align_weight * (1 - F.cosine_similarity(predicted, batch.teacher, dim=-1)).mean()

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return l1.detach().item()


def train_loop(
    config: _config.TrainConfig,
    model: _lc_act.LcAct,
    loader: torch.utils.data.DataLoader,
    stats: _normalize.NormStats,
    tasks: list[str],
    device: torch.device,
    *,
    val_loader: torch.utils.data.DataLoader | None = None,
    resume: _checkpoints.Checkpoint | None = None,
) -> None:
    """Train until the epochs, a budget, the deadline or SIGINT/SIGTERM end the run; saves to `config.out/last.pt`."""
    trainer = _Trainer(config, model, loader, stats, tasks, device, resume)
    if config.fixed_budget and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # the epoch loop; resumed runs restart at the saved epoch
    last_train, finished_epoch = 0.0, trainer.epoch
    for epoch in range(trainer.epoch, config.epochs):
        try:
            mean_loss, samples_per_second, steps = trainer.train_epoch(epoch)
        except torch.cuda.OutOfMemoryError:
            print("drop batch to 4 or drop wrist camera")
            sys.exit(1)
        if not config.fixed_budget:
            trainer.save(epoch if trainer.stop.is_set() else epoch + 1)
        if steps == 0:
            break
        last_train, finished_epoch = mean_loss, epoch
        print(f"epoch={epoch} l1={mean_loss:.4f} samples/s={samples_per_second:.1f}")
        if trainer.stop.is_set() or trainer.at_step_limit() or trainer.clock.expired():
            break

    if val_loader is not None:
        suites = {task: _data.suite_of(index) for index, task in enumerate(tasks)}
        val = _validation.evaluate_val(model, val_loader, device, config.val_batches, stats, suites)
        print(
            f"Step {trainer.step} : train {last_train:.4f} | val {val.l1:.4f} "
            f"| raw {val.raw_l1:.4f} | grip {val.grip_acc:.3f} | gpu {_peak_gpu_gb(device):.2f}GB"
        )
        print("suites: " + " | ".join(f"{name} {value:.4f}" for name, value in val.per_suite.items()))
    if config.fixed_budget:
        trainer.save(finished_epoch)


def main(config: _config.TrainConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and config.epochs != 1:
        raise RuntimeError("real training requires CUDA; use --epochs 1 for CPU smoke")
    torch.manual_seed(config.seed)

    # data: fixed-budget runs hold out the last episodes of every task for val
    train_set, val_set, data_stats, tasks = _data.load_cached(
        config.cache,
        object_only=config.object_only,
        n_obs=config.n_obs,
        obs_stride=config.obs_stride,
        holdout=config.fixed_budget,
        teacher=config.align_weight > 0,
    )
    # MINERVA: long-horizon demos are the scarce, hard ones; oversample them.
    weights = None
    if config.long_weight != 1:
        long_suite = torch.from_numpy(np.isin(train_set.task_index[train_set.rows], _data.SUITES["libero_10"]))
        weights = 1.0 + (config.long_weight - 1.0) * long_suite.double()
    loader = _make_loader(train_set, config.batch_size, workers=config.workers, weights=weights)
    val_loader = None if val_set is None else _make_loader(val_set, VAL_BATCH_SIZE, workers=2, shuffle=False)

    # model: fresh, or restored with its stats, tasks and optimizer state
    checkpoint = None if config.resume is None else _read_resume(config)
    if checkpoint is None:
        model, stats = config.model_config().create(pretrained=True).to(device), data_stats
    else:
        model, stats = _checkpoints.restore(checkpoint, device)
        tasks = checkpoint.tasks
    train_loop(config, model, loader, stats, tasks, device, val_loader=val_loader, resume=checkpoint)


def _read_resume(config: _config.TrainConfig) -> _checkpoints.Checkpoint:
    checkpoint = _checkpoints.read_checkpoint(config.resume)
    if checkpoint.optimizer is None:
        raise RuntimeError("checkpoint missing optimizer; cannot resume")
    for name in ("n_obs", "obs_stride"):
        value = getattr(checkpoint, name)
        if value != getattr(config, name):
            raise RuntimeError(f"checkpoint has {name}={value}; pass --{name.replace('_', '-')} {value}")
    return checkpoint


@dataclasses.dataclass
class _Clock:
    """Wall-clock side of a run: the deadline and the progress fraction that drives the LR schedule."""

    config: _config.TrainConfig
    total_steps: int
    started: float = dataclasses.field(default_factory=time.monotonic)
    deadline: float | None = None

    def __post_init__(self) -> None:
        if not self.config.fixed_budget and self.config.max_hours > 0:
            self.deadline = self.started + self.config.max_hours * 3600

    def arm(self) -> None:
        """Start the seconds budget after the first batch, so startup (CLIP, loader workers) is not counted."""
        if self.config.budget_seconds is not None and self.deadline is None:
            self.deadline = time.monotonic() + self.config.budget_seconds

    def expired(self) -> bool:
        return self.deadline is not None and time.monotonic() >= self.deadline

    def progress(self, step: int) -> float:
        config = self.config
        if config.budget_steps is not None:
            return step / config.budget_steps
        if config.budget_seconds is not None:
            return 0.0 if self.deadline is None else 1 - (self.deadline - time.monotonic()) / config.budget_seconds
        if self.deadline is not None:
            return (time.monotonic() - self.started) / (self.deadline - self.started)
        return step / self.total_steps


class _Trainer:
    """Optimizer, clock, stop flag and step counter shared by the epochs of one run."""

    def __init__(
        self,
        config: _config.TrainConfig,
        model: _lc_act.LcAct,
        loader: torch.utils.data.DataLoader,
        stats: _normalize.NormStats,
        tasks: list[str],
        device: torch.device,
        resume: _checkpoints.Checkpoint | None,
    ) -> None:
        self.config, self.model, self.loader, self.stats, self.tasks = config, model, loader, stats, tasks

        # optimizer and AMP scaler, restored when resuming
        params = [param for param in model.parameters() if param.requires_grad]
        self.optimizer = torch.optim.AdamW(params, lr=config.lr, fused=device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
        if resume is not None:
            self.optimizer.load_state_dict(resume.optimizer)
            if resume.scaler is not None:
                self.scaler.load_state_dict(resume.scaler)
        self.epoch = 0 if resume is None else resume.epoch
        self.step = 0 if resume is None else resume.step

        self.device = device
        self.clock = _Clock(config, total_steps=max(config.epochs * len(loader), 1))
        # CPU runs are one-step smoke tests unless a step budget says otherwise.
        self.step_limit = config.budget_steps
        if self.step_limit is None and device.type == "cpu":
            self.step_limit = self.step + 1
        self.stop = threading.Event()
        signal.signal(signal.SIGINT, lambda *_: self.stop.set())
        signal.signal(signal.SIGTERM, lambda *_: self.stop.set())

    def at_step_limit(self) -> bool:
        return self.step_limit is not None and self.step >= self.step_limit

    def save(self, epoch: int) -> None:
        _checkpoints.save_checkpoint(
            self.config.out / "last.pt",
            self.model,
            self.stats,
            self.tasks,
            epoch=epoch,
            step=self.step,
            optimizer=self.optimizer,
            scaler=self.scaler,
        )

    def train_epoch(self, epoch: int) -> tuple[float, float, int]:
        """Returns the epoch's mean train L1, samples per second, and steps taken."""
        config = self.config
        self.model.train()
        self.model.text.eval()
        total_loss, samples, steps, started = 0.0, 0, 0, time.monotonic()

        for cpu_batch in self.loader:
            if self.stop.is_set() or self.clock.expired():
                break
            batch = cpu_batch.to(self.device)
            for group in self.optimizer.param_groups:
                group["lr"] = config.lr * lr_scale(self.clock.progress(self.step), config.warmup_frac)
            loss = train_step(self.model, batch, self.optimizer, self.scaler, config.align_weight)
            self.clock.arm()

            count = batch.action.shape[0]
            total_loss += loss * count
            samples += count
            steps += 1
            self.step += 1
            if not config.fixed_budget and config.save_every > 0 and self.step % config.save_every == 0:
                self.save(epoch)
            if self.at_step_limit():
                break

        elapsed = max(time.monotonic() - started, 1e-9)
        return total_loss / max(samples, 1), samples / elapsed, steps


def _make_loader(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    *,
    workers: int,
    shuffle: bool = True,
    weights: torch.Tensor | None = None,
) -> torch.utils.data.DataLoader:
    sampler = None if weights is None else torch.utils.data.WeightedRandomSampler(weights, len(weights))
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=workers,
        collate_fn=_data.collate,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        persistent_workers=workers > 0,
    )


def _peak_gpu_gb(device: torch.device) -> float:
    return torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0.0


if __name__ == "__main__":
    main(tyro.cli(_config.TrainConfig))
