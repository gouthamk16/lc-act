from __future__ import annotations

import argparse
import math
import signal
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from lc_act.types import Batch, Checkpoint, NormalizeStats

if TYPE_CHECKING:
    from lc_act.model import LcAct


class StopFlag:
    def __init__(self) -> None:
        self.requested = False

    def request(self, *_args: object) -> None:
        self.requested = True


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LC-ACT on LIBERO-Object")
    parser.add_argument("--out", type=Path, default=Path("outputs/lc_act"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-hours", type=float, default=2)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--repo-id", default="lerobot/libero")
    parser.add_argument("--budget-seconds", type=float, default=None)
    parser.add_argument("--val-batches", type=int, default=200)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


BASE_LR = 1e-4
WARMUP_FRAC = 0.02


def cosine_lr_scale(progress: float) -> float:
    if progress < WARMUP_FRAC:
        return max(progress / WARMUP_FRAC, 0.01)
    rest = min(1.0, (progress - WARMUP_FRAC) / (1 - WARMUP_FRAC))
    return 0.5 * (1 + math.cos(math.pi * rest))


def training_deadline(max_hours: float, now: float) -> float | None:
    if max_hours <= 0:
        return None
    return now + max_hours * 3600


def make_loader(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    shuffle: bool = True,
    workers: int = 0,
) -> DataLoader:
    from lc_act.data import collate

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=workers,
        collate_fn=collate,
        shuffle=shuffle,
        persistent_workers=workers > 0,
    )


def _move_batch(batch: Batch, device: torch.device) -> Batch:
    return Batch(
        workspace=batch.workspace.to(device),
        wrist=batch.wrist.to(device),
        state=batch.state.to(device),
        action=batch.action.to(device),
        tasks=batch.tasks,
    )


def _train_batch(
    model: nn.Module,
    batch: Batch,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
) -> float:
    optimizer.zero_grad(set_to_none=True)
    autocast = torch.autocast("cuda") if device.type == "cuda" else nullcontext()
    with autocast:
        prediction = model(batch.workspace, batch.wrist, batch.state, batch.tasks)
        loss = F.l1_loss(prediction, batch.action)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return loss.detach().item()


@dataclass(frozen=True)
class ValMetrics:
    l1: float
    raw_l1: float
    grip_acc: float


def peak_gpu_gb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated() / 1e9


def _grip_sign_match(pred_raw: torch.Tensor, gt_raw: torch.Tensor) -> torch.Tensor:
    pred = pred_raw[..., -1]
    gt = gt_raw[..., -1]
    closed = (pred.abs() < 0.05) & (gt.abs() < 0.05)
    return (pred.sign() == gt.sign()) | closed


def _accumulate_val(
    prediction: torch.Tensor,
    target: torch.Tensor,
    stats: NormalizeStats,
) -> tuple[float, float, float, int]:
    count = target.shape[0]
    l1 = F.l1_loss(prediction, target).item()
    pred_raw = stats.invert_action(prediction.float())
    gt_raw = stats.invert_action(target.float())
    raw = F.l1_loss(pred_raw, gt_raw).item()
    grip = _grip_sign_match(pred_raw, gt_raw).float().mean().item()
    return l1 * count, raw * count, grip * count, count


@torch.no_grad()
def evaluate_val(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int,
    stats: NormalizeStats,
) -> ValMetrics:
    model.eval()
    stats = _stats_on_device(stats, device)
    total_l1 = 0.0
    total_raw = 0.0
    total_grip = 0.0
    samples = 0
    for index, raw_batch in enumerate(loader):
        if index >= max_batches:
            break
        batch = _move_batch(raw_batch, device)
        autocast = torch.autocast("cuda") if device.type == "cuda" else nullcontext()
        with autocast:
            prediction = model(batch.workspace, batch.wrist, batch.state, batch.tasks)
            l1, raw, grip, count = _accumulate_val(prediction, batch.action, stats)
        total_l1 += l1
        total_raw += raw
        total_grip += grip
        samples += count
    denom = max(samples, 1)
    return ValMetrics(
        l1=total_l1 / denom,
        raw_l1=total_raw / denom,
        grip_acc=total_grip / denom,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    deadline: float | None = None,
    max_steps: int | None = None,
    stop: StopFlag | None = None,
    global_step: int = 0,
    save_every: int = 0,
    persist: Callable[[int], None] | None = None,
    deadline_box: list[float | None] | None = None,
    budget_seconds: float | None = None,
    schedule: Callable[[int], None] | None = None,
) -> tuple[float, float, int, int]:
    model.train()
    if hasattr(model, "text"):
        model.text.eval()
    total_loss = 0.0
    samples = 0
    steps = 0
    started = time.monotonic()
    halt = stop if stop is not None else StopFlag()
    box = deadline_box if deadline_box is not None else [deadline]
    for raw_batch in loader:
        if halt.requested:
            break
        if box[0] is not None and time.monotonic() >= box[0]:
            break
        batch = _move_batch(raw_batch, device)
        if schedule is not None:
            schedule(global_step)
        loss = _train_batch(model, batch, optimizer, scaler, device)
        if budget_seconds is not None and box[0] is None:
            box[0] = time.monotonic() + budget_seconds
        count = batch.action.shape[0]
        total_loss += loss * count
        samples += count
        steps += 1
        global_step += 1
        if persist is not None and save_every > 0 and global_step % save_every == 0:
            persist(global_step)
        if max_steps is not None and steps >= max_steps:
            break
    elapsed = max(time.monotonic() - started, 1e-9)
    return total_loss / max(samples, 1), samples / elapsed, steps, global_step


def _atomic_torch_save(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def save_checkpoint(
    path: Path,
    model: "LcAct",
    stats: NormalizeStats,
    tasks: list[str],
    epoch: int,
    step: int,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
) -> None:
    from lc_act.model import trainable_state_dict

    checkpoint = Checkpoint(
        trainable=trainable_state_dict(model),
        stats=stats,
        horizon=model.horizon,
        tasks=tasks,
        epoch=epoch,
        step=step,
        optimizer=optimizer.state_dict(),
        scaler=scaler.state_dict(),
    )
    _atomic_torch_save(path, checkpoint.to_payload())


def _stats_on_device(stats: NormalizeStats, device: torch.device) -> NormalizeStats:
    return NormalizeStats(
        state_mean=stats.state_mean.to(device),
        state_std=stats.state_std.to(device),
        action_mean=stats.action_mean.to(device),
        action_std=stats.action_std.to(device),
    )


def _model_from_checkpoint(
    checkpoint: Checkpoint,
    device: torch.device,
) -> tuple["LcAct", NormalizeStats]:
    from lc_act.model import ClipTextEncoder, LcAct, ResNetSpatial, load_trainable

    model = LcAct(
        ResNetSpatial(pretrained=False),
        ClipTextEncoder(),
        horizon=checkpoint.horizon,
    )
    load_trainable(model, checkpoint.trainable)
    return model.to(device), _stats_on_device(checkpoint.stats, device)


def load_checkpoint(path: Path, device: torch.device) -> tuple["LcAct", NormalizeStats]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return _model_from_checkpoint(Checkpoint.from_payload(payload), device)


def load_resume(
    path: Path,
    device: torch.device,
) -> tuple["LcAct", NormalizeStats, Checkpoint]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    checkpoint = Checkpoint.from_payload(payload)
    if checkpoint.optimizer is None:
        raise RuntimeError("checkpoint missing optimizer; cannot resume")
    model, stats = _model_from_checkpoint(checkpoint, device)
    return model, stats, checkpoint


def _run_training(
    args: argparse.Namespace,
    model: "LcAct",
    loader: DataLoader,
    stats: NormalizeStats,
    tasks: list[str],
    device: torch.device,
    start_epoch: int = 0,
    start_step: int = 0,
    resume: Checkpoint | None = None,
    val_loader: DataLoader | None = None,
) -> None:
    optimizer = torch.optim.AdamW(
        filter(lambda parameter: parameter.requires_grad, model.parameters()),
        lr=BASE_LR,
        fused=device.type == "cuda",
    )
    if resume is not None and resume.optimizer is not None:
        optimizer.load_state_dict(resume.optimizer)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    if resume is not None and resume.scaler is not None:
        scaler.load_state_dict(resume.scaler)
    budget = args.budget_seconds
    if budget is not None and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    deadline_box: list[float | None] = [
        None if budget else training_deadline(args.max_hours, started)
    ]
    total_steps = max(args.epochs * len(loader), 1)

    def progress(current_step: int) -> float:
        deadline = deadline_box[0]
        if budget is not None:
            return 0.0 if deadline is None else 1 - (deadline - time.monotonic()) / budget
        if deadline is not None:
            return (time.monotonic() - started) / (deadline - started)
        return current_step / total_steps

    def schedule(current_step: int) -> None:
        lr = BASE_LR * cosine_lr_scale(progress(current_step))
        for group in optimizer.param_groups:
            group["lr"] = lr
    max_steps = 1 if device.type == "cpu" else None
    stop = StopFlag()
    signal.signal(signal.SIGINT, stop.request)
    signal.signal(signal.SIGTERM, stop.request)
    step = start_step
    last = args.out / "last.pt"
    last_train = 0.0
    finished_epoch = start_epoch

    def persist(epoch: int, current_step: int) -> None:
        save_checkpoint(
            last, model, stats, tasks, epoch, current_step, optimizer, scaler,
        )

    for epoch in range(start_epoch, args.epochs):
        try:
            mean_loss, samples_per_second, steps, step = train_one_epoch(
                model,
                loader,
                optimizer,
                scaler,
                device,
                deadline=deadline_box[0],
                max_steps=max_steps,
                stop=stop,
                global_step=step,
                save_every=0 if budget else args.save_every,
                persist=None if budget else (lambda current: persist(epoch, current)),
                deadline_box=deadline_box,
                budget_seconds=budget,
                schedule=schedule,
            )
        except torch.cuda.OutOfMemoryError:
            print("drop batch to 4 or drop wrist camera")
            sys.exit(1)
        if budget is None:
            persist(epoch if stop.requested else epoch + 1, step)
        if steps == 0:
            break
        last_train = mean_loss
        finished_epoch = epoch
        print(
            f"epoch={epoch} l1={mean_loss:.4f} "
            f"samples/s={samples_per_second:.1f}"
        )
        if device.type == "cpu" or stop.requested:
            break
        if deadline_box[0] is not None and time.monotonic() >= deadline_box[0]:
            break
    if val_loader is not None:
        val = evaluate_val(model, val_loader, device, args.val_batches, stats)
        print(
            f"Step {step} : train {last_train:.4f} | val {val.l1:.4f} "
            f"| raw {val.raw_l1:.4f} | grip {val.grip_acc:.3f} "
            f"| gpu {peak_gpu_gb(device):.2f}GB"
        )
    if budget is not None:
        persist(finished_epoch, step)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and args.epochs != 1:
        raise RuntimeError("real training requires CUDA; use --epochs 1 for CPU smoke")

    from lc_act.data import load_object_dataset, split_indices_by_episode
    from lc_act.model import ClipTextEncoder, LcAct, ResNetSpatial

    torch.manual_seed(args.seed)
    dataset, data_stats, tasks = load_object_dataset(args.repo_id)
    val_loader = None
    if args.budget_seconds is not None:
        episodes = [int(ep) for ep in dataset.raw.hf_dataset["episode_index"]]
        train_idx, val_idx = split_indices_by_episode(episodes)
        train_set = torch.utils.data.Subset(dataset, train_idx)
        # Every 8th frame spans all held-out episodes; a contiguous prefix covers only one.
        val_set = torch.utils.data.Subset(dataset, val_idx[::8])
        loader = make_loader(train_set, args.batch_size, shuffle=True, workers=args.workers)
        val_loader = make_loader(val_set, args.batch_size, shuffle=False)
    else:
        loader = make_loader(dataset, args.batch_size, workers=args.workers)
    if args.resume is not None:
        model, stats, checkpoint = load_resume(args.resume, device)
        _run_training(
            args, model, loader, stats, checkpoint.tasks, device,
            start_epoch=checkpoint.epoch, start_step=checkpoint.step,
            resume=checkpoint, val_loader=val_loader,
        )
        return
    model = LcAct(ResNetSpatial(pretrained=True), ClipTextEncoder()).to(device)
    _run_training(args, model, loader, data_stats, tasks, device, val_loader=val_loader)


if __name__ == "__main__":
    main()
