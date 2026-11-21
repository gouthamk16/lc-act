import dataclasses

import torch
import torch.nn.functional as F  # noqa: N812

from lc_act.shared import normalize as _normalize


@dataclasses.dataclass(frozen=True)
class ValMetrics:
    l1: float  # z-scored action L1
    raw_l1: float  # action L1 in env units
    grip_acc: float  # gripper sign match in env units
    per_suite: dict[str, float]


@torch.no_grad()
def evaluate_val(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    max_batches: int,
    stats: _normalize.NormStats,
    suite_of_task: dict[str, str] | None = None,
) -> ValMetrics:
    """Sample-weighted metrics over the first `max_batches` batches, optionally broken down by suite."""
    model.eval()
    stats = stats.to(device)
    totals = {"l1": 0.0, "raw_l1": 0.0, "grip_acc": 0.0}
    suite_sum: dict[str, float] = {}
    suite_count: dict[str, int] = {}
    samples = 0

    for index, cpu_batch in enumerate(loader):
        if index >= max_batches:
            break
        batch = cpu_batch.to(device)
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            prediction = model(batch.workspace, batch.wrist, batch.state, batch.tasks)
            count = batch.action.shape[0]
            pred_raw = stats.unnormalize_action(prediction.float())
            gt_raw = stats.unnormalize_action(batch.action.float())
            totals["l1"] += F.l1_loss(prediction, batch.action).item() * count
            totals["raw_l1"] += F.l1_loss(pred_raw, gt_raw).item() * count
            totals["grip_acc"] += _grip_sign_match(pred_raw, gt_raw).float().mean().item() * count
        samples += count

        if suite_of_task is not None:
            per_sample = (prediction.float() - batch.action).abs().mean((1, 2)).tolist()
            for task, value in zip(batch.tasks, per_sample, strict=True):
                suite = suite_of_task[task]
                suite_sum[suite] = suite_sum.get(suite, 0.0) + value
                suite_count[suite] = suite_count.get(suite, 0) + 1

    denom = max(samples, 1)
    per_suite = {suite: suite_sum[suite] / suite_count[suite] for suite in sorted(suite_sum)}
    return ValMetrics(**{key: value / denom for key, value in totals.items()}, per_suite=per_suite)


def _grip_sign_match(pred_raw: torch.Tensor, gt_raw: torch.Tensor) -> torch.Tensor:
    """The gripper command matches in sign, or both are near zero."""
    pred, gt = pred_raw[..., -1], gt_raw[..., -1]
    closed = (pred.abs() < 0.05) & (gt.abs() < 0.05)
    return (pred.sign() == gt.sign()) | closed
