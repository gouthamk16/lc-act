"""Ensemble action chunks for every cached frame, computed once; train.py --distill regresses one model onto them."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from lc_act.bench import ModelEnsemble
from lc_act.cache import DEFAULT_ROOT, CachedLibero
from lc_act.data import collate
from lc_act.train import _stats_on_device, load_checkpoint
from lc_act.types import ACTION_DIM


@torch.no_grad()
def build(ckpts: list[Path], root: Path, out: Path, batch: int, workers: int) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded = [load_checkpoint(path, device) for path in ckpts]
    stats = loaded[0][1]
    if any(not torch.allclose(s.action_mean, stats.action_mean) for _, s in loaded):
        raise ValueError("ensemble members were normalized differently; train them on the same split")
    model = ModelEnsemble([m for m, _ in loaded]).eval()
    first = loaded[0][0]
    rows = np.arange(np.load(root / "task_index.npy").shape[0])
    tasks = json.loads((root / "tasks.json").read_text())
    # Loader workers normalize on the CPU; the model's copy of the stats lives on the GPU.
    cpu_stats = _stats_on_device(stats, torch.device("cpu"))
    data = CachedLibero(root, rows, cpu_stats, tasks, first.n_obs, horizon=first.horizon, obs_stride=first.obs_stride)
    loader = DataLoader(data, batch_size=batch, num_workers=workers, collate_fn=collate)
    result = np.lib.format.open_memmap(out, "w+", np.float16, (len(rows), first.horizon, ACTION_DIM))
    start = 0
    for item in loader:
        chunks = model(item.workspace.to(device), item.wrist.to(device), item.state.to(device), item.tasks)
        result[start:start + len(chunks)] = stats.invert_action(chunks).cpu().numpy()
        start += len(chunks)
        if start % (batch * 200) < batch:
            print(f"{start}/{len(rows)}", flush=True)
    result.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache ensemble action chunks for distillation")
    parser.add_argument("--ckpt", type=Path, nargs="+", required=True)
    parser.add_argument("--cache", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    build(args.ckpt, args.cache, args.out, args.batch, args.workers)


if __name__ == "__main__":
    main()
