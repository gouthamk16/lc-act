"""LIBERO benchmark: every task of each suite, init states 0..N-1, envs stepped in parallel, one batched policy."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from lc_act.eval import ENSEMBLE_DECAY, temporal_ensemble
from lc_act.obs import flip_hw, pack_state

SUITE_NAMES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


@dataclass(frozen=True)
class Job:
    suite: str
    task_id: int
    init: int


@dataclass
class Episode:
    job: Job
    instruction: str
    max_steps: int
    history: deque = field(default_factory=deque)
    pending: defaultdict = field(default_factory=lambda: defaultdict(list))
    scales: defaultdict = field(default_factory=lambda: defaultdict(list))
    step: int = 0


def _observe(obs: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    workspace = np.ascontiguousarray(flip_hw(np.asarray(obs["pixels"]["image"])))
    wrist = np.ascontiguousarray(flip_hw(np.asarray(obs["pixels"]["image2"])))
    return workspace, wrist, pack_state(obs).numpy()


def _worker(conn) -> None:
    # Under WSL2, EGL falls back to llvmpipe (~42 ms per camera); GLFW through Mesa's D3D12 driver renders on the GPU.
    # Its frames match llvmpipe's to 0.5/255 on average.
    if "WSL_DISTRO_NAME" in os.environ:
        os.environ.update(MUJOCO_GL="glfw", GALLIUM_DRIVER="d3d12", MESA_D3D12_DEFAULT_ADAPTER_NAME="NVIDIA")
    os.environ.setdefault("MUJOCO_GL", "egl")
    from lc_act.eval import _ensure_libero_config

    _ensure_libero_config()
    from libero.libero import benchmark
    from lerobot.envs.libero import LiberoEnv

    env, key = None, None
    while True:
        command, payload = conn.recv()
        if command == "close":
            break
        if command == "reset":
            job: Job = payload
            if key != (job.suite, job.task_id):
                if env is not None:
                    env.close()
                suite = benchmark.get_benchmark_dict()[job.suite]()
                env = LiberoEnv(
                    task_suite=suite, task_id=job.task_id, task_suite_name=job.suite,
                    obs_type="pixels_agent_pos", control_mode="relative",
                    observation_width=256, observation_height=256, init_states=True,
                )
                key = (job.suite, job.task_id)
            env.init_state_id = job.init
            obs, _ = env.reset(seed=job.init)
            conn.send((_observe(obs), env.task_description, env._max_episode_steps))
        else:
            try:
                obs, _, terminated, truncated, info = env.step(payload)
            except ValueError as error:
                if "terminated episode" not in str(error):
                    raise
                conn.send((None, True, False))
                continue
            conn.send((_observe(obs), terminated or truncated, bool(info.get("is_success", False))))
    if env is not None:
        env.close()


def _act(
    model, stats, episodes: dict[int, Episode], observations: dict[int, tuple], device, uncertainty: bool,
) -> dict[int, np.ndarray]:
    ids = list(observations)
    for i in ids:
        episode = episodes[i]
        workspace, wrist, state = observations[i]
        current = (
            torch.from_numpy(workspace), torch.from_numpy(wrist),
            stats.apply_state(torch.from_numpy(state).to(device)).cpu(),
        )
        episode.history.extend([current] * (model.n_obs if not episode.history else 1))
    frames = [list(zip(*episodes[i].history)) for i in ids]
    workspace, wrist, state = (
        torch.stack([torch.stack(f[k]) for f in frames]).to(device) for k in range(3)
    )
    with torch.inference_mode():
        chunks, log_scales = model(
            workspace, wrist, state, [episodes[i].instruction for i in ids], with_scale=True,
        )
        chunks = stats.invert_action(chunks)
    actions = {}
    for i, chunk, log_scale in zip(ids, chunks, log_scales):
        episode = episodes[i]
        for offset, action in enumerate(chunk):
            episode.pending[episode.step + offset].append(action)
            episode.scales[episode.step + offset].append(log_scale[offset])
        scales = episode.scales.pop(episode.step)
        ensembled = temporal_ensemble(
            episode.pending.pop(episode.step), ENSEMBLE_DECAY, scales if uncertainty else None,
        )
        actions[i] = ensembled.cpu().numpy()
    return actions


def run_benchmark(
    model, stats, jobs: list[Job], n_envs: int, device: torch.device, uncertainty: bool = False, log=print,
) -> dict[Job, tuple[bool, int]]:
    ctx = mp.get_context("spawn")
    pipes, procs = [], []
    for _ in range(min(n_envs, len(jobs))):
        parent, child = ctx.Pipe()
        proc = ctx.Process(target=_worker, args=(child,), daemon=True)
        proc.start()
        pipes.append(parent)
        procs.append(proc)
    # Contiguous blocks per worker, so a worker rebuilds its env only when the task changes.
    queues = [deque(block) for block in np.array_split(np.array(jobs, dtype=object), len(pipes))]
    episodes: dict[int, Episode] = {}
    observations: dict[int, tuple] = {}
    results: dict[Job, tuple[bool, int]] = {}
    model.eval()

    def start(i: int) -> None:
        if not queues[i]:
            return
        job = queues[i].popleft()
        pipes[i].send(("reset", job))
        obs, instruction, max_steps = pipes[i].recv()
        episodes[i] = Episode(job, instruction, max_steps, deque(maxlen=model.n_obs))
        observations[i] = obs

    try:
        for i in range(len(pipes)):
            start(i)
        while observations:
            actions = _act(model, stats, episodes, observations, device, uncertainty)
            for i, action in actions.items():
                pipes[i].send(("step", action))
            for i in list(actions):
                obs, done, success = pipes[i].recv()
                episode = episodes[i]
                episode.step += 1
                if success or done or episode.step >= episode.max_steps:
                    results[episode.job] = (success, episode.step)
                    log(f"{episode.job.suite} task {episode.job.task_id} init {episode.job.init}: "
                        f"{'ok' if success else 'fail'} in {episode.step} ({len(results)}/{len(jobs)})")
                    del episodes[i], observations[i]
                    start(i)
                else:
                    observations[i] = obs
    finally:
        for pipe in pipes:
            pipe.send(("close", None))
        for proc in procs:
            proc.join(timeout=30)
    return results


def summarize(results: dict[Job, tuple[bool, int]]) -> dict:
    by_suite: dict[str, list[bool]] = defaultdict(list)
    by_task: dict[str, list[bool]] = defaultdict(list)
    for job, (success, _) in results.items():
        by_suite[job.suite].append(success)
        by_task[f"{job.suite}/{job.task_id}"].append(success)
    suites = {name: float(np.mean(v)) for name, v in by_suite.items()}
    return {
        "suites": suites,
        "average": float(np.mean(list(suites.values()))),
        "tasks": {name: float(np.mean(v)) for name, v in sorted(by_task.items())},
        "episodes": len(results),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark LC-ACT on LIBERO suites")
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--suites", nargs="+", default=list(SUITE_NAMES), choices=SUITE_NAMES)
    parser.add_argument("--episodes", type=int, default=50, help="init states per task")
    parser.add_argument("--envs", type=int, default=2, help="each LIBERO env holds ~2.9 GB (MuJoCo arena)")
    parser.add_argument("--ensemble", choices=("decay", "uncertainty"), default="decay")
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    from lc_act.train import load_checkpoint

    args = parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, stats = load_checkpoint(args.ckpt, device)
    jobs = [Job(suite, task, init) for suite in args.suites for task in range(10) for init in range(args.episodes)]
    started = time.monotonic()
    results = run_benchmark(model, stats, jobs, args.envs, device, args.ensemble == "uncertainty")
    summary = summarize(results)
    summary["minutes"] = round((time.monotonic() - started) / 60, 1)
    print(json.dumps({k: v for k, v in summary.items() if k != "tasks"}, indent=1))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
