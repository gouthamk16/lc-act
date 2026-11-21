"""LIBERO benchmark: every task of each suite at init states first_init..first_init+episodes-1, envs stepped in
parallel subprocesses, one batched policy.

    python -m lc_act.sim.bench --ckpt outputs/lc_act/last.pt --episodes 50
"""

import collections
from collections.abc import Callable
import dataclasses
import json
import multiprocessing
from multiprocessing.connection import Connection
import pathlib
import time
import typing
from typing import Literal

import numpy as np
import torch
import tyro

from lc_act.shared import normalize as _normalize
from lc_act.sim import env as _env
from lc_act.sim import policy as _policy
from lc_act.training import checkpoints as _checkpoints

Suite = Literal["libero_spatial", "libero_object", "libero_goal", "libero_10"]
SUITE_NAMES: tuple[Suite, ...] = typing.get_args(Suite)
TASKS_PER_SUITE = 10


@dataclasses.dataclass(frozen=True)
class Args:
    ckpt: tuple[pathlib.Path, ...]  # several: average their chunks
    suites: tuple[Suite, ...] = SUITE_NAMES
    tasks: tuple[str, ...] = ()  # only these, as suite/task_id (e.g. libero_10/8)
    episodes: int = 50  # init states per task
    first_init: int = 0  # start of the init range, to split a bench across runs
    envs: int = 2  # each LIBERO env holds ~2.9 GB (MuJoCo arena)
    ensemble: Literal["decay", "uncertainty"] = "decay"
    out: pathlib.Path | None = None  # full summary JSON, with per-task rates


@dataclasses.dataclass(frozen=True)
class Job:
    suite: str
    task_id: int
    init: int


@dataclasses.dataclass
class Episode:
    job: Job
    instruction: str
    max_steps: int
    ensembler: _policy.ChunkEnsembler


def main(args: Args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, stats = load_policy(args.ckpt, device)

    started = time.monotonic()
    results = run_benchmark(
        model, stats, pick_jobs(args), args.envs, device, uncertainty=args.ensemble == "uncertainty"
    )
    summary = summarize(results)
    summary["minutes"] = round((time.monotonic() - started) / 60, 1)

    print(json.dumps({key: value for key, value in summary.items() if key != "tasks"}, indent=1))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=1))


def load_policy(paths: tuple[pathlib.Path, ...], device: torch.device) -> tuple[torch.nn.Module, _normalize.NormStats]:
    """One checkpoint, or a chunk-averaging ensemble of several trained on the same split."""
    loaded = [_checkpoints.load_checkpoint(path, device) for path in paths]
    model, stats = loaded[0]
    if len(loaded) == 1:
        return model, stats

    # Members act in one normalized space, so they must share statistics (same training split).
    if any(not torch.allclose(s.action_mean, stats.action_mean) for _, s in loaded):
        raise ValueError("ensemble members were normalized differently; train them on the same split")
    # The CLIP text encoder is frozen and identical in every member; keep one copy on the GPU.
    for member, _ in loaded[1:]:
        member.text = model.text
    return _policy.ModelEnsemble([member for member, _ in loaded]), stats


def pick_jobs(args: Args) -> list[Job]:
    picked = [(suite, task) for suite in args.suites for task in range(TASKS_PER_SUITE)]
    if args.tasks:
        picked = [(suite, int(task)) for suite, task in (name.split("/") for name in args.tasks)]
    inits = range(args.first_init, args.first_init + args.episodes)
    return [Job(suite, task, init) for suite, task in picked for init in inits]


def run_benchmark(
    model: torch.nn.Module,
    stats: _normalize.NormStats,
    jobs: list[Job],
    n_envs: int,
    device: torch.device,
    *,
    uncertainty: bool = False,
    log: Callable[[str], None] = print,
) -> dict[Job, tuple[bool, int]]:
    """(success, steps) per job."""
    model.eval()
    pipes, procs = _spawn_workers(min(n_envs, len(jobs)))
    queues = _deal(jobs, len(pipes))
    episodes: dict[int, Episode] = {}
    observations: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    results: dict[Job, tuple[bool, int]] = {}

    def start(i: int) -> None:
        if not queues[i]:
            return
        job = queues[i].popleft()
        pipes[i].send(("reset", job))
        observations[i], instruction, max_steps = pipes[i].recv()
        episodes[i] = Episode(job, instruction, max_steps, _policy.ChunkEnsembler(model.config))

    try:
        for i in range(len(pipes)):
            start(i)

        # every running env takes one step per batched forward pass
        while observations:
            actions = _act(model, stats, episodes, observations, device, uncertainty=uncertainty)
            for i, action in actions.items():
                pipes[i].send(("step", action))
            for i in actions:
                obs, done, success = pipes[i].recv()
                episode, steps = episodes[i], episodes[i].ensembler.step
                if not (success or done or steps >= episode.max_steps):
                    observations[i] = obs
                    continue
                results[episode.job] = (success, steps)
                job, verdict = episode.job, "ok" if success else "fail"
                log(
                    f"{job.suite} task {job.task_id} init {job.init}: {verdict} in {steps} ({len(results)}/{len(jobs)})"
                )
                del episodes[i], observations[i]
                start(i)
    finally:
        for pipe in pipes:
            pipe.send(("close", None))
        for proc in procs:
            proc.join(timeout=30)
    return results


def summarize(results: dict[Job, tuple[bool, int]]) -> dict:
    """Success rate per suite, their average, and per task."""
    by_suite: dict[str, list[bool]] = collections.defaultdict(list)
    by_task: dict[str, list[bool]] = collections.defaultdict(list)
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


def _spawn_workers(n: int) -> tuple[list[Connection], list[multiprocessing.Process]]:
    """`n` env subprocesses; spawned, so none inherits the parent's CUDA context."""
    context = multiprocessing.get_context("spawn")
    pipes, procs = [], []
    for _ in range(n):
        parent, child = context.Pipe()
        proc = context.Process(target=_worker, args=(child,), daemon=True)
        proc.start()
        pipes.append(parent)
        procs.append(proc)
    return pipes, procs


def _deal(jobs: list[Job], n_workers: int) -> list[collections.deque[Job]]:
    """Whole tasks dealt round-robin: a worker rebuilds its env only when the task changes, and long-suite tasks
    spread over all workers instead of landing on the last one."""
    tasks: dict[tuple[str, int], list[Job]] = collections.defaultdict(list)
    for job in jobs:
        tasks[(job.suite, job.task_id)].append(job)
    queues: list[collections.deque[Job]] = [collections.deque() for _ in range(n_workers)]
    for index, group in enumerate(tasks.values()):
        queues[index % n_workers].extend(group)
    return queues


def _act(
    model: torch.nn.Module,
    stats: _normalize.NormStats,
    episodes: dict[int, Episode],
    observations: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
    device: torch.device,
    *,
    uncertainty: bool,
) -> dict[int, np.ndarray]:
    """One batched forward over every running env; returns each env's ensembled action."""
    ids = list(observations)
    inputs = [episodes[i].ensembler.observe(_policy.to_tensors(observations[i], stats, device)) for i in ids]
    workspace, wrist, state = (torch.stack(parts) for parts in zip(*inputs, strict=True))

    with torch.inference_mode():
        instructions = [episodes[i].instruction for i in ids]
        chunks, log_scales = model(workspace, wrist, state, instructions, with_scale=True)
        chunks = stats.unnormalize_action(chunks)

    actions = {}
    for i, chunk, log_scale in zip(ids, chunks, log_scales, strict=True):
        action = episodes[i].ensembler.act(chunk, log_scale if uncertainty else None)
        actions[i] = action.cpu().numpy()
    return actions


def _worker(conn: Connection) -> None:
    """Env subprocess: ("reset", Job) -> (frame, instruction, max steps); ("step", action) -> (frame, done, success)."""
    env, key = None, None
    while True:
        command, payload = conn.recv()
        if command == "close":
            break

        if command == "reset":
            job: Job = payload
            # rebuild only when the task changes; a new init state is just a reset
            if key != (job.suite, job.task_id):
                if env is not None:
                    env.close()
                env = _env.make_env(job.task_id, job.suite)
                key = (job.suite, job.task_id)
            env.init_state_id = job.init
            obs, _ = env.reset(seed=job.init)
            conn.send((_env.observe(obs), env.task_description, _env.max_steps(env)))
            continue

        try:
            obs, _, terminated, truncated, info = env.step(payload)
        except ValueError as error:
            if "terminated episode" not in str(error):
                raise
            conn.send((None, True, False))
            continue
        conn.send((_env.observe(obs), terminated or truncated, bool(info.get("is_success", False))))

    if env is not None:
        env.close()


if __name__ == "__main__":
    main(tyro.cli(Args))
