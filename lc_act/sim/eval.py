"""Roll out a checkpoint in one LIBERO-Object task and save a video of the first episode.

python -m lc_act.sim.eval --ckpt outputs/lc_act/last.pt
"""

from collections.abc import Callable
import dataclasses
import pathlib
from typing import Any

import imageio
import numpy as np
import torch
import tyro

from lc_act.shared import normalize as _normalize
from lc_act.sim import env as _env
from lc_act.sim import policy as _policy
from lc_act.training import checkpoints as _checkpoints

Frame = tuple[np.ndarray, np.ndarray, np.ndarray]  # workspace, wrist, raw state from env.observe
TASK = "pick up the alphabet soup and place it in the basket"  # LIBERO-Object task 0


@dataclasses.dataclass(frozen=True)
class Args:
    ckpt: pathlib.Path
    episodes: int = 10
    seed: int = 0  # episode i is reset with seed + i
    video: pathlib.Path = pathlib.Path("outputs/lc_act/soup_ep0.mp4")


def main(args: Args) -> None:
    if args.episodes < 1:
        raise ValueError("--episodes must be positive")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, stats = _checkpoints.load_checkpoint(args.ckpt, device)
    model.eval()

    env = _env.make_env()
    successes, frames = 0, []
    try:
        for episode in range(args.episodes):
            record = (lambda frame: frames.append(frame[0])) if episode == 0 else None
            successes += int(run_episode(model, stats, env, device, args.seed + episode, on_frame=record))
            if episode == 0:
                _save_video(args.video, frames)
    finally:
        env.close()
    print(f"successes={successes}/{args.episodes}")


def run_episode(
    model: torch.nn.Module,
    stats: _normalize.NormStats,
    env: Any,
    device: torch.device,
    seed: int,
    *,
    task: str = TASK,
    on_frame: Callable[[Frame], None] | None = None,
    stop: Callable[[], bool] | None = None,
) -> bool:
    """One episode, replanning every env step; returns success.

    `on_frame` sees every observed (workspace, wrist, state) frame, `stop` is polled before each step.
    """
    notify = on_frame or (lambda _frame: None)
    obs, _ = env.reset(seed=seed)
    frame = _env.observe(obs)
    notify(frame)

    ensembler = _policy.ChunkEnsembler(model.config)
    with torch.inference_mode():
        # One dataset row is one control step (its 10 fps is a label only), so actions run at env rate.
        for _ in range(_env.max_steps(env)):
            if stop is not None and stop():
                return False
            workspace, wrist, state = ensembler.observe(_policy.to_tensors(frame, stats, device))
            chunk = stats.unnormalize_action(model(workspace[None], wrist[None], state[None], [task])[0])
            action = ensembler.act(chunk)

            try:
                obs, _, terminated, truncated, info = env.step(action.cpu().numpy())
            except ValueError as error:
                if "executing action in terminated episode" not in str(error):
                    raise
                return False
            frame = _env.observe(obs)
            notify(frame)
            success = bool(info.get("is_success", False))
            if terminated or truncated or success:
                return success
    return False


def _save_video(path: pathlib.Path, frames: list[np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, fps=_env.ENV_FPS)


if __name__ == "__main__":
    main(tyro.cli(Args))
