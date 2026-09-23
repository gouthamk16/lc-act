from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from lc_act.obs import flip_hw, pack_state
from lc_act.train import load_checkpoint
from lc_act.types import ENV_FPS, NormalizeStats

TASK = "pick up the alphabet soup and place it in the basket"


def _ensure_libero_config() -> None:
    config_dir = Path(os.environ.get("LIBERO_CONFIG_PATH", Path.home() / ".libero"))
    config_file = config_dir / "config.yaml"
    if config_file.exists():
        return
    from importlib.util import find_spec

    spec = find_spec("libero.libero")
    if spec is None or spec.origin is None:
        raise RuntimeError("LIBERO package is not installed")
    root = Path(spec.origin).parent
    config_dir.mkdir(parents=True, exist_ok=True)
    config_file.write_text(
        "\n".join(
            [
                f"benchmark_root: {root}",
                f"bddl_files: {root / 'bddl_files'}",
                f"init_states: {root / 'init_files'}",
                f"datasets: {root / 'datasets'}",
                f"assets: {root / 'assets'}",
            ]
        )
        + "\n"
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate LC-ACT in LIBERO-Object")
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--video",
        type=Path,
        default=Path("outputs/lc_act/soup_ep0.mp4"),
    )
    return parser.parse_args(argv)


def make_env() -> Any:
    os.environ.setdefault("MUJOCO_GL", "egl")
    _ensure_libero_config()
    try:
        from libero.libero import benchmark
        from lerobot.envs.libero import LiberoEnv

        suite = benchmark.get_benchmark_dict()["libero_object"]()
        return LiberoEnv(
            task_suite=suite,
            task_id=0,
            task_suite_name="libero_object",
            episode_length=280,
            obs_type="pixels_agent_pos",
            control_mode="relative",
            observation_width=256,
            observation_height=256,
            init_states=True,
        )
    except Exception as error:
        message = f"{type(error).__name__}: {error}".lower()
        if "egl" in message or "opengl" in message:
            print("MUJOCO_GL=egl failed; try export MUJOCO_GL=osmesa")
        raise


def _image_tensor(image: torch.Tensor | np.ndarray, device: torch.device) -> torch.Tensor:
    if isinstance(image, np.ndarray):
        image = np.ascontiguousarray(image)
    return torch.as_tensor(image).to(device).unsqueeze(0)


def obs_to_tensors(
    obs: dict[str, Any],
    stats: NormalizeStats,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    workspace = _image_tensor(flip_hw(obs["pixels"]["image"]), device)
    wrist = _image_tensor(flip_hw(obs["pixels"]["image2"]), device)
    state = stats.apply_state(pack_state(obs).to(device)).unsqueeze(0)
    return workspace, wrist, state


def _reset_obs(result: Any) -> dict[str, Any]:
    if isinstance(result, tuple) and len(result) == 2:
        return result[0]
    return result


def _frame_from_obs(obs: dict[str, Any], env: Any) -> np.ndarray:
    if "pixels" in obs and "image" in obs["pixels"]:
        frame = flip_hw(obs["pixels"]["image"])
        if isinstance(frame, torch.Tensor):
            frame = frame.detach().cpu().numpy()
        return np.asarray(frame).copy()
    return np.asarray(env.render()).copy()


def _step_values(result: Any) -> tuple[dict[str, Any], bool, bool, dict[str, Any]]:
    if len(result) == 4:
        obs, _reward, done, info = result
        return obs, bool(done), False, info
    obs, _reward, terminated, truncated, info = result
    return obs, bool(terminated), bool(truncated), info


def run_episode(
    model: torch.nn.Module,
    stats: NormalizeStats,
    env: Any,
    device: torch.device,
    seed: int,
    record: bool = False,
) -> tuple[bool, list[np.ndarray]]:
    obs = _reset_obs(env.reset(seed=seed))
    frames = [_frame_from_obs(obs, env)] if record else []
    success = False
    with torch.inference_mode():
        while True:
            workspace, wrist, state = obs_to_tensors(obs, stats, device)
            actions = stats.invert_action(model(workspace, wrist, state, [TASK])[0])
            # One dataset row is one env control step (the dataset's 10 fps is a label only).
            for action in actions:
                try:
                    result = env.step(action.detach().cpu().numpy())
                except ValueError as error:
                    if "executing action in terminated episode" not in str(error):
                        raise
                    return success, frames
                obs, terminated, truncated, info = _step_values(result)
                if record:
                    frames.append(_frame_from_obs(obs, env))
                success = bool(info.get("is_success", False))
                if terminated or truncated or success:
                    return success, frames
    return success, frames


def _save_video(path: Path, frames: list[np.ndarray]) -> None:
    import imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, fps=ENV_FPS)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.episodes < 1:
        raise ValueError("--episodes must be positive")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, stats = load_checkpoint(args.ckpt, device)
    model.eval()
    env = make_env()
    successes = 0
    try:
        for episode in range(args.episodes):
            success, frames = run_episode(
                model, stats, env, device, args.seed + episode, record=episode == 0
            )
            successes += int(success)
            if episode == 0:
                _save_video(args.video, frames)
    finally:
        if hasattr(env, "close"):
            env.close()
    print(f"successes={successes}/{args.episodes}")


if __name__ == "__main__":
    main()
