import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

from lc_act.types import ACTION_DIM, DATASET_FPS, HORIZON, STATE_DIM, Batch, NormalizeStats

_OBJECT_TASK = re.compile(
    r"^pick up the .+ and place it in the basket$",
    re.IGNORECASE,
)


def is_object_task(instruction: str) -> bool:
    return bool(_OBJECT_TASK.fullmatch(instruction.strip()))


def object_task_indices(task_to_index: dict[str, int]) -> set[int]:
    return {idx for task, idx in task_to_index.items() if is_object_task(task)}


def task_indices(task_to_index: dict[str, int], all_tasks: bool) -> set[int]:
    if all_tasks:
        return set(task_to_index.values())
    return object_task_indices(task_to_index)


def split_indices_by_episode(
    episode_index: list[int],
    val_frac: float = 0.1,
) -> tuple[list[int], list[int]]:
    unique = sorted(set(episode_index))
    n_val = max(1, int(round(len(unique) * val_frac)))
    val_eps = set(unique[-n_val:])
    train_idx: list[int] = []
    val_idx: list[int] = []
    for i, episode in enumerate(episode_index):
        if episode in val_eps:
            val_idx.append(i)
        else:
            train_idx.append(i)
    if not train_idx or not val_idx:
        raise ValueError("episode split produced an empty train or val set")
    return train_idx, val_idx


def pad_action_chunk(actions: torch.Tensor, start: int, horizon: int) -> torch.Tensor:
    if actions.ndim != 2 or actions.shape[-1] != ACTION_DIM:
        raise ValueError(f"expected (T, {ACTION_DIM}), got {tuple(actions.shape)}")
    end = min(start + horizon, actions.shape[0])
    piece = actions[start:end]
    if piece.shape[0] == 0:
        raise ValueError(f"start {start} is past episode length {actions.shape[0]}")
    if piece.shape[0] < horizon:
        pad = piece[-1:].repeat(horizon - piece.shape[0], 1)
        piece = torch.cat([piece, pad], dim=0)
    return piece


def action_delta_timestamps(horizon: int = HORIZON, fps: int = DATASET_FPS) -> list[float]:
    return [i * (1.0 / fps) for i in range(horizon)]


def obs_delta_timestamps(n_obs: int, fps: int = DATASET_FPS) -> list[float]:
    return [round(-(n_obs - 1 - i) / fps, 6) for i in range(n_obs)]


@dataclass
class Sample:
    workspace: torch.Tensor
    wrist: torch.Tensor
    state: torch.Tensor
    action: torch.Tensor
    task: str


def _frames_uint8(frames: Any) -> torch.Tensor:
    """(T, H, W, 3) uint8, oldest frame first; a single (H, W, 3) or (3, H, W) frame gets T=1."""
    frames = torch.as_tensor(frames)
    if frames.ndim == 3:
        frames = frames.unsqueeze(0)
    return torch.stack([_hwc_uint8(frame) for frame in frames])


def _hwc_uint8(image: Any) -> torch.Tensor:
    image = torch.as_tensor(image)
    if image.ndim != 3:
        raise ValueError(f"expected 3D image, got {tuple(image.shape)}")
    if image.shape[-1] != 3:
        if image.shape[0] != 3:
            raise ValueError(f"expected RGB image, got {tuple(image.shape)}")
        image = image.permute(1, 2, 0)
    if image.dtype != torch.uint8:
        image = image.float()
        if image.numel() and image.max() <= 1:
            image = image * 255
        image = image.clamp(0, 255).to(torch.uint8)
    return image.contiguous()


def _action_chunk(action: Any, horizon: int) -> torch.Tensor:
    action = torch.as_tensor(action)
    if action.ndim == 1:
        return pad_action_chunk(action.unsqueeze(0), 0, horizon)
    if action.shape[0] < horizon:
        return pad_action_chunk(action, 0, horizon)
    return action


class ObjectVLADataset(torch.utils.data.Dataset):
    def __init__(self, raw: Any, stats: NormalizeStats, horizon: int = HORIZON) -> None:
        self.raw = raw
        self.stats = stats
        self.horizon = horizon

    def __len__(self) -> int:
        return len(self.raw)

    def __getitem__(self, index: int) -> Sample:
        row = self.raw[index]
        task = row["task"] if isinstance(row["task"], str) else row["task"][0]
        return Sample(
            workspace=_frames_uint8(row["observation.images.image"]),
            wrist=_frames_uint8(row["observation.images.image2"]),
            state=self.stats.apply_state(
                torch.as_tensor(row["observation.state"]).float().reshape(-1, STATE_DIM)
            ),
            action=self.stats.apply_action(
                _action_chunk(row["action"], self.horizon).float()
            ),
            task=task,
        )


def collate(samples: list[Sample]) -> Batch:
    return Batch(
        workspace=torch.stack([sample.workspace for sample in samples]),
        wrist=torch.stack([sample.wrist for sample in samples]),
        state=torch.stack([sample.state for sample in samples]),
        action=torch.stack([sample.action for sample in samples]),
        tasks=[sample.task for sample in samples],
    )


def _task_to_index(meta: Any) -> dict[str, int]:
    tasks = meta.tasks
    if isinstance(tasks, Mapping):
        if "task" in tasks and "task_index" in tasks:
            rows = zip(tasks["task"], tasks["task_index"])
        else:
            rows = (
                (task, index) if isinstance(task, str) else (index, task)
                for task, index in tasks.items()
            )
    elif hasattr(tasks, "index") and hasattr(tasks, "loc"):
        rows = ((task, tasks.loc[task].task_index) for task in tasks.index)
    elif hasattr(tasks, "to_dict"):
        rows = (
            (row["task"], row["task_index"])
            for row in tasks.to_dict("records")
        )
    else:
        rows = ((row["task"], row["task_index"]) for row in tasks)
    return {str(task): int(index) for task, index in rows}


def _is_pyav_error(error: Exception) -> bool:
    message = f"{type(error).__name__}: {error}".lower()
    return (
        "pyav" in message
        or "no module named 'av'" in message
        or ("video backend" in message and "av" in message)
    )


def _raw_object_subset(
    raw: Any,
    allowed: set[int],
) -> tuple[list[int], list[torch.Tensor], list[torch.Tensor]]:
    episodes = set()
    states = []
    actions = []
    columns = raw.hf_dataset
    rows = zip(
        columns["observation.state"],
        columns["action"],
        columns["episode_index"],
        columns["task_index"],
    )
    for state, action, episode, task in rows:
        if int(task) not in allowed:
            continue
        episodes.add(int(episode))
        states.append(torch.as_tensor(state).float())
        actions.append(_action_chunk(action, HORIZON).float())
    return sorted(episodes), states, actions


def load_libero_dataset(
    repo_id: str = "lerobot/libero",
    all_tasks: bool = False,
    n_obs: int = 1,
) -> tuple[ObjectVLADataset, NormalizeStats, list[str]]:
    """Object-suite subset by default; all_tasks trains on every LIBERO suite."""
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

        meta = LeRobotDatasetMetadata(repo_id)
    except Exception as error:
        if _is_pyav_error(error):
            raise RuntimeError("install PyAV: pip install av") from error
        raise

    mapping = _task_to_index(meta)
    allowed = task_indices(mapping, all_tasks)
    if not allowed:
        raise RuntimeError(f"no matching tasks in {repo_id}")
    delta_timestamps = {"action": action_delta_timestamps()}
    if n_obs > 1:
        for key in ("observation.images.image", "observation.images.image2", "observation.state"):
            delta_timestamps[key] = obs_delta_timestamps(n_obs)

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        raw_metadata = LeRobotDataset(
            repo_id,
            download_videos=False,
        )
        episodes, states, actions = _raw_object_subset(raw_metadata, allowed)
        if not episodes:
            raise RuntimeError(f"no Object-suite episodes in {repo_id}")
        raw = LeRobotDataset(
            repo_id,
            episodes=episodes,
            delta_timestamps=delta_timestamps,
            video_backend="pyav",
        )
    except Exception as error:
        if _is_pyav_error(error):
            raise RuntimeError("install PyAV: pip install av") from error
        raise

    if not states:
        raise RuntimeError(f"no samples in Object-suite subset of {repo_id}")
    stats = NormalizeStats.from_tensors(torch.stack(states), torch.stack(actions))
    tasks = [task for task, index in mapping.items() if index in allowed]
    return ObjectVLADataset(raw, stats), stats, tasks
