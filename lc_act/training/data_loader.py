"""Training samples from the 128 px LIBERO frame cache built by `python -m lc_act.training.cache`."""

import dataclasses
import json
import pathlib

import numpy as np
import torch
from torchvision.io import decode_jpeg

from lc_act.models import lc_act as _lc_act
from lc_act.shared import normalize as _normalize

DEFAULT_ROOT = pathlib.Path.home() / ".cache/lc_act/libero128"
# cache file stem -> lerobot/libero video key
CAMERAS = {"workspace": "observation.images.image", "wrist": "observation.images.image2"}
# lerobot/libero task indices of each suite
SUITES = {
    "libero_10": range(10),
    "libero_goal": range(10, 20),
    "libero_object": range(20, 30),
    "libero_spatial": range(30, 40),
}


def suite_of(task_index: int) -> str:
    return next(name for name, ids in SUITES.items() if task_index in ids)


@dataclasses.dataclass
class Sample:
    """One training row: (t, h, w, 3) uint8 frames and (t, 8) state oldest first, a (horizon, 7) action chunk."""

    workspace: torch.Tensor
    wrist: torch.Tensor
    state: torch.Tensor
    action: torch.Tensor
    task: str
    # (2 times, 2 cameras, TEACHER_DIM) DINOv2 targets: this frame and the frame one chunk ahead
    teacher: torch.Tensor | None = None


@dataclasses.dataclass
class Batch:
    """Stacked samples; every tensor gains a leading batch dimension."""

    workspace: torch.Tensor
    wrist: torch.Tensor
    state: torch.Tensor
    action: torch.Tensor
    tasks: list[str]
    teacher: torch.Tensor | None = None

    def to(self, device: torch.device) -> "Batch":
        moved = {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}
        return Batch(**{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in moved.items()})


def collate(samples: list[Sample]) -> Batch:
    return Batch(
        workspace=torch.stack([sample.workspace for sample in samples]),
        wrist=torch.stack([sample.wrist for sample in samples]),
        state=torch.stack([sample.state for sample in samples]),
        action=torch.stack([sample.action for sample in samples]),
        tasks=[sample.task for sample in samples],
        teacher=None if samples[0].teacher is None else torch.stack([sample.teacher for sample in samples]),
    )


class CachedLibero(torch.utils.data.Dataset):
    """Rows of the frame cache; history frames clamp to the episode start, chunks pad with its last action."""

    def __init__(
        self,
        root: pathlib.Path,
        rows: np.ndarray,
        stats: _normalize.NormStats,
        tasks: list[str],
        *,
        n_obs: int = 1,
        obs_stride: int = 1,
        horizon: int = _lc_act.HORIZON,
        teacher: bool = False,
    ) -> None:
        self.root = root
        self.rows = rows
        self.stats = stats
        self.tasks = tasks
        self.n_obs = n_obs
        self.obs_stride = obs_stride
        self.horizon = horizon

        self.state = torch.from_numpy(np.load(root / "state.npy"))
        self.action = torch.from_numpy(np.load(root / "action.npy"))
        self.task_index = np.load(root / "task_index.npy")
        self.first, self.last = episode_bounds(np.load(root / "episode_index.npy"))
        self.teacher = np.load(root / "teacher.npy", mmap_mode="r") if teacher else None
        self._images: dict[str, tuple[np.memmap, np.ndarray]] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Sample:
        row = int(self.rows[index])
        history = np.maximum(row + self.obs_stride * np.arange(-(self.n_obs - 1), 1), self.first[row])
        chunk = np.minimum(np.arange(row, row + self.horizon), self.last[row])

        teacher = None
        if self.teacher is not None:
            teacher = torch.from_numpy(self.teacher[[row, chunk[-1]]].astype(np.float32))

        return Sample(
            workspace=self._frames("workspace", history),
            wrist=self._frames("wrist", history),
            state=self.stats.normalize_state(self.state[history]),
            action=self.stats.normalize_action(self.action[chunk]),
            task=self.tasks[self.task_index[row]],
            teacher=teacher,
        )

    def _frames(self, camera: str, rows: np.ndarray) -> torch.Tensor:
        # Opened lazily so each loader worker maps the file itself.
        if camera not in self._images:
            self._images[camera] = (
                np.memmap(self.root / f"{camera}.jpg.bin", np.uint8, "r"),
                np.load(self.root / f"{camera}.jpg.offsets.npy"),
            )
        blob, offsets = self._images[camera]
        jpegs = [torch.from_numpy(np.array(blob[offsets[r] : offsets[r + 1]])) for r in rows]
        return torch.stack([decode_jpeg(jpeg).permute(1, 2, 0) for jpeg in jpegs])


def load_cached(
    root: pathlib.Path = DEFAULT_ROOT,
    *,
    object_only: bool = False,
    n_obs: int = 1,
    obs_stride: int = 1,
    holdout: bool = False,
    teacher: bool = False,
) -> tuple[CachedLibero, CachedLibero | None, _normalize.NormStats, list[str]]:
    """Train and (with `holdout`) val sets, the train split's normalization stats, and the task strings."""
    if not (root / "meta.json").exists():
        raise RuntimeError(f"no frame cache at {root}; run python -m lc_act.training.cache")
    tasks = json.loads((root / "tasks.json").read_text())
    task = np.load(root / "task_index.npy")
    episode = np.load(root / "episode_index.npy")

    # pick rows, then hold out the last episodes of every task
    keep = np.isin(task, SUITES["libero_object"]) if object_only else np.ones(len(task), bool)
    kept = np.flatnonzero(keep)
    if holdout:
        train_rows, val_rows = split_by_task(episode[keep], task[keep])
        train_rows, val_rows = kept[train_rows], kept[val_rows]
    else:
        train_rows, val_rows = kept, None

    # stats over the train rows' states and full action chunks
    _, last = episode_bounds(episode)
    state = torch.from_numpy(np.load(root / "state.npy"))[train_rows]
    chunks = np.minimum(train_rows[:, None] + np.arange(_lc_act.HORIZON), last[train_rows, None])
    actions = torch.from_numpy(np.load(root / "action.npy"))[chunks]
    stats = _normalize.NormStats.from_tensors(state, actions)

    train = CachedLibero(root, train_rows, stats, tasks, n_obs=n_obs, obs_stride=obs_stride, teacher=teacher)
    # Every 8th held-out frame: spread over all val episodes without paying for every frame.
    val = None
    if val_rows is not None:
        val = CachedLibero(root, val_rows[::8], stats, tasks, n_obs=n_obs, obs_stride=obs_stride)
    return train, val, stats, tasks


def episode_bounds(episode: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """First and last row of each row's episode; rows of one episode are contiguous."""
    starts = np.flatnonzero(np.r_[True, episode[1:] != episode[:-1]])
    ends = np.r_[starts[1:], len(episode)] - 1
    owner = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, len(episode)]))
    return starts[owner], ends[owner]


def split_by_task(episode: np.ndarray, task: np.ndarray, val_frac: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """Hold out the last `val_frac` of each task's episodes, so every suite is in val."""
    val_episodes = set()
    for t in np.unique(task):
        episodes = np.unique(episode[task == t])
        val_episodes.update(episodes[-max(1, round(len(episodes) * val_frac)) :].tolist())
    is_val = np.isin(episode, list(val_episodes))
    return np.flatnonzero(~is_val), np.flatnonzero(is_val)
