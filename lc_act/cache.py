"""Decode lerobot/libero once into 128 px uint8 memmaps so training reads frames, not AV1 video."""

import argparse
import glob
import json
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torchvision.io import decode_jpeg

from lc_act.data import Sample
from lc_act.types import HORIZON, VISION_SIZE, NormalizeStats

CAMERAS = {"workspace": "observation.images.image", "wrist": "observation.images.image2"}
DEFAULT_ROOT = Path.home() / ".cache/lc_act/libero128"


def snapshot_dir(repo_id: str = "lerobot/libero") -> Path:
    org, name = repo_id.split("/")
    pattern = Path.home() / f".cache/huggingface/lerobot/hub/datasets--{org}--{name}/snapshots/*/"
    found = glob.glob(str(pattern))
    if not found:
        raise RuntimeError(f"{repo_id} is not in the HF cache; load it once with LeRobotDataset")
    return Path(found[0])


def resize_uint8(frames: torch.Tensor) -> torch.Tensor:
    """(N, H, W, 3) uint8 -> (N, 128, 128, 3) uint8, with the same filter the model uses at eval."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = frames.to(device).permute(0, 3, 1, 2).float() / 255.0
    x = F.interpolate(x, size=VISION_SIZE, mode="bilinear", antialias=True)
    return (x * 255).round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu()


def _decode_file(job: tuple[str, str, Path, list[tuple[int, int, int]], int]) -> int:
    import av

    camera, video, out, spans, total = job
    torch.set_num_threads(1)
    target = np.memmap(out / f"{camera}.u8", np.uint8, "r+", shape=(total, VISION_SIZE, VISION_SIZE, 3))
    # spans: (first frame in file, first global row, length) per episode stored in this file
    lookup = {}
    for start, row, length in spans:
        for k in range(length):
            lookup[start + k] = row + k
    written = 0
    rows: list[int] = []
    frames: list[np.ndarray] = []

    def flush() -> None:
        nonlocal written
        target[rows] = resize_uint8(torch.from_numpy(np.stack(frames))).numpy()
        written += len(rows)
        rows.clear()
        frames.clear()

    with av.open(video) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for frame in container.decode(stream):
            index = round(float(frame.time) * 10)
            if index in lookup:
                rows.append(lookup.pop(index))
                frames.append(frame.to_ndarray(format="rgb24"))
                if len(rows) == 256:
                    flush()
    if rows:
        flush()
    target.flush()
    if lookup:
        raise RuntimeError(f"{video}: {len(lookup)} frames missing after decode")
    return written


def build(repo_id: str, out: Path, workers: int) -> None:
    root = snapshot_dir(repo_id)
    out.mkdir(parents=True, exist_ok=True)
    data = pd.concat(
        [pd.read_parquet(f) for f in sorted(glob.glob(str(root / "data/*/*.parquet")))]
    ).sort_values("index")
    total = len(data)
    np.save(out / "state.npy", np.stack(data["observation.state"].to_numpy()).astype(np.float32))
    np.save(out / "action.npy", np.stack(data["action"].to_numpy()).astype(np.float32))
    for column in ("episode_index", "task_index", "frame_index"):
        np.save(out / f"{column}.npy", data[column].to_numpy().astype(np.int64))
    tasks = pd.read_parquet(root / "meta/tasks.parquet")
    names = {int(index): str(task) for task, index in zip(tasks.index, tasks["task_index"])}
    (out / "tasks.json").write_text(json.dumps([names[i] for i in range(len(names))], indent=1))

    episodes = pd.concat(
        [pd.read_parquet(f) for f in sorted(glob.glob(str(root / "meta/episodes/*/*.parquet")))]
    )
    jobs = []
    for camera, key in CAMERAS.items():
        np.memmap(out / f"{camera}.u8", np.uint8, "w+", shape=(total, VISION_SIZE, VISION_SIZE, 3)).flush()
        by_file: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
        for record in episodes.to_dict("records"):
            file_key = (int(record[f"videos/{key}/chunk_index"]), int(record[f"videos/{key}/file_index"]))
            start = round(float(record[f"videos/{key}/from_timestamp"]) * 10)
            by_file.setdefault(file_key, []).append(
                (start, int(record["dataset_from_index"]), int(record["length"]))
            )
        for (chunk, file), spans in by_file.items():
            video = str(root / f"videos/{key}/chunk-{chunk:03d}/file-{file:03d}.mp4")
            jobs.append((camera, video, out, spans, total))
    # Workers resize on the GPU, so they must not inherit a CUDA context.
    with get_context("spawn").Pool(workers) as pool:
        done = 0
        for written in pool.imap_unordered(_decode_file, jobs):
            done += written
            print(f"decoded {done}/{2 * total} frames", flush=True)
    if done != 2 * total:
        raise RuntimeError(f"decoded {done} frames, expected {2 * total}")
    pack(out)
    (out / "meta.json").write_text(json.dumps({"rows": total, "size": VISION_SIZE}))


def pack(root: Path, quality: int = 95, batch: int = 1024) -> None:
    """Re-encode the raw frames as JPEG: ~3.5 GB stays in page cache, the 27 GB raw memmap does not.

    q95 differs from the raw frames by ~1/255 on average, well under the dataset's own AV1 loss."""
    from torchvision.io import encode_jpeg

    total = int(np.load(root / "task_index.npy").shape[0])
    for camera in CAMERAS:
        raw = np.memmap(root / f"{camera}.u8", np.uint8, "r", shape=(total, VISION_SIZE, VISION_SIZE, 3))
        offsets = [0]
        with open(root / f"{camera}.jpg.bin", "wb") as blob:
            for start in range(0, total, batch):
                x = torch.from_numpy(np.array(raw[start:start + batch])).permute(0, 3, 1, 2).to("cuda" if torch.cuda.is_available() else "cpu")
                for encoded in encode_jpeg(list(x), quality=quality):
                    data = encoded.cpu().numpy().tobytes()
                    blob.write(data)
                    offsets.append(offsets[-1] + len(data))
        np.save(root / f"{camera}.jpg.offsets.npy", np.array(offsets, dtype=np.int64))


SUITES = {"libero_10": range(0, 10), "libero_goal": range(10, 20), "libero_object": range(20, 30), "libero_spatial": range(30, 40)}


def suite_of(task_index: int) -> str:
    return next(name for name, ids in SUITES.items() if task_index in ids)


class CachedLibero(torch.utils.data.Dataset):
    """Rows of the frame cache; history frames clamp to the episode start, chunks pad with its last action."""

    def __init__(
        self,
        root: Path,
        rows: np.ndarray,
        stats: NormalizeStats,
        tasks: list[str],
        n_obs: int = 1,
        horizon: int = HORIZON,
        teacher: bool = False,
        obs_stride: int = 1,
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
        self.total = len(self.task_index)
        self._images: dict[str, tuple[np.memmap, np.ndarray]] = {}
        self.teacher = np.load(root / "teacher.npy", mmap_mode="r") if teacher else None

    def __len__(self) -> int:
        return len(self.rows)

    def _frames(self, name: str, rows: np.ndarray) -> torch.Tensor:
        # Opened lazily so each loader worker maps the file itself.
        if name not in self._images:
            self._images[name] = (
                np.memmap(self.root / f"{name}.jpg.bin", np.uint8, "r"),
                np.load(self.root / f"{name}.jpg.offsets.npy"),
            )
        blob, offsets = self._images[name]
        return torch.stack([
            decode_jpeg(torch.from_numpy(np.array(blob[offsets[r]:offsets[r + 1]]))).permute(1, 2, 0)
            for r in rows
        ])

    def __getitem__(self, index: int) -> Sample:
        row = int(self.rows[index])
        offsets = self.obs_stride * np.arange(-(self.n_obs - 1), 1)
        history = np.maximum(row + offsets, self.first[row])
        chunk = np.minimum(np.arange(row, row + self.horizon), self.last[row])
        return Sample(
            workspace=self._frames("workspace", history),
            wrist=self._frames("wrist", history),
            state=self.stats.apply_state(self.state[history]),
            action=self.stats.apply_action(self.action[chunk]),
            task=self.tasks[self.task_index[row]],
            # (now, one chunk ahead) x (workspace, wrist) teacher features
            teacher=None if self.teacher is None else torch.from_numpy(
                self.teacher[[row, chunk[-1]]].astype(np.float32)
            ),
        )


def episode_bounds(episode: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """First and last row of each row's episode; rows of one episode are contiguous."""
    starts = np.flatnonzero(np.r_[True, episode[1:] != episode[:-1]])
    ends = np.r_[starts[1:], len(episode)] - 1
    owner = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, len(episode)]))
    return starts[owner], ends[owner]


def split_by_task(episode: np.ndarray, task: np.ndarray, val_frac: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """Hold out the last `val_frac` of each task's episodes, so every suite is in val."""
    val_eps = set()
    for t in np.unique(task):
        eps = np.unique(episode[task == t])
        val_eps.update(eps[-max(1, round(len(eps) * val_frac)):].tolist())
    is_val = np.isin(episode, list(val_eps))
    return np.flatnonzero(~is_val), np.flatnonzero(is_val)


def load_cached(
    root: Path = DEFAULT_ROOT,
    all_tasks: bool = True,
    n_obs: int = 1,
    holdout: bool = False,
    teacher: bool = False,
    obs_stride: int = 1,
) -> tuple[CachedLibero, CachedLibero | None, NormalizeStats, list[str]]:
    if not (root / "meta.json").exists():
        raise RuntimeError(f"no frame cache at {root}; run python -m lc_act.cache")
    tasks = json.loads((root / "tasks.json").read_text())
    task = np.load(root / "task_index.npy")
    episode = np.load(root / "episode_index.npy")
    keep = np.ones(len(task), bool) if all_tasks else np.isin(task, list(SUITES["libero_object"]))
    if holdout:
        train_rows, val_rows = split_by_task(episode[keep], task[keep])
        train_rows, val_rows = np.flatnonzero(keep)[train_rows], np.flatnonzero(keep)[val_rows]
    else:
        train_rows, val_rows = np.flatnonzero(keep), None
    _, last = episode_bounds(episode)
    state = torch.from_numpy(np.load(root / "state.npy"))[train_rows]
    chunks = np.minimum(train_rows[:, None] + np.arange(HORIZON), last[train_rows, None])
    actions = torch.from_numpy(np.load(root / "action.npy"))[chunks]
    stats = NormalizeStats.from_tensors(state, actions)
    train = CachedLibero(root, train_rows, stats, tasks, n_obs, teacher=teacher, obs_stride=obs_stride)
    # Every 8th held-out frame: spread over all val episodes without paying for every frame.
    val = None if val_rows is None else CachedLibero(root, val_rows[::8], stats, tasks, n_obs, obs_stride=obs_stride)
    return train, val, stats, tasks


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache LIBERO frames at 128 px")
    parser.add_argument("--repo-id", default="lerobot/libero")
    parser.add_argument("--out", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--pack-only", action="store_true", help="re-encode an existing raw cache as JPEG")
    args = parser.parse_args()
    if args.pack_only:
        pack(args.out)
    else:
        build(args.repo_id, args.out, args.workers)


if __name__ == "__main__":
    main()
