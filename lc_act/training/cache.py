"""Decode lerobot/libero once into 128 px frames so training reads JPEGs, not AV1 video.

python -m lc_act.training.cache
"""

import dataclasses
import glob
import json
import multiprocessing
import pathlib

import av
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F  # noqa: N812
from torchvision.io import encode_jpeg
import tyro

from lc_act.models import vision as _vision
from lc_act.training import data_loader as _data

SIZE = _vision.VISION_SIZE
DATASET_FPS = 10  # lerobot/libero timestamps; frame index = round(time * fps)

# (camera, video file, cache dir, spans, total rows); a span is (first frame in file, first global row, length)
DecodeJob = tuple[str, str, pathlib.Path, list[tuple[int, int, int]], int]


@dataclasses.dataclass(frozen=True)
class Args:
    repo_id: str = "lerobot/libero"
    out: pathlib.Path = _data.DEFAULT_ROOT
    workers: int = 10  # video decode processes
    pack_only: bool = False  # re-encode an existing raw cache as JPEG


def main(args: Args) -> None:
    if args.pack_only:
        pack(args.out)
    else:
        build(args.repo_id, args.out, args.workers)


def build(repo_id: str, out: pathlib.Path, workers: int) -> None:
    """Write state/action/index arrays and tasks.json, decode both cameras to raw memmaps, then pack as JPEG."""
    root = snapshot_dir(repo_id)
    out.mkdir(parents=True, exist_ok=True)
    total = _save_tables(root, out)

    # one decode job per (camera, video file), spread over a process pool
    episodes = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(root / "meta/episodes/*/*.parquet")))])
    jobs = []
    for camera, key in _data.CAMERAS.items():
        np.memmap(out / f"{camera}.u8", np.uint8, "w+", shape=(total, SIZE, SIZE, 3)).flush()
        jobs += _decode_jobs(camera, key, episodes, root, out, total)

    # Workers resize on the GPU, so they must not inherit a CUDA context.
    with multiprocessing.get_context("spawn").Pool(workers) as pool:
        done = 0
        for written in pool.imap_unordered(_decode_file, jobs):
            done += written
            print(f"decoded {done}/{2 * total} frames", flush=True)
    if done != 2 * total:
        raise RuntimeError(f"decoded {done} frames, expected {2 * total}")

    pack(out)
    (out / "meta.json").write_text(json.dumps({"rows": total, "size": SIZE}))


def pack(root: pathlib.Path, quality: int = 95, batch: int = 1024) -> None:
    """Re-encode the raw frames as JPEG: ~3.5 GB stays in page cache, the 27 GB raw memmap does not.

    q95 differs from the raw frames by ~1/255 on average, well under the dataset's own AV1 loss.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    total = int(np.load(root / "task_index.npy").shape[0])
    for camera in _data.CAMERAS:
        raw = np.memmap(root / f"{camera}.u8", np.uint8, "r", shape=(total, SIZE, SIZE, 3))
        offsets = [0]
        with open(root / f"{camera}.jpg.bin", "wb") as blob:
            for start in range(0, total, batch):
                x = torch.from_numpy(np.array(raw[start : start + batch])).permute(0, 3, 1, 2).to(device)
                for encoded in encode_jpeg(list(x), quality=quality):
                    data = encoded.cpu().numpy().tobytes()
                    blob.write(data)
                    offsets.append(offsets[-1] + len(data))
        np.save(root / f"{camera}.jpg.offsets.npy", np.array(offsets, dtype=np.int64))


def snapshot_dir(repo_id: str = "lerobot/libero") -> pathlib.Path:
    org, name = repo_id.split("/")
    pattern = pathlib.Path.home() / f".cache/huggingface/lerobot/hub/datasets--{org}--{name}/snapshots/*/"
    found = glob.glob(str(pattern))
    if not found:
        raise RuntimeError(f"{repo_id} is not in the HF cache; load it once with LeRobotDataset")
    return pathlib.Path(found[0])


def resize_uint8(frames: torch.Tensor) -> torch.Tensor:
    """(n, h, w, 3) uint8 -> (n, 128, 128, 3) uint8, with the same filter the model uses at eval."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = frames.to(device).permute(0, 3, 1, 2).float() / 255.0
    x = F.interpolate(x, size=SIZE, mode="bilinear", antialias=True)
    return (x * 255).round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu()


def _save_tables(root: pathlib.Path, out: pathlib.Path) -> int:
    """Per-row state, action and indices in dataset order, plus tasks.json; returns the row count."""
    parquets = sorted(glob.glob(str(root / "data/*/*.parquet")))
    data = pd.concat([pd.read_parquet(f) for f in parquets]).sort_values("index")
    np.save(out / "state.npy", np.stack(data["observation.state"].to_numpy()).astype(np.float32))
    np.save(out / "action.npy", np.stack(data["action"].to_numpy()).astype(np.float32))
    for column in ("episode_index", "task_index", "frame_index"):
        np.save(out / f"{column}.npy", data[column].to_numpy().astype(np.int64))

    tasks = pd.read_parquet(root / "meta/tasks.parquet")
    names = {int(index): str(task) for task, index in zip(tasks.index, tasks["task_index"], strict=True)}
    (out / "tasks.json").write_text(json.dumps([names[i] for i in range(len(names))], indent=1))
    return len(data)


def _decode_jobs(
    camera: str, key: str, episodes: pd.DataFrame, root: pathlib.Path, out: pathlib.Path, total: int
) -> list[DecodeJob]:
    """Group one camera's episodes by the video file that stores them."""
    by_file: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
    for record in episodes.to_dict("records"):
        file_key = (int(record[f"videos/{key}/chunk_index"]), int(record[f"videos/{key}/file_index"]))
        start = round(float(record[f"videos/{key}/from_timestamp"]) * DATASET_FPS)
        by_file.setdefault(file_key, []).append((start, int(record["dataset_from_index"]), int(record["length"])))

    jobs = []
    for (chunk, file), spans in by_file.items():
        video = str(root / f"videos/{key}/chunk-{chunk:03d}/file-{file:03d}.mp4")
        jobs.append((camera, video, out, spans, total))
    return jobs


def _decode_file(job: DecodeJob) -> int:
    """Decode one video file, resize, and write its frames to their global rows; returns frames written."""
    camera, video, out, spans, total = job
    torch.set_num_threads(1)
    target = np.memmap(out / f"{camera}.u8", np.uint8, "r+", shape=(total, SIZE, SIZE, 3))
    lookup = {start + k: row + k for start, row, length in spans for k in range(length)}

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
            index = round(float(frame.time) * DATASET_FPS)
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


if __name__ == "__main__":
    main(tyro.cli(Args))
