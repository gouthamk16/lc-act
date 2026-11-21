import dataclasses
import json

import numpy as np
import torch

from lc_act.models import vision as _vision
from lc_act.training import cache as _cache
from lc_act.training import data_loader as _data


def test_episode_bounds_per_row():
    first, last = _data.episode_bounds(np.array([0, 0, 0, 1, 1, 2]))
    assert first.tolist() == [0, 0, 0, 3, 3, 5]
    assert last.tolist() == [2, 2, 2, 4, 4, 5]


def test_split_by_task_holds_out_last_episode_of_every_task():
    episode = np.array([0, 0, 1, 2, 3, 3, 4, 5])
    task = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    train, val = _data.split_by_task(episode, task, val_frac=0.3)
    assert val.tolist() == [3, 7]
    assert train.tolist() == [0, 1, 2, 4, 5, 6]


def test_suite_of_matches_dataset_task_order():
    assert _data.suite_of(0) == "libero_10"
    assert _data.suite_of(24) == "libero_object"
    assert _data.suite_of(39) == "libero_spatial"


def test_cached_sample_clamps_history_and_pads_chunk_within_episode(tmp_path):
    _fake_cache(tmp_path, [0, 0, 0, 1, 1, 1])
    train, _, stats, _ = _data.load_cached(tmp_path, n_obs=2)
    data = _data.CachedLibero(tmp_path, np.array([3, 5]), stats, train.tasks, n_obs=2, horizon=4)

    start = data[0]
    assert torch.allclose(start.workspace[:, 0, 0, 0].float(), torch.tensor([30.0, 30.0]), atol=2)
    assert start.task == "task 24"

    end = data[1]
    assert torch.allclose(end.wrist[:, 0, 0, 0].float(), torch.tensor([40.0, 50.0]), atol=2)
    raw = stats.unnormalize_action(end.action)
    assert torch.allclose(raw[:, 0], torch.tensor([35.0, 35.0, 35.0, 35.0]))


def test_load_cached_holdout_keeps_val_out_of_train(tmp_path):
    _fake_cache(tmp_path, [0] * 8 + [1] * 8)
    train, val, _, _ = _data.load_cached(tmp_path, holdout=True)
    assert set(train.rows.tolist()).isdisjoint(val.rows.tolist())
    assert val.rows.tolist() == [8]


def test_strided_history_reaches_back_and_clamps_at_episode_start(tmp_path):
    _fake_cache(tmp_path, [0] * 12)
    train, _, stats, _ = _data.load_cached(tmp_path, n_obs=2)
    data = _data.CachedLibero(tmp_path, np.array([3, 11]), stats, train.tasks, n_obs=2, obs_stride=8)
    assert torch.allclose(data[0].workspace[:, 0, 0, 0].float(), torch.tensor([0.0, 30.0]), atol=2)
    assert torch.allclose(data[1].workspace[:, 0, 0, 0].float(), torch.tensor([30.0, 110.0]), atol=2)


def test_collate_stacks_samples_and_keeps_task_order():
    sample = _data.Sample(torch.zeros(1, 8, 8, 3), torch.zeros(1, 8, 8, 3), torch.zeros(1, 8), torch.zeros(16, 7), "a")
    batch = _data.collate([sample, dataclasses.replace(sample, task="b")])
    assert batch.workspace.shape == (2, 1, 8, 8, 3)
    assert batch.tasks == ["a", "b"]
    assert batch.teacher is None


def _fake_cache(root, episodes):
    """A cache of `len(episodes)` rows whose frame r is filled with 10 * r."""
    rows = len(episodes)
    size = _vision.VISION_SIZE
    for camera in _data.CAMERAS:
        images = np.memmap(root / f"{camera}.u8", np.uint8, "w+", shape=(rows, size, size, 3))
        images[:] = (np.arange(rows, dtype=np.uint8) * 10)[:, None, None, None]
        images.flush()
    np.save(root / "state.npy", np.arange(rows * 8, dtype=np.float32).reshape(rows, 8))
    np.save(root / "action.npy", np.arange(rows * 7, dtype=np.float32).reshape(rows, 7))
    np.save(root / "episode_index.npy", np.array(episodes))
    np.save(root / "task_index.npy", np.array([24] * rows))
    (root / "tasks.json").write_text(json.dumps([f"task {i}" for i in range(40)]))
    _cache.pack(root)
    (root / "meta.json").write_text(json.dumps({"rows": rows}))
