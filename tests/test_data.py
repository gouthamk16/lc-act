import torch

from lc_act.data import (
    _task_to_index,
    action_delta_timestamps,
    ObjectVLADataset,
    collate,
    is_object_task,
    object_task_indices,
    pad_action_chunk,
)
from lc_act.types import NormalizeStats


def test_object_filter_keeps_basket_picks_only():
    keep = "pick up the alphabet soup and place it in the basket"
    drop_spatial = (
        "pick up the black bowl between the plate and the ramekin and place it on the plate"
    )
    drop_long = "put both the alphabet soup and the cream cheese box in the basket"
    assert is_object_task(keep)
    assert not is_object_task(drop_spatial)
    assert not is_object_task(drop_long)


def test_object_task_indices_uses_strings_not_hub_order():
    mapping = {
        "pick up the alphabet soup and place it in the basket": 17,
        "pick up the black bowl and place it on the plate": 3,
    }
    assert object_task_indices(mapping) == {17}


def test_pad_action_chunk_repeats_last_row():
    actions = torch.tensor([[1.0, 0, 0, 0, 0, 0, -1], [2.0, 0, 0, 0, 0, 0, 1]])
    chunk = pad_action_chunk(actions, start=0, horizon=16)
    assert chunk.shape == (16, 7)
    assert torch.equal(chunk[0], actions[0])
    assert torch.equal(chunk[1], actions[1])
    assert torch.equal(chunk[-1], actions[-1])


def test_delta_timestamps_are_16_steps_at_10hz():
    deltas = action_delta_timestamps()
    assert deltas == [i * 0.1 for i in range(16)]


class _FakeRaw(list):
    pass


class _FakeTaskTable:
    index = [
        "pick up the alphabet soup and place it in the basket",
        "pick up the black bowl and place it on the plate",
    ]
    loc = {
        index[0]: type("TaskRow", (), {"task_index": 17})(),
        index[1]: type("TaskRow", (), {"task_index": 3})(),
    }


class _FakeMeta:
    tasks = _FakeTaskTable()


def test_task_to_index_reads_task_dataframe_index():
    assert _task_to_index(_FakeMeta()) == {
        "pick up the alphabet soup and place it in the basket": 17,
        "pick up the black bowl and place it on the plate": 3,
    }


def test_dataset_getitem_normalizes_and_collate_batches():
    action = torch.ones(16, 7)
    state = torch.ones(8)
    image = torch.zeros(256, 256, 3, dtype=torch.uint8)
    raw = _FakeRaw([
        {
            "observation.images.image": image,
            "observation.images.image2": image,
            "observation.state": state,
            "action": action,
            "task": "pick up the alphabet soup and place it in the basket",
        }
    ])
    stats = NormalizeStats.from_tensors(state.unsqueeze(0), action.unsqueeze(0))
    ds = ObjectVLADataset(raw, stats)
    batch = collate([ds[0], ds[0]])
    assert batch.workspace.shape == (2, 256, 256, 3)
    assert batch.state.shape == (2, 8)
    assert batch.action.shape == (2, 16, 7)
    assert torch.allclose(batch.action, torch.zeros_like(batch.action), atol=1e-5)
