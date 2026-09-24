import torch
from torch import nn

from lc_act.model import LcAct, _random_shift, freeze_module, trainable_state_dict
from lc_act.types import ACTION_DIM, CLIP_DIM, HORIZON, IMAGE_SIZE, RESNET_DIM

TASK = "pick up the alphabet soup and place it in the basket"


class FakeVision(nn.Module):
    def encode(self, images: torch.Tensor, size: int = 0, group: int = 1) -> torch.Tensor:
        return torch.zeros(images.shape[0], 4, RESNET_DIM, device=images.device)


class FakeText(nn.Module):
    def encode(self, texts: list[str]) -> torch.Tensor:
        return torch.zeros(len(texts), CLIP_DIM)


def _frames(b: int, t: int) -> torch.Tensor:
    return torch.zeros(b, t, IMAGE_SIZE, IMAGE_SIZE, 3, dtype=torch.uint8)


def test_forward_shape_is_batch_horizon_action():
    model = LcAct(FakeVision(), FakeText())
    b = 2
    out = model(_frames(b, 1), _frames(b, 1), torch.zeros(b, 1, 8), [TASK] * b)
    assert out.shape == (b, HORIZON, ACTION_DIM)


def test_two_frame_history_forward_shape():
    model = LcAct(FakeVision(), FakeText(), n_obs=2)
    b = 3
    out = model(_frames(b, 2), _frames(b, 2), torch.zeros(b, 2, 8), [TASK] * b)
    assert out.shape == (b, HORIZON, ACTION_DIM)


def test_random_shift_uses_one_offset_per_frame_stack():
    torch.manual_seed(0)
    sample = torch.rand(4, 3, 16, 16)
    stacked = sample.repeat_interleave(2, dim=0)
    shifted = _random_shift(stacked, pad=4, group=2)
    assert torch.equal(shifted[0::2], shifted[1::2])


def test_freeze_module_sets_requires_grad_false():
    linear = nn.Linear(4, 4)
    assert linear.weight.requires_grad
    freeze_module(linear)
    assert not linear.weight.requires_grad


def test_workspace_and_wrist_tokens_differ_when_images_match():
    model = LcAct(FakeVision(), FakeText())
    captured: dict[str, torch.Tensor] = {}

    def _hook(_module, inputs, _output) -> None:
        captured["tokens"] = inputs[0].detach()

    model.encoder.register_forward_hook(_hook)
    images = _frames(1, 1)
    model(images, images, torch.zeros(1, 1, 8), [TASK])
    tokens = captured["tokens"]
    workspace, wrist = tokens[:, :4], tokens[:, 4:8]
    assert not torch.allclose(workspace, wrist)


def test_trainable_state_dict_skips_text_keys():
    class Text(FakeText):
        def __init__(self):
            super().__init__()
            self.dummy = nn.Parameter(torch.ones(CLIP_DIM))

    model = LcAct(FakeVision(), Text())
    keys = trainable_state_dict(model)
    assert all(not key.startswith("text.") for key in keys)
    assert any(key.startswith("action_head.") for key in keys)
