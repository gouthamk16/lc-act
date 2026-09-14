import torch
from torch import nn

from lc_act.model import LcAct, freeze_module, trainable_state_dict
from lc_act.types import ACTION_DIM, CLIP_DIM, HORIZON, IMAGE_SIZE, RESNET_DIM


class FakeVision(nn.Module):
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return torch.zeros(images.shape[0], 4, RESNET_DIM, device=images.device)


class FakeText(nn.Module):
    def encode(self, texts: list[str]) -> torch.Tensor:
        return torch.zeros(len(texts), CLIP_DIM)


def test_forward_shape_is_batch_horizon_action():
    model = LcAct(FakeVision(), FakeText())
    b = 2
    workspace = torch.zeros(b, 3, IMAGE_SIZE, IMAGE_SIZE)
    wrist = torch.zeros(b, 3, IMAGE_SIZE, IMAGE_SIZE)
    state = torch.zeros(b, 8)
    out = model(workspace, wrist, state, ["pick up the alphabet soup and place it in the basket"] * b)
    assert out.shape == (b, HORIZON, ACTION_DIM)


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
    images = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    state = torch.zeros(1, 8)
    model(images, images, state, ["pick up the alphabet soup and place it in the basket"])
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
