import torch
from torch import nn

from lc_act.models import lc_act as _lc_act
from lc_act.models import text as _text
from lc_act.models import vision as _vision

TASK = "pick up the alphabet soup and place it in the basket"


class FakeVision(nn.Module):
    def encode(self, images: torch.Tensor, film: list[torch.Tensor], group: int = 1) -> torch.Tensor:
        return torch.zeros(images.shape[0], 4, _vision.FEATURE_DIM, device=images.device)


class FakeText(nn.Module):
    def encode(self, texts: list[str]) -> torch.Tensor:
        return torch.zeros(len(texts), _text.TEXT_DIM)


def make_model(**overrides) -> _lc_act.LcAct:
    return _lc_act.LcAct(_lc_act.LcActConfig(**overrides), FakeVision(), FakeText())


def _frames(b: int, t: int) -> torch.Tensor:
    return torch.zeros(b, t, 256, 256, 3, dtype=torch.uint8)


def test_forward_shape_is_batch_horizon_action():
    model = make_model()
    out = model(_frames(2, 1), _frames(2, 1), torch.zeros(2, 1, 8), [TASK] * 2)
    assert out.shape == (2, _lc_act.HORIZON, _lc_act.ACTION_DIM)


def test_two_frame_history_forward_shape():
    model = make_model(n_obs=2)
    actions, log_scale = model(_frames(3, 2), _frames(3, 2), torch.zeros(3, 2, 8), [TASK] * 3, with_scale=True)
    assert actions.shape == log_scale.shape == (3, _lc_act.HORIZON, _lc_act.ACTION_DIM)


def test_history_size_reaches_the_oldest_strided_frame():
    assert _lc_act.LcActConfig().history_size == 1
    assert _lc_act.LcActConfig(n_obs=2, obs_stride=8).history_size == 9


def test_teacher_prediction_is_two_times_two_cameras():
    model = make_model()
    memory = model.encode(_frames(2, 1), _frames(2, 1), torch.zeros(2, 1, 8), [TASK] * 2)
    assert model.teacher_prediction(memory).shape == (2, 2, 2, _lc_act.TEACHER_DIM)


def test_workspace_and_wrist_tokens_differ_when_images_match():
    model = make_model()
    captured: dict[str, torch.Tensor] = {}

    def hook(_module, inputs, _output) -> None:
        captured["tokens"] = inputs[0].detach()

    model.encoder.register_forward_hook(hook)
    images = _frames(1, 1)
    model(images, images, torch.zeros(1, 1, 8), [TASK])

    workspace, wrist = captured["tokens"][:, :4], captured["tokens"][:, 4:8]
    assert not torch.allclose(workspace, wrist)
