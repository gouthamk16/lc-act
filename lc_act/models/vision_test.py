import torch

from lc_act.models import vision as _vision


def test_random_shift_uses_one_offset_per_frame_stack():
    torch.manual_seed(0)
    stacked = torch.rand(4, 3, 16, 16).repeat_interleave(2, dim=0)
    shifted = _vision._random_shift(stacked, pad=4, group=2)
    assert torch.equal(shifted[0::2], shifted[1::2])


def test_resnet_spatial_is_an_8x8_grid_at_128px():
    film = [torch.zeros(2, 2 * c) for c in _vision.STAGE_CHANNELS]
    tokens = _vision.ResNetSpatial(pretrained=False).eval().encode(torch.zeros(2, 256, 256, 3, dtype=torch.uint8), film)
    assert tokens.shape == (2, 64, _vision.FEATURE_DIM)
