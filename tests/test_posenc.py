import pytest
import torch

from lc_act.posenc import grid_hw, sinusoidal_2d


def test_grid_hw_rejects_non_square_token_count():
    with pytest.raises(ValueError, match="square grid"):
        grid_hw(5)


def test_sinusoidal_2d_shape_is_one_by_cells_by_dim():
    pos = sinusoidal_2d(2, 2, 8, torch.device("cpu"), torch.float32)
    assert pos.shape == (1, 4, 8)


def test_sinusoidal_2d_differs_across_grid_cells():
    pos = sinusoidal_2d(2, 2, 8, torch.device("cpu"), torch.float32)
    assert not torch.allclose(pos[0, 0], pos[0, 1])
    assert not torch.allclose(pos[0, 0], pos[0, 3])
