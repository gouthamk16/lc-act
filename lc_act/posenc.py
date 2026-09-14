import math

import torch


def grid_hw(n_tokens: int) -> tuple[int, int]:
    side = int(math.isqrt(n_tokens))
    if side * side != n_tokens:
        raise ValueError(f"spatial tokens must form a square grid, got {n_tokens}")
    return side, side


def sinusoidal_2d(
    height: int,
    width: int,
    dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if dim % 2 != 0:
        raise ValueError(f"pos-embed dim must be even, got {dim}")
    half = dim // 2
    rows = torch.arange(1, height + 1, device=device, dtype=torch.float32)
    cols = torch.arange(1, width + 1, device=device, dtype=torch.float32)
    rows = rows / rows[-1] * (2 * math.pi)
    cols = cols / cols[-1] * (2 * math.pi)
    grid_y, grid_x = torch.meshgrid(rows, cols, indexing="ij")
    freq = 10000 ** (
        2 * (torch.arange(half, device=device, dtype=torch.float32) // 2) / half
    )
    y = grid_y.unsqueeze(-1) / freq
    x = grid_x.unsqueeze(-1) / freq
    pos_y = torch.stack((y[..., 0::2].sin(), y[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_x = torch.stack((x[..., 0::2].sin(), x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos = torch.cat((pos_y, pos_x), dim=-1)
    return pos.reshape(1, height * width, dim).to(dtype=dtype)
