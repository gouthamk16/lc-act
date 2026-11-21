"""Frozen DINOv2 features for every cached frame, computed once; training aligns to them, inference never loads them.

python -m lc_act.training.teacher
"""

import dataclasses
import pathlib

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import tyro

from lc_act.models import lc_act as _lc_act
from lc_act.models import vision as _vision
from lc_act.training import data_loader as _data

TEACHER_ID = "facebook/dinov2-small"
# 224 px is DINOv2's native 16x16 patch grid; upsampling the 128 px cache keeps teacher and student on the same pixels.
TEACHER_SIZE = 224


@dataclasses.dataclass(frozen=True)
class Args:
    cache: pathlib.Path = _data.DEFAULT_ROOT
    batch: int = 256


@torch.no_grad()
def main(args: Args) -> None:
    # Deferred: transformers takes seconds to import, and --help should not pay for it.
    from transformers import AutoModel  # noqa: PLC0415

    device = torch.device("cuda")
    model = AutoModel.from_pretrained(TEACHER_ID, torch_dtype=torch.float16).to(device).eval()
    mean = torch.tensor(_vision.IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(_vision.IMAGENET_STD, device=device).view(1, 3, 1, 1)

    size = _vision.VISION_SIZE
    rows = int(np.load(args.cache / "task_index.npy").shape[0])
    shape = (rows, len(_data.CAMERAS), _lc_act.TEACHER_DIM)
    out = np.lib.format.open_memmap(args.cache / "teacher.npy", "w+", np.float16, shape)
    for cam, name in enumerate(_data.CAMERAS):
        images = np.memmap(args.cache / f"{name}.u8", np.uint8, "r", shape=(rows, size, size, 3))
        for start in range(0, rows, args.batch):
            x = torch.from_numpy(np.ascontiguousarray(images[start : start + args.batch])).to(device)
            x = x.permute(0, 3, 1, 2).half() / 255.0
            x = F.interpolate(x, size=TEACHER_SIZE, mode="bilinear", align_corners=False)

            # mean-pool the patch tokens, dropping CLS
            tokens = model(pixel_values=(x - mean) / std).last_hidden_state[:, 1:]
            out[start : start + args.batch, cam] = tokens.mean(1).cpu().numpy()
            if start % (args.batch * 100) == 0:
                print(f"{name} {start}/{rows}", flush=True)
    out.flush()


if __name__ == "__main__":
    main(tyro.cli(Args))
