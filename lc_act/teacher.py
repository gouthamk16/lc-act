"""Frozen DINOv2 features for every cached frame, computed once; training aligns to them, inference never loads them."""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from lc_act.cache import CAMERAS, DEFAULT_ROOT
from lc_act.types import VISION_SIZE

TEACHER_ID = "facebook/dinov2-small"
TEACHER_DIM = 384


@torch.no_grad()
def build(root: Path, batch: int) -> None:
    from transformers import AutoModel

    device = torch.device("cuda")
    model = AutoModel.from_pretrained(TEACHER_ID, torch_dtype=torch.float16).to(device).eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    rows = int(np.load(root / "task_index.npy").shape[0])
    out = np.lib.format.open_memmap(root / "teacher.npy", "w+", np.float16, (rows, len(CAMERAS), TEACHER_DIM))
    for cam, name in enumerate(CAMERAS):
        images = np.memmap(root / f"{name}.u8", np.uint8, "r", shape=(rows, VISION_SIZE, VISION_SIZE, 3))
        for start in range(0, rows, batch):
            x = torch.from_numpy(np.ascontiguousarray(images[start:start + batch])).to(device)
            x = x.permute(0, 3, 1, 2).half() / 255.0
            # 224 px is DINOv2's native 16x16 patch grid; upsampling the 128 px cache keeps teacher and student on the same pixels.
            x = F.interpolate(x, size=224, mode="bilinear", align_corners=False)
            tokens = model(pixel_values=(x - mean) / std).last_hidden_state[:, 1:]
            out[start:start + batch, cam] = tokens.mean(1).cpu().numpy()
            if start % (batch * 100) == 0:
                print(f"{name} {start}/{rows}", flush=True)
    out.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache frozen DINOv2 features per frame")
    parser.add_argument("--cache", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--batch", type=int, default=256)
    args = parser.parse_args()
    build(args.cache, args.batch)


if __name__ == "__main__":
    main()
