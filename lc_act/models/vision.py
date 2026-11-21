import torch
from torch import nn
import torch.nn.functional as F  # noqa: N812
from torchvision.models import ResNet18_Weights
from torchvision.models import resnet18

VISION_SIZE = 128  # input resolution of the ResNet, and of the training frame cache
STAGE_CHANNELS = (64, 128, 256)  # ResNet-18 layer1..layer3, each FiLM'd by language
FEATURE_DIM = STAGE_CHANNELS[-1]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ResNetSpatial(nn.Module):
    """ResNet-18 through layer3 with per-stage FiLM; returns an 8x8 grid of feature tokens at 128 px."""

    def __init__(self, *, pretrained: bool = True) -> None:
        super().__init__()
        net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        # Through layer3: at 128 px this is an 8x8 grid without layer4's 8.4M params.
        # channels_last: 2.2x faster fp16 convs on Ada (157 -> 70 ms per batch-64 step).
        self.stem = nn.Sequential(*list(net.children())[:-3]).to(memory_format=torch.channels_last)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def encode(self, images: torch.Tensor, film: list[torch.Tensor], group: int = 1) -> torch.Tensor:
        """Images (n, h, w, 3) uint8 or float -> tokens (n, grid, FEATURE_DIM).

        Args:
          images: frames of any size; they are resized to VISION_SIZE.
          film: per-stage (n, 2 * channels) FiLM parameters, gamma then beta.
          group: consecutive frames that are one sample's history share a random-shift offset in training.
        """
        x = _as_nchw_float(images).to(device=self.mean.device, dtype=self.mean.dtype)
        if x.shape[-1] != VISION_SIZE:
            x = F.interpolate(x, size=VISION_SIZE, mode="bilinear", antialias=True)
        if self.training:
            x = _random_shift(x, pad=8, group=group)
        x = ((x - self.mean) / self.std).contiguous(memory_format=torch.channels_last)

        # conv1, bn1, relu, maxpool, then the three FiLM'd stages
        feat = self.stem[:4](x)
        for stage, stage_film in zip(self.stem[4:], film, strict=True):
            feat = stage(feat)
            gamma, beta = stage_film[:, :, None, None].to(feat.dtype).chunk(2, dim=1)
            feat = feat * (1 + gamma) + beta

        return feat.flatten(2).transpose(1, 2)


def _as_nchw_float(images: torch.Tensor) -> torch.Tensor:
    """(n, h, w, 3) or (n, 3, h, w), uint8 or [0, 1] float -> (n, 3, h, w) float in [0, 1]."""
    if images.ndim != 4:
        raise ValueError(f"expected 4D image batch, got {tuple(images.shape)}")
    if images.shape[-1] == 3:
        images = images.permute(0, 3, 1, 2)
    x = images.float()
    if x.max() > 1.5:
        x = x / 255.0
    return x


def _random_shift(x: torch.Tensor, pad: int, group: int = 1) -> torch.Tensor:
    """Shift each image by up to `pad` px; each run of `group` history frames shares one offset."""
    n, _, h, w = x.shape
    padded = F.pad(x, (pad, pad, pad, pad), mode="replicate")
    offsets = torch.randint(0, 2 * pad + 1, (n // group, 2)).repeat_interleave(group, 0).tolist()
    return torch.stack([padded[i, :, dy : dy + h, dx : dx + w] for i, (dy, dx) in enumerate(offsets)])
