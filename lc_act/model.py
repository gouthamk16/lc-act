from typing import Protocol

import torch
from torch import nn

from lc_act.posenc import grid_hw, sinusoidal_2d
from lc_act.types import (
    ACTION_DIM,
    CLIP_DIM,
    FFN_DIM,
    HORIZON,
    N_DECODER_LAYERS,
    N_ENCODER_LAYERS,
    N_HEADS,
    RESNET_DIM,
    STATE_DIM,
    TOKEN_DIM,
)


class SpatialVision(Protocol):
    def encode(self, images: torch.Tensor) -> torch.Tensor: ...


class TextEncoder(Protocol):
    def encode(self, texts: list[str]) -> torch.Tensor: ...


def freeze_module(module: nn.Module) -> None:
    module.eval()
    for param in module.parameters():
        param.requires_grad = False


def trainable_state_dict(model: "LcAct") -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
        if not key.startswith("text.")
    }


def load_trainable(model: "LcAct", state_dict: dict[str, torch.Tensor]) -> "LcAct":
    result = model.load_state_dict(state_dict, strict=False)
    missing = [key for key in result.missing_keys if not key.startswith("text.")]
    if missing:
        raise RuntimeError(f"checkpoint missing trainable keys: {missing}")
    return model


def _make_encoder(d_model: int, n_heads: int, ffn_dim: int, n_layers: int) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model=d_model, nhead=n_heads, dim_feedforward=ffn_dim, batch_first=True,
        norm_first=True,
    )
    return nn.TransformerEncoder(layer, num_layers=n_layers)


def _make_decoder(d_model: int, n_heads: int, ffn_dim: int, n_layers: int) -> nn.TransformerDecoder:
    layer = nn.TransformerDecoderLayer(
        d_model=d_model, nhead=n_heads, dim_feedforward=ffn_dim, batch_first=True,
        norm_first=True,
    )
    return nn.TransformerDecoder(layer, num_layers=n_layers)


class LcAct(nn.Module):
    def __init__(
        self,
        vision: SpatialVision,
        text: TextEncoder,
        d_model: int = TOKEN_DIM,
        n_encoder_layers: int = N_ENCODER_LAYERS,
        n_decoder_layers: int = N_DECODER_LAYERS,
        n_heads: int = N_HEADS,
        ffn_dim: int = FFN_DIM,
        horizon: int = HORIZON,
    ) -> None:
        super().__init__()
        self.vision = vision
        self.text = text
        self.horizon = horizon
        self.vision_proj = nn.Linear(RESNET_DIM, d_model)
        self.text_proj = nn.Linear(CLIP_DIM, d_model)
        self.state_proj = nn.Linear(STATE_DIM, d_model)
        self.action_queries = nn.Parameter(torch.randn(horizon, d_model) * 0.02)
        self.camera_embed = nn.Parameter(torch.randn(2, d_model) * 0.02)
        self.type_embed = nn.Parameter(torch.randn(2, d_model) * 0.02)
        self.encoder = _make_encoder(d_model, n_heads, ffn_dim, n_encoder_layers)
        self.decoder = _make_decoder(d_model, n_heads, ffn_dim, n_decoder_layers)
        self.action_head = nn.Linear(d_model, ACTION_DIM)
        self.film = nn.Linear(d_model, 2 * d_model)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(
        self,
        workspace: torch.Tensor,
        wrist: torch.Tensor,
        state: torch.Tensor,
        tasks: list[str],
    ) -> torch.Tensor:
        lang = self.text_proj(self.text.encode(tasks)).unsqueeze(1)
        ws = self._modulate(self._image_tokens(workspace, camera=0), lang)
        wr = self._modulate(self._image_tokens(wrist, camera=1), lang)
        tx = lang + self.type_embed[0]
        st = self.state_proj(state).unsqueeze(1) + self.type_embed[1]
        memory = self.encoder(torch.cat([ws, wr, tx, st], dim=1))
        queries = self.action_queries.unsqueeze(0).expand(state.shape[0], -1, -1)
        decoded = self.decoder(queries, memory)
        return self.action_head(decoded)

    def _modulate(self, tokens: torch.Tensor, lang: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.film(lang).chunk(2, dim=-1)
        return tokens * (1 + gamma) + beta

    def _image_tokens(self, images: torch.Tensor, camera: int) -> torch.Tensor:
        tokens = self.vision_proj(self.vision.encode(images))
        height, width = grid_hw(tokens.shape[1])
        pos = sinusoidal_2d(
            height, width, tokens.shape[-1], tokens.device, tokens.dtype,
        )
        return tokens + pos + self.camera_embed[camera]


def _as_nchw_float(images: torch.Tensor) -> torch.Tensor:
    if images.ndim != 4:
        raise ValueError(f"expected 4D image batch, got {tuple(images.shape)}")
    if images.shape[-1] == 3:
        images = images.permute(0, 3, 1, 2)
    x = images.float()
    if x.max() > 1.5:
        x = x / 255.0
    return x


class ResNetSpatial(nn.Module):
    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18

        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = resnet18(weights=weights)
        self.stem = nn.Sequential(*list(net.children())[:-2])
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        x = _as_nchw_float(images).to(device=self.mean.device, dtype=self.mean.dtype)
        x = (x - self.mean) / self.std
        feat = self.stem(x)
        return feat.flatten(2).transpose(1, 2)


class ClipTextEncoder(nn.Module):
    def __init__(self, model_id: str = "openai/clip-vit-base-patch32") -> None:
        super().__init__()
        from transformers import AutoTokenizer, CLIPTextModelWithProjection

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = CLIPTextModelWithProjection.from_pretrained(model_id)
        freeze_module(self.model)

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @torch.no_grad()
    def encode(self, texts: list[str]) -> torch.Tensor:
        tokens = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True,
        )
        tokens = {k: v.to(self.device) for k, v in tokens.items()}
        return self.model(**tokens).text_embeds
