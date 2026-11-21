"""LC-ACT: a language-conditioned action-chunking transformer for LIBERO."""

import dataclasses
from typing import Protocol

import torch
from torch import nn

from lc_act.models import posenc as _posenc
from lc_act.models import text as _text
from lc_act.models import vision as _vision

STATE_DIM = 8  # eef position (3), eef axis-angle (3), gripper qpos (2)
ACTION_DIM = 7  # relative eef pose (6), gripper (1)
HORIZON = 16  # actions per chunk
TEACHER_DIM = 384  # DINOv2-small pooled patch feature, the alignment target


class Vision(Protocol):
    def encode(self, images: torch.Tensor, film: list[torch.Tensor], group: int = 1) -> torch.Tensor: ...


class TextEncoder(Protocol):
    def encode(self, texts: list[str]) -> torch.Tensor: ...


@dataclasses.dataclass(frozen=True)
class LcActConfig:
    width: int = 256
    num_heads: int = 8
    num_encoder_layers: int = 3
    num_decoder_layers: int = 2
    ffn_dim: int = 1024
    horizon: int = HORIZON
    # History frames per sample (oldest first) and the gap between them in env steps.
    n_obs: int = 1
    obs_stride: int = 1

    @property
    def history_size(self) -> int:
        """Past steps to keep so the oldest of n_obs frames, obs_stride apart, is in reach."""
        return self.obs_stride * (self.n_obs - 1) + 1

    def create(self, *, pretrained: bool = True) -> "LcAct":
        # Built in this order (vision, text, policy) so seeded runs draw the same initial weights.
        vision = _vision.ResNetSpatial(pretrained=pretrained)
        text = _text.ClipTextEncoder()
        return LcAct(self, vision, text)


class LcAct(nn.Module):
    """Encoder over camera, language and state tokens; a decoder turns `horizon` learned queries into actions."""

    def __init__(self, config: LcActConfig, vision: Vision, text: TextEncoder) -> None:
        super().__init__()
        self.config = config
        self.vision = vision
        self.text = text
        width = config.width

        # History frames are fused per token by channel concat, so the token count stays fixed.
        self.vision_proj = nn.Linear(config.n_obs * _vision.FEATURE_DIM, width)
        self.text_proj = nn.Linear(_text.TEXT_DIM, width)
        self.state_proj = nn.Linear(config.n_obs * STATE_DIM, width)
        self.action_queries = nn.Parameter(torch.randn(config.horizon, width) * 0.02)
        self.camera_embed = nn.Parameter(torch.randn(2, width) * 0.02)
        self.type_embed = nn.Parameter(torch.randn(2, width) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            width, config.num_heads, config.ffn_dim, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, config.num_encoder_layers, enable_nested_tensor=False)
        decoder_layer = nn.TransformerDecoderLayer(
            width, config.num_heads, config.ffn_dim, batch_first=True, norm_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, config.num_decoder_layers)
        self.action_head = nn.Linear(width, ACTION_DIM)
        # Laplace log-scale per action dim, fit on detached features so it never changes the actions.
        self.scale_head = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, ACTION_DIM))

        # FiLM layers start at zero, so the pretrained ResNet starts unmodulated.
        self.film = nn.Linear(width, 2 * width)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        # Language steers each ResNet stage, so Goal-suite instructions (same scene, different target) change
        # what the features pick out.
        self.stage_film = nn.Linear(width, 2 * sum(_vision.STAGE_CHANNELS))
        nn.init.zeros_(self.stage_film.weight)
        nn.init.zeros_(self.stage_film.bias)

        # Training-only: predict frozen DINOv2 features of this frame and of the frame one chunk ahead.
        self.align = nn.ModuleList([_teacher_head(width) for _ in range(2)])

    def forward(
        self,
        workspace: torch.Tensor,
        wrist: torch.Tensor,
        state: torch.Tensor,
        tasks: list[str],
        *,
        with_scale: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Images (b, t, h, w, 3) and state (b, t, 8), oldest frame first, t = n_obs -> actions (b, horizon, 7).

        With `with_scale`, also the Laplace log-scale of every action, same shape.
        """
        return self.decode(self.encode(workspace, wrist, state, tasks), with_scale=with_scale)

    def encode(
        self, workspace: torch.Tensor, wrist: torch.Tensor, state: torch.Tensor, tasks: list[str]
    ) -> torch.Tensor:
        """Encoder memory (b, 2 * grid + 2, width): workspace tokens, wrist tokens, a text token, a state token."""
        # one language token, which also FiLMs every image token and ResNet stage
        lang = self.text_proj(self.text.encode(tasks)).unsqueeze(1)

        workspace_tokens = self._modulate(self._image_tokens(workspace, 0, lang), lang)
        wrist_tokens = self._modulate(self._image_tokens(wrist, 1, lang), lang)
        text_token = lang + self.type_embed[0]
        state_token = self.state_proj(state.flatten(1)).unsqueeze(1) + self.type_embed[1]

        return self.encoder(torch.cat([workspace_tokens, wrist_tokens, text_token, state_token], dim=1))

    def decode(
        self, memory: torch.Tensor, *, with_scale: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        queries = self.action_queries.unsqueeze(0).expand(memory.shape[0], -1, -1)
        hidden = self.decoder(queries, memory)
        actions = self.action_head(hidden)
        if not with_scale:
            return actions
        return actions, self.scale_head(hidden.detach())

    def teacher_prediction(self, memory: torch.Tensor) -> torch.Tensor:
        """(b, 2 times, 2 cameras, TEACHER_DIM) from each camera's mean-pooled encoder tokens."""
        # drop the text and state tokens; `per_camera` is (b, 2, width)
        per_camera = memory[:, : memory.shape[1] - 2].unflatten(1, (2, -1)).mean(2)
        return torch.stack([head(per_camera) for head in self.align], dim=1)

    def _modulate(self, tokens: torch.Tensor, lang: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.film(lang).chunk(2, dim=-1)
        return tokens * (1 + gamma) + beta

    def _image_tokens(self, images: torch.Tensor, camera: int, lang: torch.Tensor) -> torch.Tensor:
        """One camera's frames (b, t, h, w, 3) -> tokens (b, grid, width)."""
        batch, frames = images.shape[:2]

        # per-stage FiLM from language, repeated for every history frame
        film = self.stage_film(lang[:, 0]).repeat_interleave(frames, 0)
        film = list(film.split([2 * c for c in _vision.STAGE_CHANNELS], dim=-1))

        # `feats` is (b * t, grid, c); history frames are concatenated per grid cell -> (b, grid, t * c)
        feats = self.vision.encode(images.flatten(0, 1), film, group=frames)
        feats = feats.reshape(batch, frames, *feats.shape[1:]).transpose(1, 2).flatten(2)

        tokens = self.vision_proj(feats)
        height, width = _posenc.grid_hw(tokens.shape[1])
        pos = _posenc.sinusoidal_2d(height, width, tokens.shape[-1], tokens.device, tokens.dtype)
        return tokens + pos + self.camera_embed[camera]


def _teacher_head(width: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(width, 2 * width), nn.GELU(), nn.Linear(2 * width, TEACHER_DIM))
