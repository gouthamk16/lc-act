import torch
from torch import nn

CLIP_ID = "openai/clip-vit-base-patch32"
TEXT_DIM = 512  # CLIP ViT-B/32 projected text embedding


class ClipTextEncoder(nn.Module):
    """Frozen CLIP text tower; each distinct instruction is embedded once and cached."""

    def __init__(self, model_id: str = CLIP_ID) -> None:
        super().__init__()
        # Deferred: transformers takes seconds to import, and tests use a fake encoder.
        from transformers import AutoTokenizer  # noqa: PLC0415
        from transformers import CLIPTextModelWithProjection  # noqa: PLC0415

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = CLIPTextModelWithProjection.from_pretrained(model_id)
        self.model.eval()
        self.model.requires_grad_(requires_grad=False)
        self._cache: dict[str, torch.Tensor] = {}

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @torch.no_grad()
    def encode(self, texts: list[str]) -> torch.Tensor:
        """(len(texts), TEXT_DIM) embeddings."""
        missing = sorted(set(texts) - self._cache.keys())
        if missing:
            tokens = self.tokenizer(missing, return_tensors="pt", padding=True, truncation=True)
            tokens = {key: value.to(self.device) for key, value in tokens.items()}
            for text, embed in zip(missing, self.model(**tokens).text_embeds, strict=True):
                self._cache[text] = embed
        return torch.stack([self._cache[text] for text in texts]).to(self.device)
