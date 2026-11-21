import dataclasses

import torch


@dataclasses.dataclass
class NormStats:
    """Per-dimension z-score statistics of the robot state and the actions."""

    state_mean: torch.Tensor
    state_std: torch.Tensor
    action_mean: torch.Tensor
    action_std: torch.Tensor

    @classmethod
    def from_tensors(cls, state: torch.Tensor, action: torch.Tensor) -> "NormStats":
        """Statistics over every dimension but the last; std is floored so constant dimensions stay finite."""
        state_dims = tuple(range(state.ndim - 1))
        action_dims = tuple(range(action.ndim - 1))
        return cls(
            state_mean=state.mean(dim=state_dims),
            state_std=state.std(dim=state_dims, unbiased=False).clamp_min(1e-6),
            action_mean=action.mean(dim=action_dims),
            action_std=action.std(dim=action_dims, unbiased=False).clamp_min(1e-6),
        )

    @classmethod
    def from_dict(cls, data: dict[str, torch.Tensor]) -> "NormStats":
        return cls(**{field.name: data[field.name] for field in dataclasses.fields(cls)})

    def to_dict(self) -> dict[str, torch.Tensor]:
        return {field.name: getattr(self, field.name).cpu() for field in dataclasses.fields(self)}

    def to(self, device: torch.device) -> "NormStats":
        return NormStats(**{field.name: getattr(self, field.name).to(device) for field in dataclasses.fields(self)})

    def normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        return (state - self.state_mean) / self.state_std

    def normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        return (action - self.action_mean) / self.action_std

    def unnormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        return action * self.action_std + self.action_mean
