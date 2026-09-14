from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch


def flip_hw(image: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    if isinstance(image, torch.Tensor):
        return torch.flip(image, dims=(0, 1))
    return image[::-1, ::-1].copy()


def quat_to_axis_angle(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32)
    xyz = quat[:3]
    w = float(quat[3])
    norm = np.linalg.norm(xyz)
    if norm < 1e-8:
        return np.zeros(3, dtype=np.float32)
    axis = xyz / norm
    angle = 2.0 * np.arctan2(norm, w)
    return (axis * angle).astype(np.float32)


def _robot_state(obs: Mapping[str, object]) -> Mapping[str, object]:
    if "robot_state" in obs:
        return obs["robot_state"]  # type: ignore[return-value]
    state = torch.as_tensor(obs["observation.state"]).float()
    if state.shape != (8,):
        raise ValueError(f"expected observation.state shape (8,), got {tuple(state.shape)}")
    return {"packed": state}


def pack_state(obs: dict) -> torch.Tensor:
    robot_state = _robot_state(obs)
    if "packed" in robot_state:
        return robot_state["packed"]  # type: ignore[return-value]
    eef = robot_state["eef"]  # type: ignore[index]
    gripper = robot_state["gripper"]  # type: ignore[index]
    position = torch.as_tensor(eef["pos"]).float()  # type: ignore[index]
    quat = np.asarray(eef["quat"], dtype=np.float32)  # type: ignore[index]
    axis_angle = torch.from_numpy(quat_to_axis_angle(quat))
    qpos = torch.as_tensor(gripper["qpos"]).float()  # type: ignore[index]
    packed = torch.cat([position, axis_angle, qpos])
    if packed.shape != (8,):
        raise ValueError(f"expected packed state shape (8,), got {tuple(packed.shape)}")
    return packed
