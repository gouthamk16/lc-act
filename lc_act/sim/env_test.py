import numpy as np
import torch

from lc_act.sim import env as _env


def test_flip_hw_rotates_180():
    image = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    out = _env.flip_hw(image)
    assert out[0, 0, 0] == image[-1, -1, 0]
    assert out[-1, -1, 0] == image[0, 0, 0]


def test_identity_quat_axis_angle_is_zero():
    assert np.allclose(_env.quat_to_axis_angle(np.array([0, 0, 0, 1], dtype=np.float32)), 0)


def test_observe_packs_8d_state_and_flips_both_cameras():
    image = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)
    obs = {
        "pixels": {"image": image, "image2": image},
        "robot_state": {
            "eef": {"pos": np.array([0.1, 0.2, 0.3]), "quat": np.array([0, 0, 0, 1.0])},
            "gripper": {"qpos": np.array([0.04, -0.04])},
        },
    }
    workspace, wrist, state = _env.observe(obs)
    assert state.shape == (8,)
    assert torch.allclose(torch.from_numpy(state[:3]), torch.tensor([0.1, 0.2, 0.3]))
    assert workspace[0, 0, 0] == image[-1, -1, 0]
    assert np.array_equal(workspace, wrist)
