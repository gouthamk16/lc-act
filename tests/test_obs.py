import numpy as np
import torch

from lc_act.obs import flip_hw, pack_state, quat_to_axis_angle


def test_flip_hw_rotates_180():
    image = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    out = flip_hw(image)
    assert out[0, 0, 0] == image[-1, -1, 0]
    assert out[-1, -1, 0] == image[0, 0, 0]


def test_identity_quat_axis_angle_is_zero():
    assert np.allclose(
        quat_to_axis_angle(np.array([0, 0, 0, 1], dtype=np.float32)),
        0,
    )


def test_pack_state_is_8d():
    obs = {
        "robot_state": {
            "eef": {
                "pos": np.array([0.1, 0.2, 0.3]),
                "quat": np.array([0, 0, 0, 1.0]),
            },
            "gripper": {"qpos": np.array([0.04, -0.04])},
        }
    }
    packed = pack_state(obs)
    assert packed.shape == (8,)
    assert torch.allclose(packed[:3], torch.tensor([0.1, 0.2, 0.3]))
