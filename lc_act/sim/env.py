"""LIBERO environments through LeRobot's wrapper, and the observation layout the policy was trained on."""

import importlib.util
import os
import pathlib
from typing import Any

import numpy as np

ENV_FPS = 20  # LIBERO control rate
ENV_RESOLUTION = 256  # rendered camera size, as in the training data


def configure_gl() -> None:
    """Pick the MuJoCo GL backend; call before importing MuJoCo.

    Under WSL2, EGL falls back to llvmpipe (~42 ms per camera); GLFW through Mesa's D3D12 driver renders on the GPU,
    and its frames match llvmpipe's to 0.5/255 on average.
    """
    if "WSL_DISTRO_NAME" in os.environ and "MUJOCO_GL" not in os.environ:
        os.environ.update(MUJOCO_GL="glfw", GALLIUM_DRIVER="d3d12", MESA_D3D12_DEFAULT_ADAPTER_NAME="NVIDIA")
    os.environ.setdefault("MUJOCO_GL", "egl")


def ensure_libero_config() -> None:
    """Write ~/.libero/config.yaml if missing, so importing LIBERO does not stop to prompt for paths."""
    config_dir = pathlib.Path(os.environ.get("LIBERO_CONFIG_PATH", pathlib.Path.home() / ".libero"))
    config_file = config_dir / "config.yaml"
    if config_file.exists():
        return
    spec = importlib.util.find_spec("libero.libero")
    if spec is None or spec.origin is None:
        raise RuntimeError("LIBERO package is not installed")

    root = pathlib.Path(spec.origin).parent
    config_dir.mkdir(parents=True, exist_ok=True)
    config_file.write_text(
        f"benchmark_root: {root}\n"
        f"bddl_files: {root / 'bddl_files'}\n"
        f"init_states: {root / 'init_files'}\n"
        f"datasets: {root / 'datasets'}\n"
        f"assets: {root / 'assets'}\n"
    )


def benchmark_dict() -> dict[str, Any]:
    """LIBERO's suite constructors by name, imported only after the GL backend and config are in place."""
    configure_gl()
    ensure_libero_config()
    from libero.libero import benchmark  # noqa: PLC0415

    return benchmark.get_benchmark_dict()


def make_env(task_id: int = 0, suite_name: str = "libero_object") -> Any:
    """One LIBERO task with LeRobot's step cap for its suite (Spatial/Object 280, Goal 300, Long 520)."""
    try:
        suite = benchmark_dict()[suite_name]()
        from lerobot.envs.libero import LiberoEnv  # noqa: PLC0415

        return LiberoEnv(
            task_suite=suite,
            task_id=task_id,
            task_suite_name=suite_name,
            obs_type="pixels_agent_pos",
            control_mode="relative",
            observation_width=ENV_RESOLUTION,
            observation_height=ENV_RESOLUTION,
            init_states=True,
        )
    except Exception as error:
        message = f"{type(error).__name__}: {error}".lower()
        if "egl" in message or "opengl" in message:
            print("MUJOCO_GL=egl failed; try export MUJOCO_GL=osmesa")
        raise


def max_steps(env: Any) -> int:
    # LeRobot's LiberoEnv keeps the suite's step cap only in this private attribute.
    return env._max_episode_steps  # noqa: SLF001


def observe(obs: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Workspace and wrist frames (256, 256, 3) uint8 and the raw (8,) state, in the training data's layout."""
    # IMPORTANT: rotate 180 degrees; LIBERO renders upside down relative to the dataset.
    workspace = flip_hw(np.asarray(obs["pixels"]["image"]))
    wrist = flip_hw(np.asarray(obs["pixels"]["image2"]))
    return workspace, wrist, pack_state(obs)


def flip_hw(image: np.ndarray) -> np.ndarray:
    """Rotate an (h, w, c) image 180 degrees, as a contiguous copy."""
    return image[::-1, ::-1].copy()


def pack_state(obs: dict[str, Any]) -> np.ndarray:
    """eef position (3), eef axis-angle (3) and gripper qpos (2) as float32, the dataset's `observation.state`."""
    eef, gripper = obs["robot_state"]["eef"], obs["robot_state"]["gripper"]
    position = np.asarray(eef["pos"], dtype=np.float32)
    qpos = np.asarray(gripper["qpos"], dtype=np.float32)

    packed = np.concatenate([position, quat_to_axis_angle(eef["quat"]), qpos])
    if packed.shape != (8,):
        raise ValueError(f"expected packed state shape (8,), got {tuple(packed.shape)}")
    return packed


def quat_to_axis_angle(quat: np.ndarray) -> np.ndarray:
    """(x, y, z, w) quaternion -> (3,) axis-angle."""
    quat = np.asarray(quat, dtype=np.float32)
    xyz, w = quat[:3], float(quat[3])
    norm = np.linalg.norm(xyz)
    if norm < 1e-8:
        return np.zeros(3, dtype=np.float32)
    axis = xyz / norm
    angle = 2.0 * np.arctan2(norm, w)
    return (axis * angle).astype(np.float32)
