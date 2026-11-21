import dataclasses
import pathlib

from lc_act.models import lc_act as _lc_act
from lc_act.training import data_loader as _data


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    #################################################################################################################
    # Run
    #################################################################################################################
    out: pathlib.Path = pathlib.Path("outputs/lc_act")  # last.pt is written here
    resume: pathlib.Path | None = None  # checkpoint to continue, with its optimizer state
    cache: pathlib.Path = _data.DEFAULT_ROOT  # frame cache dir (python -m lc_act.training.cache)
    seed: int = 0
    workers: int = 6  # JPEG decode workers

    #################################################################################################################
    # Length. A seconds or steps budget makes the run fixed: holdout val, no deadline, one save at the end.
    #################################################################################################################
    epochs: int = 20
    max_hours: float = 2  # wall-clock deadline; 0 disables
    save_every: int = 100  # steps between checkpoints; 0 saves at epoch ends only
    budget_seconds: float | None = None  # armed after the first batch, so startup is not counted
    budget_steps: int | None = None  # like --budget-seconds, but machine-independent
    val_batches: int = 1000

    #################################################################################################################
    # Data
    #################################################################################################################
    batch_size: int = 64
    object_only: bool = False  # train on LIBERO-Object only
    n_obs: int = 2  # observation frames per sample
    obs_stride: int = 1  # gap in env steps between history frames
    long_weight: float = 1.0  # sampling weight of LIBERO-Long rows

    #################################################################################################################
    # Optimization
    #################################################################################################################
    lr: float = 1.2e-3  # AdamW peak
    warmup_frac: float = 0.02  # linear warmup, then cosine decay to zero
    align_weight: float = 0.5  # DINOv2 alignment loss; 0 skips loading the teacher features

    @property
    def fixed_budget(self) -> bool:
        return self.budget_seconds is not None or self.budget_steps is not None

    def model_config(self) -> _lc_act.LcActConfig:
        return _lc_act.LcActConfig(n_obs=self.n_obs, obs_stride=self.obs_stride)
