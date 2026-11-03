# LC-ACT

LC-ACT is a tabletop Franka Panda arm policy in MuJoCo LIBERO. This repository
is a **VLA research lab** for future [Starscream](https://github.com/gouthamk16/starscream) policies: same observe → chunk
→ act slot as the drone's `HoverPolicy`, different body. It is not the x500
and not the [Starscream](https://github.com/gouthamk16/starscream) flight stack.

GitHub: [gouthamk16/lc-act](https://github.com/gouthamk16/lc-act). In
[Starscream](https://github.com/gouthamk16/starscream) this repo is the `vla/` git submodule.

https://github.com/user-attachments/assets/625e6b4f-1673-4be4-ab51-8bfe6b167ec2

*The all-40-task policy driven from the web console (`python -m lc_act.live`):
type or pick an instruction, and the matching LIBERO scene runs closed loop.*

![Panda arm picking an object and placing it in the basket, frames left to right](assets/image.png)

*Closed-loop LIBERO-Object rollout: reach, grasp, carry, and release into the
basket (frames left to right).*

## The observe-to-act loop

The model has the same three boxes as any small vision-language-action (VLA)
policy:

1. **Vision**: workspace and wrist images are resized to 128×128 and encoded by
   a shared, trainable ResNet-18 cut after layer3 into an 8×8 grid of tokens
   per camera. Each token gets an ACT-style 2-D sinusoidal position embedding,
   a learned camera id (workspace vs wrist), and FiLM modulation from the
   instruction.
2. **Language**: frozen CLIP text turns the instruction into one language
   token (plus a learned type tag). Embeddings are cached per instruction
   string.
3. **Action**: a small transformer (width 256, 3 encoder and 2 decoder layers)
   combines those tokens with the 8-D robot state (plus a type tag) and
   predicts a 16-step chunk of relative 7-D pose/gripper actions.

The closed loop is **observe → predict a 16-step chunk → execute one action →
observe again**. The policy replans every env step and executes the ACT
temporal ensemble of every chunk that covers that step. One dataset row is one
env control step; the dataset's 10 fps is a metadata label only.

Training uses all 40 tasks of the four standard LIBERO suites (Spatial,
Object, Goal, Long) and gives the model the previous frame and state as well
(`--n-obs 2`). During training only, the encoder also learns to predict frozen
DINOv2 features of the current frame and of the frame one chunk ahead; those
heads are dropped at inference. Evaluation (`lc_act.bench`) runs every task of
all four suites, the protocol MINERVA and π₀.₅ report.

The model is 8.0M parameters at inference.

More about the model architecture in [MODEL.md](MODEL.md).

## Install

CUDA wheels first, then the rest. From **this repository root** (standalone
clone of `lc-act`, or `cd vla` inside [Starscream](https://github.com/gouthamk16/starscream)):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install -e .
```

`av` in `requirements.txt` is the PyAV video backend used by LeRobot.

[Starscream](https://github.com/gouthamk16/starscream) checkout:

```bash
git clone --recurse-submodules https://github.com/gouthamk16/starscream.git
cd starscream/vla
```

If you already cloned [Starscream](https://github.com/gouthamk16/starscream) without submodules:

```bash
git submodule update --init vla
```

## Run

Build the frame cache once (~25 min: decodes the AV1 videos to 128 px, then
packs them as q95 JPEG, ~30 GB of scratch on disk and 3.6 GB kept), then the
frozen DINOv2 targets (~15 min, 420 MB):

```bash
python -m lc_act.cache
python -m lc_act.teacher
```

Long train. The cosine learning rate schedule spans the `--max-hours`
deadline, or `--epochs` when there is none:

```bash
python -u -m lc_act.train --out outputs/lc_act_all40 --max-hours 4 --epochs 999 --save-every 2000
```

`--object-only` restricts training to the ten Object tasks.

`--budget-steps N` fixes the run length in optimizer steps and trains on the
90% autoresearch split; 24,656 steps is the 40-minute laptop budget of the
best configuration.

Resume after a stop or crash (pass the same `--n-obs` the checkpoint was
trained with):

```bash
python -m lc_act.train --resume outputs/lc_act/last.pt --out outputs/lc_act --epochs 20 --max-hours 0
```

`--max-hours 2` is still the default if you omit the flag. `--max-hours 0`
means no deadline. Training reads the frame cache with 6 loader workers at
batch 64 (~830 samples/s, 1.7 GB VRAM). If RAM runs out, lower `--workers`.

Benchmark every task of the four suites (init states 0..N-1 per task, as
MINERVA reports with N = 50). Each LIBERO env holds ~2.9 GB of RAM, so keep
`--envs 2` on a 10 GB WSL box and do not run it next to training:

```bash
python -m lc_act.bench --ckpt outputs/lc_act_all40/last.pt --episodes 20 --out outputs/lc_act_all40/bench.json
```

Several `--ckpt` paths average the members' action chunks (they must share
normalization, i.e. the same training split). `--first-init K` starts at init
state K, so a 50/task bench can be split across machines. `--ensemble
uncertainty` weights each chunk's vote by its predicted Laplace scale instead
of age alone.

The older single-task eval runs ten fixed-seed soup episodes and saves episode zero:

```bash
python -m lc_act.eval \
  --ckpt outputs/lc_act/last.pt \
  --episodes 10 \
  --seed 0 \
  --video outputs/lc_act/soup_ep0.mp4
```

The original done bar was at least 2/10 successful soup episodes plus a
playable `outputs/lc_act/soup_ep0.mp4`; the current weights clear it.

Watch the trained policy in a browser. This loads `outputs/lc_act_all40/last.pt`
and streams the simulator cameras (rendered on the GPU under WSL2). The page
lists all 40 tasks by suite with each one's benchmark success rate. A typed
sentence is passed to the policy as written; the scene loads from the trained
task that shares the most words with it. Stop the benchmark first: each LIBERO
env holds ~2.9 GB of RAM.

```bash
python -m lc_act.live
```

Then open http://127.0.0.1:8765.

## Mapping to [Starscream](https://github.com/gouthamk16/starscream)

The tabletop mapping is the same policy slot used later by the drone:

```text
LcAct → pose deltas → Panda
HoverPolicy → TrajectorySetpoint → PX4
```

The body and simulator differ; this README does not add a drone or PX4
integration.

## Results

All numbers come from `lc_act.bench`: every task of the four suites, LeRobot
step caps, replan every step with temporal ensembling. "20/task" is init
states 0–19 (800 rollouts); "50/task" is init states 0–49 (2,000 rollouts), the
protocol MINERVA reports.

| Suite | Best single model, 20/task | 4-model ensemble, 20/task | **4-model ensemble, 50/task** | MINERVA-9.66M, 50/task | π₀.₅ via LeRobot |
| --- | --- | --- | --- | --- | --- |
| Spatial | 98.5% | 98.0% | **97.6%** | 98.4% | 97.0% |
| Object | 100.0% | 99.5% | **99.6%** | 99.0% | 99.0% |
| Goal | 99.0% | 100.0% | **98.8%** | 98.4% | 98.0% |
| Long | 91.0% | 95.0% | **92.6%** | 94.0% | 96.0% |
| **Average** | 97.12% | 98.12% | **97.15%** | 97.45% | 97.5% |
| Params at inference | 8.0M | 31.8M | 31.8M | 9.66M | ~3B |

**Best result: 97.15% at MINERVA's protocol, 0.30 points behind it.** The
ensemble averages the action chunks of four models of the same configuration
(language FiLM in every ResNet stage, two adjacent frames, DINOv2 alignment,
40 minutes on the laptop or the same 24,656 optimizer steps on Kaggle, on the
90% autoresearch split), trained with different seeds. It beats MINERVA on
Object and Goal and loses on Spatial and Long, at 3.3× the parameters. The four checkpoints are not in the repository:

```bash
python -m lc_act.bench --episodes 50 --ckpt \
  outputs/lc_act_film40/last.pt outputs/ar_ckpt/p2s1_seed1/last.pt \
  outputs/ar_ckpt/p2s1_seed2/last.pt outputs/kaggle/r2/base_s24k/last.pt
```

The best single model (`outputs/kaggle/r2/base_s24k/last.pt`, seed 1, trained
on a Kaggle T4) scored 97.12% on 20/task; it has not been run on 50/task.
Four benchmarked single-model runs of the same configuration scored 94.25,
95.00, 95.75 and 97.12% (mean 95.5), so seed noise is about ±1.5 points,
larger than any single change tried since. The ensemble's 98.12% on 20/task
fell to 97.15% on 50/task (init states 20–49 alone: 96.7%), so a claim against
MINERVA needs all 50.

The gap is two Long tasks that every model fails the same way: mug on plate
plus pudding (`libero_10/6`, 74% over 50) and both moka pots on the stove
(`libero_10/8`, 86%). On the moka pots, the arm parks the open gripper at grasp
height over the second pot and sends a near-constant tiny action; that task
has the fewest demos (29). Temporal-ensemble decay does not change it
(0.01 ≈ 0.1 ≈ 0.3).

What did not beat the baseline (20/task, Kaggle T4, seed 1 unless noted;
baseline draws 94.25–97.12%):

| Change | Average |
| --- | --- |
| Half the steps (12,328) | 96.12% |
| 1.5× steps (36,984) | 94.25% |
| Peak LR 4e-4, 24,656 / 49,312 steps | 96.00% / 94.88% |
| MINERVA-like LR 3e-4 / 1e-4, 2.5× steps (61,640) | 94.00% / 93.75% |
| Training-only progress head (MINERVA) | 95.38% |
| Distill the 4-model ensemble into one model, distill weight 0.5 / 1.0 | 94.50% / 96.25% |
| Third laptop seed added to a 2-model ensemble | +0.13 points |
| Uncertainty-weighted temporal ensembling | −2.0 points |
| History frames 8 steps apart | closed-loop panel 48 vs 56 of 60 |

Every longer run lost, whatever the learning rate: the per-stage language FiLM
overfits with long training. The distilled students both failed one task the
others solve (BBQ sauce in basket, 15% / 65%) even though the teacher targets
there are as accurate as elsewhere. The next lever is the action head: L1
regression averages modes at the grasp decision, which matches both Long
failures; MINERVA uses flow matching.

The autoresearch log (every run, kept or discarded) is
`artifacts/autoresearch_log.tsv`.

## Kaggle

Most of the November runs trained and benchmarked on Kaggle (2× T4, 12 h
sessions, ~30 GPU-hours a week). `kaggle/launch.py` writes a kernel from
`kaggle/run_template.py` and pushes it: each experiment trains on one T4
(`--budget-steps`, so a run is the same length on any GPU) and is then
benchmarked on it. `--first-init` splits a 50/task bench across both GPUs.

```bash
git archive HEAD lc_act | tar -x -C <staging>   # then: kaggle datasets version -p <staging> -r tar
python kaggle/launch.py r8 '{"experiments": [{"tag": "base", "args": ["--budget-steps", "24656", "--seed", "4"]}]}'
```

It needs three private datasets (`lc-act-code`, the 4 GB `lc-act-libero128`
frame cache, and `lc-act-ckpts` for ensemble members). Two Kaggle traps are
handled in the template: `egl_probe` does not build on Python 3.13, so
`hf-libero` and `robomimic` install with `--no-deps`; and Kaggle registers only
Mesa's EGL vendor, so MuJoCo would render on the CPU at 11 frames/s until
`__EGL_VENDOR_LIBRARY_FILENAMES` points at NVIDIA's library (300 frames/s).

## History

- 2026-09-14: the first model (28.5M trainable, 3 epochs) scored 0/10 soup.
- 2026-09-23: that 0/10, and later 0/10 results, came from an eval bug. The
  eval executed every action twice, but one dataset row is one env step. With
  one step per action, the 3-epoch five-encoder/three-decoder checkpoint scores
  10/10 soup (2/10 with the old repeat, same checkpoint and seeds).
- 2026-09-24: an autoresearch pass shrank the model to 7.6M trainable
  parameters (128 px, 256-wide transformer) while raising all-task success from
  62% to 86% open-loop, and 95% with temporal ensembling.
- 2026-09-27: all 40 tasks from a JPEG frame cache (830 vs 220 samples/s),
  two-frame history, batch 64, and training-only alignment to frozen DINOv2
  features of the current frame and the frame one chunk ahead. 91.75% over the
  four suites (800 rollouts).
- 2026-10-12: language FiLM in every ResNet stage, chosen by a closed-loop
  panel instead of val L1, and trained 40 minutes instead of hours: 94.25%.
  History frames 8 steps apart had won val L1 by the widest margin of the
  loop and cost closed-loop success (panel 48 vs 56 of 60). The same model
  trained 4.5 hours scored 89.25%.
- 2026-10-14: a second seed scored 95.00%, and averaging the two models'
  chunks scored 96.62% (Long 92.0%). A third seed added only 0.13 points.
- 2026-11-02: runs moved to Kaggle. A fourth seed scored 97.12%, and the
  4-model ensemble 98.12% on 20/task. Learning rate, run length, a progress
  head and ensemble distillation all stayed inside seed noise or lost.
- 2026-11-03: the 4-model ensemble at MINERVA's 50/task protocol: 97.15%
  (MINERVA 97.45%). Lower learning rate with 2.5× longer training lost
  (94.00%, 93.75%).

The earlier Object-only model (`outputs/lc_act/last.pt`, commit `be73e88`)
scored 95% on the ten Object tasks, measured without a step cap.
