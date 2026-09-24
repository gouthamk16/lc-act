# LC-ACT

![LC-ACT model](assets/lcact-model.png)
---

**Status:** approved. Canonical card for the code in this folder. If implementation drifts, update this file in the same change.

Language-conditioned action chunking transformer. Small imitation policy for LIBERO-Object (Franka in MuJoCo). Not a 7B VLA. Not the x500.

| | |
| --- | --- |
| Name | LC-ACT |
| Package | `lc_act` |
| Primary eval | “pick up the alphabet soup and place it in the basket” |
| Done bar | ≥2 / 10 soup episodes + a rollout video |
| Recipe | ACT (Zhao et al., 2023) + OFT L1/chunks (Kim et al., 2025) + LIBERO ResNet-T language (Liu et al., 2023) |

## Why this shape

Collapsing each camera to one CLIP vector throws away grasp geometry. ACT and LIBERO’s ResNet-T keep a **spatial feature map**. OpenVLA-OFT on LIBERO:

- Parallel decode + **action chunking**: +14 average success vs autoregressive OpenVLA (76.5% → 90.2%).
- **Continuous L1** vs discrete tokens: +5 (→ 95.3%). L1 matched diffusion; diffusion was slower.
- Wrist + proprio + language: **98.4% LIBERO-Object** on a 7B model (not ours).

Scratch Diffusion Policy already hits **92.5% Object**. We need spatial chunks + L1, not a giant VLM.

Rejected on this 8 GB laptop: OpenVLA-OFT 7B (25–62 GB), SmolVLA (10–16 GB), MiniVLA/TinyVLA VLMs, CLIP-CLS TinyVLA, CVAE, flow matching.

## Architecture

![LC-ACT architecture](assets/architecture.png)

The PNG is a schematic. Token tagging below matches the code.

```
workspace RGB 256×256 ──┐  resize to 128×128 (train: ±8 px random shift)
                        ├── shared ResNet-18 through layer3 (trainable)
wrist RGB 256×256 ──────┘         │
                                  ▼
                         8×8 spatial tokens per camera (256-d)
                         + 2-D sinusoidal pos embed (ACT)
                         + learned camera id (workspace vs wrist)
                         FiLM (γ, β) from the language token
instruction ── frozen CLIP text (cached per string) ──► 1 language token + type tag
proprio 8-D ── linear ────────────► 1 state token + type tag

concat → transformer encoder (d=256, 8 heads, 3 pre-norm layers, FFN 1024)
      → 16 action queries, 2-layer decoder (cross-attend)
      → linear → (16, 7) relative pose + gripper
      → L1 vs z-scored demo chunk
```

Training: AdamW, 2% warmup then cosine decay from a 6e-4 peak over the run, batch 16, 6 decode workers.

Without the 2-D sine and camera tags, the encoder is permutation-invariant and cannot use patch location or which camera a token came from.

**Inference:** replan every env step and execute the ACT temporal ensemble (decay 0.01, older predictions weigh more) of all chunks covering that step. At 7 ms per call this fits a 20 Hz loop. With `--n-obs 2` the model also sees the previous frame and state; at episode start the first frame is repeated, as in training. The dataset's 10 fps is a label only: its longest Object episode is 254 rows, matching openpi's 254-env-step count, so one row is one control step. Repeating each action twice (the old eval) doubled every motion: the same 3-epoch checkpoint scored 2/10 soup with the repeat and 10/10 without it.

| Piece | Params | Train? |
| --- | --- | --- |
| ResNet-18 through layer3 | 2,782,784 | yes |
| Projections, tags, FiLM, transformer, head | 4,814,087 | yes |
| CLIP text | ~63M | frozen |
| Trainable total | 7,596,871 | |
| Train memory | 0.6 GB peak at batch 16 (RTX 4060 Laptop 8 GB) | |
| Inference | 7.1 ms per chunk, 0.30 GB VRAM (batch 1, fp32; CLIP is most of it) | |

Current weights: `outputs/lc_act/last.pt` (2026-09-24, commit `be73e88`, Object suite only, single frame, 3 h / 36 epochs). Closed loop over 10 Object tasks: **95%** (95/100, 10 episodes per task) with temporal ensembling; 86% (43/50, 5 per task) executing full 16-step chunks open-loop. Mean 135 env steps per success.

Older weights (`last_3epoch_5enc3dec.pt`: 512-d, 5+3 layers, 256 px, 40.6M trainable) reach 62% on the same 50 episodes and do not load into this architecture.

## Data and eval

- Dataset: [`lerobot/libero`](https://huggingface.co/datasets/lerobot/libero) (~1.9 GB). LeRobot `video_backend="pyav"`; the installable package is `av`. Keep `pick up the .+ and place it in the basket`.
- Normalize state/action with subset mean/std; store stats in the checkpoint.
- Eval images: flip H and W (dataset is 180° from raw MuJoCo).
- Env: single gym `LiberoEnv`, `libero_object` task 0, relative 7-D, 280 steps.

## Intended use / not

Learn image + language → action chunk → env. Same slot as Starscream `HoverPolicy`, different body.

Not for drone flight, OpenVLA-level generalization, or beating 98% Object.

## Risks

- Ten short instructions: language can be ignored. Cream-cheese eval is the check.
- If the arm overshoots, drop H to 8 before adding diffusion.
- WSL 10 GB RAM: 6 decode workers leave ~3.5 GB free; drop `--workers` if RAM runs out.
- Remaining failures are grasp misses on small or flat objects (cream cheese, milk) with no re-grasp: the policy finishes the memorized carry to the basket empty-handed.

## Cite

- Zhao et al., ACT, 2023.
- Kim et al., OpenVLA-OFT, 2025.
- Liu et al., LIBERO, 2023.
- Wen et al., TinyVLA, RA-L 2025 — different model; name collision only.

## How to determine completion / termination criteria in realtime

Not in the current design, but common extensions:

- Option A: 8th action dim = terminate probability
- Option B: separate head: model(obs) → (16,7) + done_logit
- Option C: heuristic on chunk - if all 16 actions ≈ zero, treat as "hold/stop"

None of these exist in LC-ACT now. Training also doesn't label “done” - demos just end when the expert finishes, and late frames get padded with the last action, not a stop signal.