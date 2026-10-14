# LC-ACT

**Status:** approved. Canonical card for the code in this folder. If implementation drifts, update this file in the same change.

Language-conditioned action chunking transformer. Small imitation policy for LIBERO (Franka in MuJoCo), 8.0M parameters at inference. Trains on all 40 tasks of the four standard suites by default (`--object-only` for the Object suite). Not a 7B VLA. Not the x500.

| | |
| --- | --- |
| Name | LC-ACT |
| Package | `lc_act` |
| Primary eval | `lc_act.bench`: every task of LIBERO-Spatial, Object, Goal and Long at init states 0..N-1, LeRobot step caps (MINERVA reports N = 50) |
| Target | MINERVA-9.66M: 98.4 / 99.0 / 98.4 / 94.0, 97.45 average (π₀.₅ via LeRobot: 97.5) |
| Recipe | ACT (Zhao et al., 2023) + OFT L1/chunks (Kim et al., 2025) + LIBERO ResNet-T language (Liu et al., 2023) + MINERVA small-policy findings (Sendai et al., 2026) + training-only alignment to cached frozen DINOv2 features (this work) |

## Why this shape

Collapsing each camera to one CLIP vector throws away grasp geometry. ACT and LIBERO’s ResNet-T keep a **spatial feature map**. OpenVLA-OFT on LIBERO:

- Parallel decode + **action chunking**: +14 average success vs autoregressive OpenVLA (76.5% → 90.2%).
- **Continuous L1** vs discrete tokens: +5 (→ 95.3%). L1 matched diffusion; diffusion was slower.
- Wrist + proprio + language: **98.4% LIBERO-Object** on a 7B model (not ours).

Scratch Diffusion Policy already hits **92.5% Object**. MINERVA reaches 99.6% Object with a 0.54M-parameter policy at 128 px, where L1 regression matched flow matching and replan-every-step temporal ensembling beat long open-loop chunks. We need spatial chunks + L1, not a giant VLM.

Rejected on this 8 GB laptop: OpenVLA-OFT 7B (25–62 GB), SmolVLA (10–16 GB), MiniVLA/TinyVLA VLMs, CLIP-CLS TinyVLA, CVAE, flow matching.

## Architecture

The diagram matches the code, including token tagging.

```
workspace RGB 256×256 ──┐  resize to 128×128 (train: ±8 px random shift)
                        ├── shared ResNet-18 through layer3 (trainable),
                        │   each stage FiLM'd by the language token
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
      → linear → (16, 7) relative pose + gripper      → L1 vs z-scored demo chunk
      → MLP on detached features → (16, 7) log-scale  → Laplace NLL (inference weighting only)

training only:
  mean-pooled encoder tokens per camera → 2 MLPs → DINOv2-S features of this frame
                                                   and of the frame 15 steps ahead → cosine
```

Two observation frames, 8 env steps apart (the default: `--n-obs 2`, `OBS_STRIDE = 8` in `train.py`, saved in the checkpoint): each sample carries the frame and state from 8 steps earlier plus the current ones (oldest first; at episode start the first frame repeats). Adjacent frames barely differ; frames a chunk apart show progress, which multi-step Long tasks need to tell which half they are in. Both frames go through the shared ResNet with one shared random shift, and each token concatenates the two frames' 256-d features before the linear projection, so the token count stays the same. The state token projects the 2×8 stacked states. On the 10-minute proxy this cut held-out L1 from 0.3249 to 0.3048 on every suite. Because a history policy can score well offline by extrapolating past motion and then react badly in closed loop ("copycat"), it was checked closed-loop before keeping: 90.0% vs 88.8% on 80 paired rollouts, so no harm, but no clear gain at that budget either.

**Language FiLM in every ResNet stage.** The language token also produces a zero-initialized (γ, β) per channel for the outputs of layer1, layer2 and layer3, so the instruction changes what the vision features pick out, not only how the finished tokens are weighted. Goal-suite tasks share one scene and differ only in the instruction, and that suite moved most (held-out L1 0.3548 → 0.3492; overall 0.3277 → 0.3249). On Object alone (Sept 2026) the same change was neutral, because language barely matters there.

**Teacher alignment and latent foresight (the novel piece).** `python -m lc_act.teacher` runs frozen DINOv2-small once over every cached frame and stores its mean-pooled patch features (420 MB). Training adds a cosine loss (weight 0.5) that makes each camera's pooled encoder tokens predict that feature for the current frame and for the frame one chunk ahead. The first is REPA-style representation alignment; the second asks the encoder to anticipate where the scene is going, which is the part a chunked policy needs. It combines two 2026 results: world-model representation distillation into 0.8B VLAs (cached teacher features, cosine alignment, 95.3 → 97.9% LIBERO) and MINERVA's finding that small LIBERO policies are limited by vision capacity, not size. The heads cost 0.66M parameters in training and nothing at inference. On the 10-minute proxy it improved held-out L1 on all four suites (0.3300 → 0.3273). A spatial variant (4×4 DINOv2 patch grid, PCA-64) tied and was dropped.

Training: AdamW, 2% warmup then cosine decay from a 1.2e-3 peak over the run, batch 64, channels-last ResNet, 6 decode workers reading a q95 JPEG frame cache (`python -m lc_act.cache`; ~830 samples/s on an RTX 4060 Laptop).

Without the 2-D sine and camera tags, the encoder is permutation-invariant and cannot use patch location or which camera a token came from.

**Inference:** replan every env step and execute the ACT temporal ensemble (decay 0.01, older predictions weigh more) of all chunks covering that step. `lc_act.bench --ensemble uncertainty` also divides each chunk's vote by its predicted Laplace scale per action dimension, so a confident fresh prediction outvotes an unsure old one (aimed at the fine positioning just before a grasp). At 7 ms per call this fits a 20 Hz loop. The model also sees the previous frame and state; at episode start the first frame is repeated, as in training. The dataset's 10 fps is a label only: its longest Object episode is 254 rows, matching openpi's 254-env-step count, so one row is one control step. Repeating each action twice (the old eval) doubled every motion: the same 3-epoch checkpoint scored 2/10 soup with the repeat and 10/10 without it.

| Piece | Params | Train? |
| --- | --- | --- |
| ResNet-18 through layer3 | 2,782,784 | yes |
| Projections, tags, token and stage FiLM, transformer, action + scale heads | 5,179,534 | yes |
| Teacher-alignment heads | 657,152 | training only |
| CLIP text | ~63M | frozen |
| Inference total | 7,962,318 | |
| Train memory | 1.7 GB peak at batch 64 (RTX 4060 Laptop 8 GB) | |

Current weights: `outputs/lc_act_film40/last.pt` (2026-10-12, stage FiLM, stride 1, 40 minutes on the 90% autoresearch split). `lc_act.bench`, 20 init states per task (800 rollouts): Spatial 97.5, Object 92.0, Goal 99.0, Long 88.5, **average 94.25%**. A second seed (`outputs/ar_ckpt/p2s1_seed1/last.pt`) scored 97.5 / 95.0 / 98.5 / 89.0, **95.00%**, so the configuration reproduces; the weaker all-data runs (89.75% at 40 minutes, 86.88% at 80) are unexplained. The seeds fail on different tasks (tomato sauce 40 vs 95%), and averaging their chunks (`--ckpt` with both) scores 98.0 / 97.5 / 99.0 / 92.0, **96.62%**, against MINERVA-9.66M's 97.45, at 16.0M inference parameters. A third seed (23.9M) scored 99.0 / 97.5 / 98.0 / 92.5, 96.75%: ensembling has saturated, and the remaining failures (both moka pots 70%, mug plus pudding 85%, salad dressing 85%) are shared by every seed.

Earlier: `outputs/lc_act_all40/last.pt` (2026-09-27, no stage FiLM, 4 h): 95.5 / 98.0 / 91.0 / 82.5, average 91.75%. `outputs/lc_act_oct12/last.pt` (this config trained 4.5 h): 88.5 / 91.0 / 92.0 / 85.5, average 89.25% — long training overfits the per-stage language conditioning.

Follow-up weights: `outputs/lc_act_oct11/last.pt` (2026-10-12, stage FiLM + stride-8 history, 4.5 h / 44 epochs): 92.5 / 95.5 / 93.0 / 84.5, average 91.38% — a tie; Goal and Long improved, Spatial and Object regressed. With `--ensemble uncertainty`: 89.38% (paired 24 wins / 40 losses vs decay), so uncertainty weighting is not used.

Older Object-only weights: `outputs/lc_act/last.pt` (commit `be73e88`, single frame): 95% over the 10 Object tasks with `lc_act.eval` (no step cap).

## Data and eval

- Dataset: [`lerobot/libero`](https://huggingface.co/datasets/lerobot/libero), all 40 tasks (1,693 episodes, 273k frames); dataset task indices 0–9 are LIBERO-Long, 10–19 Goal, 20–29 Object, 30–39 Spatial. `python -m lc_act.cache` decodes the AV1 videos once, resizes to 128 px with the model's own filter (bit-exact against LeRobot's decoder), and re-encodes as q95 JPEG (3.6 GB, ~1/255 mean error, under the dataset's own AV1 loss), so random access costs a JPEG decode instead of a video seek: 830 vs 220 samples/s.
- Autoresearch holdout: the last 10% of each task's episodes (168 episodes, every 8th frame), so every suite is in val. The long run trains on everything.
- History: the previous row of the same episode; at episode start the first frame repeats.
- Normalize state/action with the training subset's mean/std; store stats in the checkpoint.
- Eval images: flip H and W (dataset is 180° from raw MuJoCo).
- Env: LeRobot `LiberoEnv`, relative 7-D, init state k for the k-th episode of a task, LeRobot's step caps (Spatial 280, Object 280, Goal 300, Long 520). `lc_act.bench` steps N envs in worker processes and batches the policy. Each env holds ~2.9 GB (MuJoCo's constraint arena for robosuite's njmax = nconmax = 5000; LIBERO really uses up to 4,914 rows, so it cannot shrink), so the 10 GB WSL box runs 2 envs at ~16 env steps/s. Under WSL2, EGL falls back to llvmpipe; the bench renders through GLFW + Mesa D3D12 on the GPU instead (frames match llvmpipe to 0.5/255 on average).
- The earlier Object numbers (95%) used `lc_act.eval`, which had no step cap beyond robosuite's own horizon; `lc_act.bench` enforces LeRobot's caps.

## Intended use / not

Learn image + language → action chunk → env. Same slot as [Starscream](https://github.com/gouthamk16/starscream) `HoverPolicy`, different body.

Not for drone flight, OpenVLA-level generalization, or beating 98% Object.

## Risks

- Ten short instructions: language can be ignored. Cream-cheese eval is the check.
- Keep H = 16: MINERVA found chunk 8 and chunk 32 each cost 2–3 points on LIBERO.
- WSL 10 GB RAM: 6 decode workers leave ~3.5 GB free; drop `--workers` if RAM runs out.
- Remaining failures are grasp misses on small or flat objects (cream cheese, milk) with no re-grasp: the policy finishes the memorized carry to the basket empty-handed.

## Cite

- Zhao et al., ACT, 2023.
- Kim et al., OpenVLA-OFT, 2025.
- Liu et al., LIBERO, 2023.
- Sendai et al., MINERVA: How Small Can a Manipulation Policy Be and Still Solve LIBERO?, arXiv:2609.03715, 2026.
- Think Like a World Model, Act Like a VLA: Distilling World-Model Representations into Compact Robot Policies, arXiv:2609.24682, 2026.
- Oquab et al., DINOv2, 2023. Yu et al., REPA (representation alignment), 2024.
- Wen et al., TinyVLA, RA-L 2025 — different model; name collision only.

## How to determine completion / termination criteria in realtime

Not in the current design, but common extensions:

- Option A: 8th action dim = terminate probability
- Option B: separate head: model(obs) → (16,7) + done_logit
- Option C: heuristic on chunk - if all 16 actions ≈ zero, treat as "hold/stop"

None of these exist in LC-ACT now. Training also doesn't label “done” - demos just end when the expert finishes, and late frames get padded with the last action, not a stop signal.