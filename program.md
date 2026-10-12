# autoresearch

This is an experiment to have the LLM do its own research on LC-ACT.

LC-ACT is a small language-conditioned action-chunking transformer for
LIBERO (Franka in MuJoCo). Since `sep25` it trains on **all 40 tasks** of the
four standard suites (Spatial, Object, Goal, Long) from a pre-decoded 128 px
frame cache, and the target is MINERVA (arXiv:2609.03715): 97.45% average over
2,000 rollouts with 9.66M parameters. The keep metric is
**validation L1** on a held-out split (the last 10% of each task's episodes). Each val line also reports
**raw L1** (denormalized env units) and **grip** (gripper sign match after
invert) so a z-scored L1 win that would fail soup can be rejected. Do not run
10-episode soup inside the 5-minute loop.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `sep15`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current main.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — lab context and [Starscream](https://github.com/gouthamk16/starscream) mapping.
   - `MODEL.md` — canonical architecture card.
   - `lc_act/eval.py` / `lc_act/bench.py` — closed-loop protocol (replan every step, temporal ensembling; bench runs every task of each suite at init states 0..N-1 with LeRobot's per-suite step caps). **Do not modify** the protocol.
   - `lc_act/cache.py` — frame cache and per-task holdout split. **Do not modify** the split or sample layout (frozen like Karpathy `prepare.py`); build once with `python -m lc_act.cache`.
   - `lc_act/train.py` / `lc_act/model.py` / `lc_act/posenc.py` / `lc_act/types.py` — the files you modify.
4. **Verify data exists**: HuggingFace `lerobot/libero` should already be cached from prior trains. If load fails, tell the human.
5. **Initialize results.tsv**: Create `artifacts/results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU. The training script runs for a **fixed time budget of 10 minutes** (wall clock training time, excluding startup / CLIP download / dataset open). All 40 tasks are 4x the Object data, so 5 minutes no longer covers an epoch. Launch from the repo root:

```
python -m lc_act.train --budget-seconds 600 --epochs 99 --max-hours 0 --save-every 0 --out /tmp/lc_act_ar
```

`--budget-seconds 600` arms the 10-minute clock after the first train batch. `--out /tmp/lc_act_ar` keeps AR junk off the overnight checkpoint.

**Priority (this lab):** Prefer **novel architectural changes** over configuration knobs. Architecture first: attention / residual paths, token layout, cameras, horizon/chunking, aux heads, conditioning (FiLM, extra language tokens), posenc, encoder/decoder depth. LR / batch / weight decay / warmup only after architectural ideas stall. A 0.002 val L1 win from `lr *= 2` is weaker evidence than a smaller win from a cleaner architecture.

**What you CAN do:**
- Modify `lc_act/train.py`, `lc_act/model.py`, `lc_act/posenc.py`, `lc_act/types.py`. Architecture, optimizer, hyperparameters, training loop, batch size, model size.

**What you CANNOT do:**
- Install new packages or add dependencies. You can only use what's already in `requirements.txt` / `pyproject.toml`.
- Modify the eval protocol in `lc_act/eval.py` / `lc_act/bench.py`.
- Modify the holdout split or sample layout in `lc_act/cache.py`.
- Overwrite `outputs/lc_act/last.pt` from the overnight 20-epoch run.

**The goal is simple: get the lowest val L1.** Since the time budget is fixed, you don't need to worry about training time — it's always 5 minutes. The only constraint is that the code runs without crashing and finishes within the time budget.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful val L1 gains, but it should not blow up dramatically (laptop 8 GB).

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.001 val L1 improvement that adds 20 lines of hacky code? Probably not worth it. A 0.001 val L1 improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Output format

The script prints epoch train L1 during the budget, then one grep-able eval line:

```
Step 120 : train 0.5421 | val 0.5503 | raw 0.1420 | grip 0.810 | gpu 2.10GB
```

- `val` — z-scored L1 (the training objective). Lower is better. This is the keep/discard metric.
- `raw` — L1 after `invert_action`, in env units. Lower is better. Use it to catch z-score gaming.
- `grip` — fraction of chunk steps whose denormalized gripper sign matches the demo. Higher is better.

Keep a change only if `val` is lower **and** `grip` does not drop by more than 0.03 vs the current champion. If `val` is slightly better but `raw` is much worse, discard.

Extract the final val line:

```
grep " | val " run.log | tail -1
```

## Logging results

When an experiment is done, log it to `artifacts/results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 7 columns:

```
commit	val_loss	raw_l1	grip_acc	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. val_loss achieved (e.g. 1.234567) — use 0.000000 for crashes
3. raw_l1 (denormalized) — use 0.000000 for crashes; `-` for rows logged before this column existed
4. grip_acc (0-1) — use 0.000 for crashes; `-` for older rows
5. peak memory in GB, round to .1f — use 0.0 for crashes
6. status: `keep`, `discard`, or `crash`
7. short text description of what this experiment tried

Example:

```
commit	val_loss	raw_l1	grip_acc	memory_gb	status	description
a1b2c3d	0.410000	0.120000	0.810	2.1	keep	baseline
b2c3d4e	0.390000	0.110000	0.840	2.2	keep	QK-norm on encoder/decoder
c3d4e5f	0.430000	0.130000	0.800	2.1	discard	drop wrist camera tokens
d4e5f6g	0.000000	0.000000	0.000	0.0	crash	double d_model (OOM)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/sep15`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `lc_act/train.py` / `lc_act/model.py` / `lc_act/posenc.py` with an experimental idea by directly hacking the code. Prefer architecture.
3. git commit (author gouthamk16, no Cursor co-author trailer)
4. Run: `python -m lc_act.train --budget-seconds 600 --epochs 99 --max-hours 0 --save-every 0 --out /tmp/lc_act_ar > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep " | val " run.log | tail -1`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv (NOTE: do not commit the artifacts/results.tsv file, leave it untracked by git)
8. If val_loss improved (lower) **and** grip_acc did not drop more than 0.03, you "advance" the branch, keeping the git commit
9. If val_loss is equal or worse, or grip_acc collapsed, you git reset back to where you started

After a stretch of architecture experiments, start a 10–15 hour train (`--max-hours 14 --epochs 99 --out outputs/lc_act`) on the champion **without** `--budget-seconds`. Backup any existing `outputs/lc_act/last.pt` first. Do this when either val L1 is clearly better than the instrumented champion (~0.60) or ideas have stalled and the champion's raw L1 / grip look healthy. The overnight run is the real bet; 5-minute val never replaces soup eval.

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should take ~10 minutes of train time (+ startup). If a run exceeds 20 minutes, kill it and treat it as a failure (discard and revert).

**Closed-loop check**: val L1 cannot see inference-time changes (ensembling, uncertainty) or compounding error. For those, and for any keep that changes the model's inputs, run `python -m lc_act.bench --ckpt /tmp/lc_act_ar/last.pt --episodes 5` (200 rollouts) against the champion's number.

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers cited in MODEL.md, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

## Phase 2 (oct12): closed-loop keep metric

Val L1 stopped predicting rollouts. History frames 8 steps apart cut val L1
by 0.020 (the biggest win of the loop) yet were neutral in closed loop at 10
minutes and cost four tasks in the long run; 10-minute models already reach
~89% closed loop against 91.75% for 4 hours. Phase 2 keeps or discards on
rollouts instead.

**Panel.** Eight tasks, init states 0–9 each (80 rollouts, ~25 min with 2 envs):

- target (Goal/Long, where MINERVA leads by 5–10 points): `libero_10/8` (both
  moka pots), `libero_10/7`, `libero_10/6`, `libero_goal/3` (top drawer + bowl),
  `libero_goal/0` (middle drawer), `libero_goal/6`
- guard (tasks the stride/FiLM run broke): `libero_spatial/5`, `libero_object/1`

**Run.** 40 minutes of training (`--budget-seconds 2400`, holdout split, so
val L1 is still printed), then `lc_act.bench --tasks <panel> --episodes 10`.
Same inits for every experiment, so results compare pair by pair.

**Keep** when the 60 target rollouts gain at least 3 successes over the
champion and the 20 guard rollouts lose at most 2. Record target, guard,
paired wins/losses and val L1 in `results.tsv`.

**Order.** (1) baseline: language FiLM in every stage + stride 8; (2) stride 1;
(3) stride 4; (4) LIBERO-Long oversampled 3x; (5) larger action decoder;
(6) uncertainty ensembling on the best checkpoint (bench flag only). Then a
4.5-hour run of the winner and the full four-suite bench (20 per task).
