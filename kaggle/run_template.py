"""One Kaggle round: train up to two LC-ACT experiments in parallel (one per T4), then bench each on its GPU.

CONFIG is injected by the launcher: {"experiments": [{"tag", "overrides", "args", "episodes", "envs", "first_init", "ckpts"}]}.
An experiment without "args" skips training and benches "ckpts" (paths under /kaggle/input or /kaggle/working).
"distill" (teacher checkpoint globs) needs lc_act.distill, which lives on the exp/distill branch.
"""
import glob
import json
import os
import subprocess
import sys
import time

CONFIG = json.loads(r'''__CONFIG__''')
WORK = "/kaggle/working"
status = {"started": time.strftime("%H:%M:%S")}


def note(key, value):
    status[key] = value
    print(f"== {key}: {value}", flush=True)
    with open(f"{WORK}/status.json", "w") as f:
        json.dump(status, f, indent=1)


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()


def setup():
    # egl_probe (a robomimic dependency) fails to build on Kaggle and only serves robomimic's own env wrappers.
    sh("pip install -q 'lerobot[dataset,transformers-dep,scipy-dep]==0.6.1' 'transformers==5.5.4' av imageio imageio-ffmpeg "
       "hydra-core robosuite==1.4.0 bddl==1.0.1 easydict einops thop matplotlib cloudpickle opencv-python "
       "'gymnasium>=0.29' 'mujoco<3.9' future h5py psutil termcolor tensorboardX && "
       "pip install -q --no-deps hf-libero==0.1.4 robomimic==0.2.0")
    # Kaggle mounts the code dataset with its top folder stripped; rebuild the package directory.
    src = os.path.dirname(glob.glob("/kaggle/input/**/lc-act-code/bench.py", recursive=True)[0])
    sh(f"rm -rf {WORK}/code && mkdir -p {WORK}/code && cp -r {src} {WORK}/code/lc_act")
    # Kaggle ships NVIDIA's EGL library but registers only Mesa's, so MuJoCo would render on the CPU (11 vs 300 fps).
    with open(f"{WORK}/10_nvidia.json", "w") as f:
        json.dump({"file_format_version": "1.0.0", "ICD": {"library_path": "/usr/local/nvidia/lib64/libEGL_nvidia.so.0"}}, f)
    with open(f"{WORK}/trainx.py", "w") as f:
        f.write(
            "import os, sys\nimport lc_act.train as train\n"
            "if __name__ == '__main__':\n"
            "    for k, v in os.environ.items():\n"
            "        if k.startswith('TX_'):\n"
            "            setattr(train, k[3:], type(getattr(train, k[3:]))(v)); print('override', k[3:], v, flush=True)\n"
            "    train.main(sys.argv[1:])\n"
        )
    return os.path.dirname(glob.glob("/kaggle/input/**/workspace.jpg.bin", recursive=True)[0])


def env_for(gpu, overrides):
    env = dict(os.environ, PYTHONPATH=f"{WORK}/code", CUDA_VISIBLE_DEVICES=str(gpu), MUJOCO_GL="egl",
               MUJOCO_EGL_DEVICE_ID=str(gpu), __EGL_VENDOR_LIBRARY_FILENAMES=f"{WORK}/10_nvidia.json")
    env.update({f"TX_{k}": str(v) for k, v in overrides.items()})
    return env


def pipeline(gpu, exp, cache):
    """Shell chain for one experiment on one GPU: train (unless bench-only), then bench."""
    out = f"{WORK}/{exp['tag']}"
    steps, extra = [], []
    if exp.get("distill"):
        teachers = [sorted(glob.glob(c, recursive=True))[0] for c in exp["distill"]]
        steps.append(" ".join([sys.executable, "-u", "-m", "lc_act.distill", "--ckpt", *teachers, "--cache", cache,
                               "--out", f"{out}/distill.npy", "--workers", "3"]) + f" > {out}/distill.log 2>&1")
        extra = ["--distill", f"{out}/distill.npy"]
    if exp.get("args"):
        steps.append(" ".join([sys.executable, "-u", f"{WORK}/trainx.py", "--cache", cache, "--out", out, "--epochs", "999",
                               "--max-hours", "0", "--save-every", "0", "--workers", "2", *exp["args"], *extra])
                     + f" > {out}/train.log 2>&1")
    # ckpts are glob patterns: Kaggle mount paths for datasets and kernel outputs are not fixed.
    ckpts = [sorted(glob.glob(c, recursive=True))[0] for c in exp["ckpts"]] if exp.get("ckpts") else [f"{out}/last.pt"]
    steps.append(" ".join([sys.executable, "-u", "-m", "lc_act.bench", "--ckpt", *ckpts, "--episodes",
                           str(exp.get("episodes", 20)), "--first-init", str(exp.get("first_init", 0)), "--envs", str(exp.get("envs", 2)), "--out", f"{out}/bench.json"])
                 + f" > {out}/bench.log 2>&1")
    os.makedirs(out, exist_ok=True)
    return subprocess.Popen(" && ".join(steps), shell=True, env=env_for(gpu, exp.get("overrides", {})))


def main():
    started = time.time()
    cache = setup()
    note("setup_seconds", round(time.time() - started))
    experiments = CONFIG["experiments"]
    procs = [pipeline(gpu, exp, cache) for gpu, exp in enumerate(experiments)]
    note("exit", {exp["tag"]: proc.wait() for exp, proc in zip(experiments, procs)})
    for exp in experiments:
        path = f"{WORK}/{exp['tag']}/bench.json"
        if os.path.exists(path):
            note(f"result_{exp['tag']}", json.load(open(path))["suites"])
    note("total_minutes", round((time.time() - started) / 60, 1))


main()
