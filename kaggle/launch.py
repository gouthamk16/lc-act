"""Launch a Kaggle round: python kaggle/launch.py <slug> '<config json>' [kernel_source ...]  (pushes gouthamk16/lc-act-<slug>)."""
import json
import subprocess
import sys
from pathlib import Path

D = Path(__file__).parent
slug, config, sources = sys.argv[1], json.loads(sys.argv[2]), sys.argv[3:]
folder = D / "rounds" / slug
folder.mkdir(parents=True, exist_ok=True)
(folder / "run.py").write_text((D / "run_template.py").read_text().replace("__CONFIG__", json.dumps(config)))
(folder / "kernel-metadata.json").write_text(json.dumps({
    "id": f"gouthamk16/lc-act-{slug}", "title": f"lc-act-{slug}", "code_file": "run.py",
    "language": "python", "kernel_type": "script", "is_private": "true", "enable_gpu": "true",
    "enable_tpu": "false", "enable_internet": "true", "machine_shape": "NvidiaTeslaT4",
    "dataset_sources": ["gouthamk16/lc-act-code", "gouthamk16/lc-act-libero128", *config.get("datasets", [])],
    "competition_sources": [], "kernel_sources": sources, "model_sources": [],
}, indent=1))
subprocess.run(["python3", "-m", "py_compile", str(folder / "run.py")], check=True)
push = subprocess.run([str(Path.home() / ".local/bin/kaggle"), "kernels", "push", "-p", str(folder),
                       "--accelerator", "NvidiaTeslaT4"], capture_output=True, text=True)
print(push.stdout.strip(), push.stderr.strip())
