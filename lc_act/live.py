"""Local page that streams a LIBERO rollout of the trained policy."""

from __future__ import annotations

import argparse
import io
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Sequence

import numpy as np

from lc_act.eval import _ensure_libero_config, _reset_obs, make_env, run_episode
from lc_act.obs import flip_hw
from lc_act.train import load_checkpoint

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>LC-ACT</title>
<style>
  :root { color-scheme: dark; }
  body { margin: 0; font-family: "Segoe UI", sans-serif; background: #12140f; color: #efe7d2; }
  main { max-width: 980px; margin: 0 auto; padding: 24px 20px 32px; }
  h1 { font-weight: 560; font-size: 1.4rem; margin: 0 0 4px; }
  p.note { margin: 0 0 16px; color: #b7ad96; }
  img { width: 100%; background: #000; border-radius: 8px; }
  #status { min-height: 1.4em; margin: 10px 0; }
  form { display: flex; gap: 8px; }
  input { flex: 1; font: inherit; padding: 10px 12px; border-radius: 8px; border: 1px solid #3c4033; background: #1c1f17; color: inherit; }
  button { font: inherit; padding: 10px 14px; border-radius: 8px; border: 0; background: #e2b657; color: #1a160c; cursor: pointer; }
  button.reset { background: #2a2e24; color: #efe7d2; }
  .chips { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }
  .chips button { background: #2a2e24; color: #efe7d2; }
</style>
</head>
<body>
<main>
  <h1>LC-ACT</h1>
  <p class="note">Trained LIBERO-Object policy. Type an instruction in your own words; the buttons load the ten trained tasks. The scene is chosen from the object you name. Workspace camera on the left, wrist on the right.</p>
  <img id="view" alt="MuJoCo cameras" src="/stream">
  <p id="status">Loading the policy and the simulator…</p>
  <form id="go">
    <input id="prompt" name="prompt" placeholder="pick up the alphabet soup and place it in the basket" autocomplete="off">
    <button type="submit">Run</button>
    <button type="button" class="reset" id="reset">Reset</button>
  </form>
  <div class="chips" id="chips"></div>
</main>
<script>
const status = document.getElementById("status");
const prompt = document.getElementById("prompt");
async function refresh() {
  const res = await fetch("/status");
  const body = await res.json();
  status.textContent = body.message;
}
async function tasks() {
  const res = await fetch("/tasks");
  const body = await res.json();
  if (!body.tasks.length) { setTimeout(tasks, 500); return; }
  const chips = document.getElementById("chips");
  for (const task of body.tasks) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = task.replace("pick up the ", "").replace(" and place it in the basket", "");
    button.onclick = () => { prompt.value = task; document.getElementById("go").requestSubmit(); };
    chips.appendChild(button);
  }
}
document.getElementById("go").onsubmit = async (event) => {
  event.preventDefault();
  const res = await fetch("/prompt", {method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({text: prompt.value})});
  const body = await res.json();
  status.textContent = body.message;
};
document.getElementById("reset").onclick = async () => {
  const res = await fetch("/reset", {method: "POST"});
  const body = await res.json();
  status.textContent = body.message;
};
tasks();
setInterval(refresh, 500);
</script>
</body>
</html>
"""


def resolve_prompt(query: str, tasks: Sequence[str]) -> tuple[str, str]:
    """Return (scene task, text the policy hears). The scene is one of `tasks`."""
    spoken = " ".join(query.split())
    text = spoken.lower()
    if not text:
        raise ValueError("type an instruction, or choose one of the ten tasks")
    named = [_object_name(task) for task in tasks]
    full = [task for name, task in zip(named, tasks) if name in text]
    if full:
        longest = max(len(_object_name(task)) for task in full)
        hits = [task for task in full if len(_object_name(task)) == longest]
        if len(hits) > 1:
            raise ValueError("that matches more than one object: " + ", ".join(hits))
        return hits[0], spoken
    words = set(text.split())
    partial = [task for name, task in zip(named, tasks) if words & set(name.split())]
    if len(partial) == 1:
        return partial[0], spoken
    if len(partial) > 1:
        raise ValueError("that matches more than one object: " + ", ".join(partial))
    raise ValueError("name one of the objects so the scene can load: " + ", ".join(named))


def _object_name(task: str) -> str:
    lowered = task.lower()
    head = "pick up the "
    tail = " and place it in the basket"
    if lowered.startswith(head) and lowered.endswith(tail):
        return lowered[len(head):-len(tail)]
    return lowered


def _as_uint8(image: object) -> np.ndarray:
    frame = flip_hw(image)  # type: ignore[arg-type]
    if hasattr(frame, "detach"):
        frame = frame.detach().cpu().numpy()
    return np.asarray(frame)


def camera_strip(obs: dict) -> np.ndarray:
    workspace = _as_uint8(obs["pixels"]["image"])
    wrist = _as_uint8(obs["pixels"]["image2"])
    return np.concatenate([workspace, wrist], axis=1)


def _jpeg(frame: np.ndarray) -> bytes:
    import imageio.v2 as imageio

    buf = io.BytesIO()
    imageio.imwrite(buf, frame, format="JPEG")
    return buf.getvalue()


class Console:
    def __init__(self, ckpt: Path) -> None:
        self.ckpt = ckpt
        self.tasks: list[str] = []
        self._jpeg: bytes | None = None
        self._seq = 0
        self._cond = threading.Condition()
        self._request: tuple[str, str, str] | None = None
        self._cancel = threading.Event()
        self.message = "Loading the policy and the simulator…"
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def publish(self, frame: np.ndarray) -> None:
        encoded = _jpeg(frame)
        with self._cond:
            self._jpeg = encoded
            self._seq += 1
            self._cond.notify_all()

    def latest(self) -> tuple[int, bytes | None]:
        with self._cond:
            return self._seq, self._jpeg

    def wait_frame(self, seen: int, timeout: float = 1.0) -> tuple[int, bytes | None]:
        with self._cond:
            self._cond.wait_for(lambda: self._seq != seen, timeout)
            return self._seq, self._jpeg

    def submit(self, text: str) -> str:
        if not self.tasks:
            raise ValueError("still loading the simulator")
        scene, spoken = resolve_prompt(text, self.tasks)
        with self._cond:
            self._request = ("run", scene, spoken)
            self.message = f"Starting: {spoken}"
            self._cancel.set()
            self._cond.notify_all()
        return spoken

    def reset_scene(self) -> None:
        with self._cond:
            self._request = ("reset", "", "")
            self.message = "Resetting…"
            self._cancel.set()
            self._cond.notify_all()

    def _take(self) -> tuple[str, str, str] | None:
        with self._cond:
            request = self._request
            self._request = None
            self._cancel.clear()
            return request

    def _loop(self) -> None:
        os.environ.setdefault("MUJOCO_GL", "egl")
        import torch
        from libero.libero import benchmark

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, stats = load_checkpoint(self.ckpt, device)
        model.eval()
        _ensure_libero_config()
        suite = benchmark.get_benchmark_dict()["libero_object"]()
        self.tasks = [suite.get_task(i).language for i in range(10)]
        env = make_env(0)
        current = 0
        self._show_start(env)
        self.message = "Ready. Choose a task."
        while True:
            request = self._take()
            if request is None:
                with self._cond:
                    self._cond.wait(timeout=0.2)
                continue
            kind, scene, spoken = request
            if kind == "reset":
                self._show_start(env)
                self.message = "Ready. Choose a task."
                continue
            task_id = self.tasks.index(scene)
            if env is None or current != task_id:
                if env is not None and hasattr(env, "close"):
                    env.close()
                env = make_env(task_id)
                current = task_id
            self.message = f"Running: {spoken}"

            def on_frame(_frame: np.ndarray, obs: dict) -> None:
                self.publish(camera_strip(obs))

            success, _frames = run_episode(
                model, stats, env, device, 0, task=spoken,
                on_frame=on_frame, stop=self._cancel.is_set,
            )
            if not self._cancel.is_set():
                self.message = f"{'Done' if success else 'Missed'}: {spoken}"

    def _show_start(self, env: object) -> None:
        obs = _reset_obs(env.reset(seed=0))  # type: ignore[attr-defined]
        self.publish(camera_strip(obs))


def serve(console: Console, host: str, port: int) -> None:
    handler = _handler(console)
    server = ThreadingHTTPServer((host, port), handler)
    print(f"open http://{host}:{port}", flush=True)
    server.serve_forever()


def _handler(console: Console) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: object) -> None:
            return

        def _json(self, code: int, payload: dict) -> None:
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/":
                body = PAGE.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path == "/tasks":
                self._json(200, {"tasks": console.tasks})
                return
            if self.path == "/status":
                self._json(200, {"message": console.message})
                return
            if self.path == "/stream":
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                seen = -1
                try:
                    while True:
                        seen, jpeg = console.wait_frame(seen)
                        if jpeg is None:
                            continue
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n")
                        self.wfile.write(jpeg)
                        self.wfile.write(b"\r\n")
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    return
            self.send_error(404)

        def do_POST(self) -> None:  # noqa: N802
            if self.path == "/reset":
                console.reset_scene()
                self._json(200, {"message": "Resetting…"})
                return
            if self.path != "/prompt":
                self.send_error(404)
                return
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            try:
                task = console.submit(str(payload.get("text", "")))
            except ValueError as error:
                self._json(400, {"message": str(error)})
                return
            self._json(200, {"message": f"Starting: {task}"})

    return Handler


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Watch LC-ACT in a browser")
    parser.add_argument("--ckpt", type=Path, default=Path("outputs/lc_act/last.pt"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    if not args.ckpt.exists():
        raise SystemExit(f"missing checkpoint: {args.ckpt}")
    console = Console(args.ckpt)
    console.start()
    serve(console, args.host, args.port)


if __name__ == "__main__":
    main()
