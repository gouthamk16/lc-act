"""Local web console that streams LIBERO rollouts of a trained policy and takes instructions in your own words.

python -m lc_act.sim.live
"""

from collections.abc import Sequence
import dataclasses
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
import io
import json
import pathlib
import re
import threading
from typing import Any

import imageio.v2 as imageio
import numpy as np
import torch
import tyro

from lc_act.shared import normalize as _normalize
from lc_act.sim import env as _env
from lc_act.sim import eval as _eval
from lc_act.training import checkpoints as _checkpoints

PAGE = (pathlib.Path(__file__).parent / "live.html").read_bytes()
SUITE_LABELS = {"libero_spatial": "Spatial", "libero_object": "Object", "libero_goal": "Goal", "libero_10": "Long"}
READY = "Ready. Choose a task."
# words that do not tell tasks apart when matching a typed instruction
_FILLER = {"the", "a", "an", "and", "it", "in", "on", "of", "to", "up", "put", "pick", "place", "both"}


@dataclasses.dataclass(frozen=True)
class Args:
    ckpt: pathlib.Path = pathlib.Path("outputs/lc_act_film40/last.pt")
    bench: pathlib.Path = pathlib.Path("outputs/lc_act_film40/bench20_decay.json")  # bench summary that labels tasks
    host: str = "127.0.0.1"
    port: int = 8765


@dataclasses.dataclass(frozen=True)
class Task:
    suite: str
    task_id: int
    text: str


def main(args: Args) -> None:
    if not args.ckpt.exists():
        raise SystemExit(f"missing checkpoint: {args.ckpt}")
    console = Console(args.ckpt, args.bench)
    console.start()
    server = ThreadingHTTPServer((args.host, args.port), _handler(console))
    print(f"open http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


def resolve_prompt(query: str, tasks: Sequence[str]) -> tuple[str, str]:
    """Return (scene task, text the policy hears). The scene is the trained task sharing the most words."""
    spoken = " ".join(query.split())
    if not spoken:
        raise ValueError("type an instruction, or choose one of the trained tasks")
    exact = [task for task in tasks if task.lower() == spoken.lower()]
    if exact:
        return exact[0], spoken

    # More shared words first; then the task whose own words are most fully covered.
    words = _words(spoken)
    scored = []
    for task in tasks:
        overlap = len(words & _words(task))
        if overlap:
            scored.append(((overlap, overlap / len(_words(task))), task))
    if not scored:
        raise ValueError("no trained task matches; name an object or action from the task list")

    best = max(score for score, _ in scored)
    hits = [task for score, task in scored if score == best]
    if len(hits) > 1:
        raise ValueError("that matches more than one task: " + "; ".join(hits[:4]))
    return hits[0], spoken


def camera_strip(frame: _eval.Frame) -> np.ndarray:
    """Workspace and wrist cameras side by side, upright."""
    workspace, wrist, _ = frame
    return np.concatenate([workspace, wrist], axis=1)


class Console:
    """Owns the policy and one LIBERO env on a background thread; the HTTP handler only queues requests."""

    def __init__(self, ckpt: pathlib.Path, bench: pathlib.Path | None) -> None:
        self.ckpt = ckpt
        self.bench = json.loads(bench.read_text()) if bench is not None and bench.exists() else None
        self.tasks: list[Task] = []
        self.message = "Loading the policy and the simulator…"
        self._jpeg: bytes | None = None
        self._seq = 0
        self._cond = threading.Condition()
        self._request: tuple[str, str, str] | None = None  # (kind, scene task, spoken text)
        self._cancel = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def publish(self, frame: np.ndarray) -> None:
        encoded = _jpeg(frame)
        with self._cond:
            self._jpeg = encoded
            self._seq += 1
            self._cond.notify_all()

    def wait_frame(self, seen: int, timeout: float = 1.0) -> tuple[int, bytes | None]:
        """The latest frame once it is newer than `seen`, or the current one after `timeout`."""
        with self._cond:
            self._cond.wait_for(lambda: self._seq != seen, timeout)
            return self._seq, self._jpeg

    def submit(self, text: str) -> str:
        if not self.tasks:
            raise ValueError("still loading the simulator")
        scene, spoken = resolve_prompt(text, [task.text for task in self.tasks])
        self._queue(("run", scene, spoken), f"Starting: {spoken}")
        return spoken

    def reset_scene(self) -> None:
        self._queue(("reset", "", ""), "Resetting…")

    def suites(self) -> list[dict]:
        """Tasks grouped by suite with their bench success rates, for the page's buttons."""
        if not self.tasks:
            return []
        scores = (self.bench or {}).get("tasks", {})
        by_suite = (self.bench or {}).get("suites", {})
        return [
            {
                "label": label,
                "success": by_suite.get(suite),
                "tasks": [
                    {"text": task.text, "success": scores.get(f"{suite}/{task.task_id}")}
                    for task in self.tasks
                    if task.suite == suite
                ],
            }
            for suite, label in SUITE_LABELS.items()
        ]

    def _queue(self, request: tuple[str, str, str], message: str) -> None:
        # A new request cancels the running episode.
        with self._cond:
            self._request = request
            self.message = message
            self._cancel.set()
            self._cond.notify_all()

    def _take(self) -> tuple[str, str, str] | None:
        with self._cond:
            request = self._request
            self._request = None
            self._cancel.clear()
            return request

    def _loop(self) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, stats = _checkpoints.load_checkpoint(self.ckpt, device)
        model.eval()
        suites = _env.benchmark_dict()
        self.tasks = [Task(name, i, suites[name]().get_task(i).language) for name in SUITE_LABELS for i in range(10)]

        current = self.tasks[0]
        env = _env.make_env(current.task_id, current.suite)
        self._show_start(env)
        while True:
            request = self._take()
            if request is None:
                with self._cond:
                    self._cond.wait(timeout=0.2)
                continue
            kind, scene, spoken = request
            if kind == "reset":
                self._show_start(env)
                continue

            # switch scenes only when the matched task changes
            task = next(task for task in self.tasks if task.text == scene)
            if task != current:
                env.close()
                env = _env.make_env(task.task_id, task.suite)
                current = task
            self._run(model, stats, env, device, task, spoken)

    def _run(
        self,
        model: torch.nn.Module,
        stats: _normalize.NormStats,
        env: Any,
        device: torch.device,
        task: Task,
        spoken: str,
    ) -> None:
        label = f"{SUITE_LABELS[task.suite]} scene"
        steps = 0

        def on_frame(frame: _eval.Frame) -> None:
            nonlocal steps
            self.publish(camera_strip(frame))
            self.message = f"Running ({label}, step {steps}/{_env.max_steps(env)}): {spoken}"
            steps += 1

        success = _eval.run_episode(
            model, stats, env, device, 0, task=spoken, on_frame=on_frame, stop=self._cancel.is_set
        )
        if not self._cancel.is_set():
            verdict = "Done" if success else "Missed"
            self.message = f"{verdict} in {steps - 1} steps ({label}): {spoken}"

    def _show_start(self, env: Any) -> None:
        obs, _ = env.reset(seed=0)
        self.publish(camera_strip(_env.observe(obs)))
        self.message = READY


def _handler(console: Console) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: object) -> None:
            return

        def do_GET(self) -> None:
            if self.path == "/":
                self._send(200, "text/html; charset=utf-8", PAGE)
            elif self.path == "/tasks":
                self._json(200, {"suites": console.suites()})
            elif self.path == "/status":
                self._json(200, {"message": console.message})
            elif self.path == "/stream":
                self._stream()
            else:
                self.send_error(404)

        def do_POST(self) -> None:
            if self.path == "/reset":
                console.reset_scene()
                self._json(200, {"message": "Resetting…"})
            elif self.path == "/prompt":
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length) or b"{}")
                try:
                    task = console.submit(str(payload.get("text", "")))
                except ValueError as error:
                    self._json(400, {"message": str(error)})
                    return
                self._json(200, {"message": f"Starting: {task}"})
            else:
                self.send_error(404)

        def _stream(self) -> None:
            """MJPEG: push each new frame until the browser disconnects."""
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            seen = -1
            try:
                while True:
                    seen, jpeg = console.wait_frame(seen)
                    if jpeg is None:
                        continue
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                return

        def _json(self, code: int, payload: dict) -> None:
            self._send(code, "application/json", json.dumps(payload).encode())

        def _send(self, code: int, content_type: str, body: bytes) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+", text.lower())) - _FILLER


def _jpeg(frame: np.ndarray) -> bytes:
    buf = io.BytesIO()
    imageio.imwrite(buf, frame, format="JPEG")
    return buf.getvalue()


if __name__ == "__main__":
    main(tyro.cli(Args))
