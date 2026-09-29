"""aiohttp server: serves the dashboard and streams the simulation over a websocket.

The simulation runs in its own thread (GPU calls block), producing one frame
per 14.4 ms of simulated time. Frames are binary:

    uint32 json_length | json (utf-8) | pad to 4 bytes | uint32 spiking neuron indices
"""

from __future__ import annotations

import asyncio
import json
import queue
import struct
import threading
import time
from pathlib import Path

import numpy as np
import torch
from aiohttp import WSMsgType, web

from .brain import Brain
from .connectome import Connectome
from .interface import Interface
from .world import World

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
BLOCKS_PER_FRAME = 8  # 8 x 1.8 ms = 14.4 ms of simulated time per frame (Eon synced every 15 ms)


class Sim(threading.Thread):
    def __init__(self, conn: Connectome, device: str | None = None, seed: int = 0, cpu_rng: str = "sparse"):
        super().__init__(daemon=True)
        self.conn = conn
        self.brain = Brain(conn.indptr, conn.indices, conn.weights, conn.n, device=device, seed=seed, cpu_rng=cpu_rng)
        self.iface = Interface(conn, self.brain.device)
        self.world = World(mode="chase")
        self.frame_s = BLOCKS_PER_FRAME * self.brain.D * self.brain.p.dt * 1e-3
        self.cmds: queue.Queue = queue.Queue()
        self.running = False
        self.settings = {"drive": 0.6, "assist": 0.0, "speed": 1.0}
        self.pokes: list[dict] = []
        self.listeners: list = []
        self.clients = 0
        self._stop_event = threading.Event()
        self._wake_event = threading.Event()
        self._frame_requested = True
        self._rtf = 0.0
        self._brain_ms = 0.0
        self._sps = 0.0
        self._last_senses = None
        self._last_decoded = None
        self._epoch = 0
        self._pending_counts = torch.zeros_like(self.brain.counts)
        self._pending_blocks = 0
        self._pending_senses = None
        self._pending_running = False
        self._pending_started = 0.0
        self._pending_brain_s = 0.0
        self._next_t = 0.0
        dev = self.brain.device
        self.device_name = torch.cuda.get_device_name(dev) if dev.type == "cuda" else "CPU"

    # ------------------------------------------------------------------ API
    def status(self) -> dict:
        return {"type": "status", "running": self.running, "mode": self.world.mode,
                "settings": dict(self.settings), "epoch": self._epoch}

    def init_payload(self) -> dict:
        meta = {
            "dataset": self.conn.dataset,
            "neurons": self.conn.n,
            "connections": self.conn.n_connections,
            "synapses": self.conn.n_synapses,
            "device": self.device_name + (" · Triton" if self.brain.use_triton else ""),
            "frameMs": self.frame_s * 1e3,
        }
        return {"type": "init", "world": self.world.static_state(), "meta": meta, "status": self.status(),
                **self.iface.init_payload()}

    def submit(self, cmd: dict) -> None:
        self.cmds.put(cmd)
        self._wake_event.set()

    def stop(self) -> None:
        self._stop_event.set()
        self._wake_event.set()

    # ------------------------------------------------------------------ loop
    def _handle_command(self, c: dict) -> None:
        kind = c.get("cmd")
        if kind == "run":
            on = bool(c.get("on"))
            if on != self.running or not on:
                # Commit only completed blocks, using their actual duration.
                # A paused poke frame must never become a moving world frame.
                self._finish_frame()
                self._next_t = time.perf_counter()
            self.running = on
            if not self.running:
                self.pokes.clear()
        elif kind == "reset":
            self._reset()
        elif kind == "mode":
            if c.get("mode") in ("chase", "free"):
                self.world.mode = c["mode"]
                self._reset()
        elif kind == "set":
            key, val = c.get("key"), c.get("value")
            if key in self.settings and isinstance(val, (int, float)) and np.isfinite(val):
                if key != "speed" and self.settings[key] != val:
                    self._finish_frame()
                self.settings[key] = float(val)
                if key == "speed":
                    self._next_t = time.perf_counter()
        elif kind == "poke":
            idx = self.iface.poke_group(c.get("id", ""))
            if idx is not None and len(idx):
                self._finish_frame()
                self.pokes.append({"idx": idx, "until": self.brain.t_ms + 400.0})
        elif kind == "pokeNeuron":
            index = c.get("index")
            if type(index) is not int:
                return
            idx = self.iface.poke_type_of(index)
            if idx is not None and len(idx):
                self._finish_frame()
                self.pokes.append({"idx": idx, "until": self.brain.t_ms + 400.0})
        self._frame_requested = True
        for fn in self.listeners:
            fn(json.dumps(self.status()).encode(), text=True)

    def _reset(self) -> None:
        self.world.reset()
        self.brain.reset_state()
        self.iface.reset()
        self.pokes.clear()
        self._last_senses = None
        self._last_decoded = None
        self._pending_counts.zero_()
        self._pending_blocks = 0
        self._pending_senses = None
        self._pending_brain_s = 0.0
        self._epoch += 1
        self._next_t = time.perf_counter()
        self._frame_requested = True
        self._rtf = self._brain_ms = self._sps = 0.0

    def run(self) -> None:
        brain = self.brain
        block_s = self.frame_s / BLOCKS_PER_FRAME
        self._next_t = time.perf_counter()
        while not self._stop_event.is_set():
            # Clear before draining, so a command arriving afterwards cannot
            # lose its wake-up while the thread computes or waits for pacing.
            self._wake_event.clear()
            try:
                while True:
                    self._handle_command(self.cmds.get_nowait())
            except queue.Empty:
                pass
            if self._stop_event.is_set():
                break
            active_pokes = [p for p in self.pokes if p["until"] > brain.t_ms]
            if len(active_pokes) != len(self.pokes):
                self._finish_frame()
                self.pokes = active_pokes
                self._frame_requested = True
            advance = self.running or bool(self.pokes)
            if self.clients == 0:
                self._wake_event.wait()
                self._next_t = time.perf_counter()
                continue
            if self._frame_requested:
                self._snapshot()
            if not advance:
                self._wake_event.wait()
                self._next_t = time.perf_counter()
                continue
            delay = self._next_t - time.perf_counter()
            if delay > 0:
                self._wake_event.wait(delay)
                continue

            if not self._pending_blocks:
                self._pending_started = time.perf_counter()
                self._pending_senses = self._senses()
                self._pending_running = self.running
                # Encoding also updates visual OFF-edge history, so do it once
                # per sensory frame, never once per interruptible brain block.
                brain.set_rates(self.iface.encode(self._pending_senses, self.settings, self.pokes))
                self._pending_brain_s = time.perf_counter() - self._pending_started
            t0 = time.perf_counter()
            # CPU blocks are slow enough to need intervening command checks.
            # Keep the original eight-block GPU batch to avoid extra count
            # resets, accumulation kernels, and fine-grained host pacing.
            blocks = 1 if brain.device.type == "cpu" else BLOCKS_PER_FRAME - self._pending_blocks
            self._pending_counts.add_(brain.run(blocks))
            self._pending_blocks += blocks
            self._pending_brain_s += time.perf_counter() - t0
            if self._pending_blocks == BLOCKS_PER_FRAME:
                self._finish_frame()

            # Pace the completed batch, with an interruptible wait.
            speed = self.settings["speed"]
            if speed > 0:
                self._next_t += blocks * block_s / speed
                if self._next_t < time.perf_counter() - 0.25:
                    self._next_t = time.perf_counter()
            else:
                self._next_t = time.perf_counter()

    def _senses(self) -> dict:
        if self.running or self._last_senses is None:
            self._last_senses = self.world.sense(self.frame_s)
        if self.running:
            return self._last_senses
        # Explicit pokes stimulate the brain while leaving the kart frozen.
        return dict(self._last_senses, loom=[], loom_l=0.0, loom_r=0.0, touch=0.0, sugar=0.0)

    def _finish_frame(self) -> None:
        if not self._pending_blocks:
            return
        dt_s = self._pending_blocks * self.frame_s / BLOCKS_PER_FRAME
        t0 = time.perf_counter()
        counts = self._pending_counts.cpu().numpy()
        spk_idx = np.flatnonzero(counts)
        self._last_decoded = self.iface.decode(counts, spk_idx, dt_s)
        self._pending_brain_s += time.perf_counter() - t0
        if self._pending_running:
            self.world.step(dt_s, {**self._last_decoded[0], "assist": self.settings["assist"]})
        elapsed = time.perf_counter() - self._pending_started
        self._rtf += (dt_s / max(elapsed, 1e-6) - self._rtf) * 0.1
        self._brain_ms += (self._pending_brain_s * 1e3 - self._brain_ms) * 0.1
        self._sps += (len(spk_idx) / dt_s - self._sps) * 0.1
        self._publish(self._pending_senses, spk_idx, self._last_decoded, dt_s)
        self._pending_counts.zero_()
        self._pending_blocks = 0
        self._pending_senses = None
        self._pending_brain_s = 0.0

    def _snapshot(self) -> None:
        senses = self._pending_senses if self._pending_senses is not None else self._senses()
        if self._last_decoded is None:
            counts = np.zeros(self.brain.n, dtype=np.int32)
            self._last_decoded = self.iface.decode(counts, np.array([], dtype=np.int64), self.frame_s)
        act, motor, key_spikes, class_counts = self._last_decoded
        # A status refresh does not advance filters or replay old spikes.
        decoded = ({**act, "jump": False}, motor, [0] * len(key_spikes), [0] * len(class_counts))
        self._publish(senses, np.array([], dtype=np.uint32), decoded, 0.0)

    def _publish(self, senses: dict, spk_idx: np.ndarray, decoded: tuple, dt_s: float) -> None:
        self._frame_requested = False
        act, motor, key_spikes, class_counts = decoded
        advance = dt_s > 0
        frame = {
            "nSpk": len(spk_idx),
            "world": self.world.dynamic_state(),
            "senses": {k: round(float(senses[k]), 3) for k in ("target_l", "target_r", "loom_l", "loom_r", "sugar")},
            "eye": np.round(senses["eye"] * 255).astype(int).ravel().tolist(),
            "motor": motor,
            "act": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in act.items()},
            "keySpikes": key_spikes,
            "classCounts": class_counts,
            "perf": {"rtf": self._rtf if advance else 0.0, "brainMs": self._brain_ms if advance else 0.0,
                     "spikesPerSec": self._sps if advance else 0.0},
            "tMs": round(self.brain.t_ms, 1),
            "frameMs": round(dt_s * 1e3, 3),
            "running": self.running,
            "epoch": self._epoch,
        }
        payload = json.dumps(frame, separators=(",", ":")).encode()
        pad = (-(4 + len(payload))) % 4
        msg = struct.pack("<I", len(payload)) + payload + b" " * pad + spk_idx.astype(np.uint32).tobytes()
        for fn in self.listeners:
            fn(msg, text=False)


class Hub:
    """Fan frames out from the sim thread to every websocket client."""

    def __init__(self, sim: Sim, loop: asyncio.AbstractEventLoop):
        self.sim = sim
        self.loop = loop
        self.clients: set[web.WebSocketResponse] = set()
        sim.listeners.append(self._from_thread)

    def _from_thread(self, data: bytes, text: bool) -> None:
        self.loop.call_soon_threadsafe(self._broadcast, data, text)

    def _broadcast(self, data: bytes, text: bool) -> None:
        for ws in list(self.clients):
            if ws.closed:
                continue
            # Drop frames for slow clients instead of queueing forever.
            transport = ws._req.transport if ws._req else None
            if not text and transport is not None and transport.get_write_buffer_size() > 2_000_000:
                continue
            if text:
                asyncio.ensure_future(ws.send_str(data.decode()))
            else:
                asyncio.ensure_future(ws.send_bytes(data))


async def ws_handler(request: web.Request) -> web.WebSocketResponse:
    hub: Hub = request.app["hub"]
    ws = web.WebSocketResponse(heartbeat=20, max_msg_size=1 << 20)
    await ws.prepare(request)
    await ws.send_str(json.dumps(hub.sim.init_payload()))
    hub.clients.add(ws)
    hub.sim.clients = len(hub.clients)
    hub.sim.submit({"cmd": "snapshot"})
    try:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                try:
                    command = json.loads(msg.data)
                except json.JSONDecodeError:
                    continue
                if isinstance(command, dict):
                    hub.sim.submit(command)
    finally:
        hub.clients.discard(ws)
        hub.sim.clients = len(hub.clients)
        if not hub.clients:
            hub.sim.submit({"cmd": "run", "on": False})  # next visitor starts paused
    return ws


async def index(_request: web.Request) -> web.FileResponse:
    return web.FileResponse(WEB_DIR / "index.html")


def _port_free(host: str, port: int) -> bool:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
        except OSError:
            return False
    return True


def _open_browser(url: str) -> None:
    """Open the dashboard in a graphical browser if there's an obvious one (never a terminal browser)."""
    import os
    import platform
    import shutil
    import subprocess
    import sys
    import webbrowser

    quiet = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
    try:
        if "microsoft" in platform.release().lower():  # WSL: hand the URL to Windows
            if shutil.which("wslview"):
                subprocess.Popen(["wslview", url], **quiet)
            elif shutil.which("cmd.exe"):
                subprocess.Popen(["cmd.exe", "/c", "start", url], cwd="/mnt/c" if os.path.isdir("/mnt/c") else None, **quiet)
        elif sys.platform in ("darwin", "win32") or os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
            webbrowser.open(url)
    except Exception:
        pass


def serve(host: str = "127.0.0.1", port: int = 8765, device: str | None = None,
          open_browser: bool = True, cpu_rng: str = "sparse") -> None:
    if not _port_free(host, port):
        raise SystemExit(f"Port {port} is already in use (is FlyKart already running?). Try:  uv run --no-sync flykart --port {port + 1}")
    conn = Connectome.load()
    print(f"Loaded {conn.dataset}: {conn.n:,} neurons, {conn.n_connections:,} connections, {conn.n_synapses:,} synapses")
    sim = Sim(conn, device=device, cpu_rng=cpu_rng)
    backend = "Numba compiled kernels" if sim.brain.device.type == "cpu" else (
        "Triton kernel" if sim.brain.use_triton else "PyTorch ops")
    print(f"Brain on {sim.device_name} ({backend})")
    bench = sim.brain.benchmark(200.0)
    sim.brain.reset_state()
    print(f"Idle brain speed: {bench['realtime_factor']:.1f}x real time")
    if bench["realtime_factor"] < 0.9:
        print("  (slower than real time: the kart will move in slow motion. `uv run --no-sync flykart doctor` explains why.)")
    url = f"http://{'localhost' if host in ('127.0.0.1', '0.0.0.0') else host}:{port}"

    async def on_startup(app: web.Application) -> None:
        app["hub"] = Hub(sim, asyncio.get_running_loop())
        sim.start()
        if open_browser:
            _open_browser(url)

    async def on_cleanup(_app: web.Application) -> None:
        sim.stop()

    app = web.Application()
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    app.router.add_get("/", index)
    app.router.add_get("/ws", ws_handler)
    app.router.add_static("/", WEB_DIR, show_index=False)
    print(f"\n  FlyKart dashboard →  {url}\n  (press Ctrl+C to stop)\n")
    web.run_app(app, host=host, port=port, print=None)
