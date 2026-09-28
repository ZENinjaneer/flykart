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
    def __init__(self, conn: Connectome, device: str | None = None, seed: int = 0):
        super().__init__(daemon=True)
        self.conn = conn
        self.brain = Brain(conn.indptr, conn.indices, conn.weights, conn.n, device=device, seed=seed)
        self.iface = Interface(conn, self.brain.device)
        self.world = World(mode="chase")
        self.frame_s = BLOCKS_PER_FRAME * self.brain.D * self.brain.p.dt * 1e-3
        self.cmds: queue.Queue = queue.Queue()
        self.running = False
        self.settings = {"drive": 0.6, "assist": 0.0, "speed": 1.0}
        self.pokes: list[dict] = []
        self.listeners: list = []
        self.clients = 0
        self._stop = threading.Event()
        self._rtf = 1.0
        self._brain_ms = 0.0
        self._sps = 0.0
        self._last_senses = None
        dev = self.brain.device
        self.device_name = torch.cuda.get_device_name(dev) if dev.type == "cuda" else "CPU"

    # ------------------------------------------------------------------ API
    def status(self) -> dict:
        return {"type": "status", "running": self.running, "mode": self.world.mode, "settings": dict(self.settings)}

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

    def stop(self) -> None:
        self._stop.set()

    # ------------------------------------------------------------------ loop
    def _handle(self, c: dict) -> None:
        kind = c.get("cmd")
        if kind == "run":
            self.running = bool(c.get("on"))
        elif kind == "reset":
            self._reset()
        elif kind == "mode":
            if c.get("mode") in ("chase", "free"):
                self.world.mode = c["mode"]
                self._reset()
        elif kind == "set":
            key, val = c.get("key"), c.get("value")
            if key in self.settings and isinstance(val, (int, float)):
                self.settings[key] = float(val)
        elif kind == "poke":
            idx = self.iface.poke_group(c.get("id", ""))
            if idx is not None and len(idx):
                self.pokes.append({"idx": idx, "until": self.brain.t_ms + 400.0})
        elif kind == "pokeNeuron":
            idx = self.iface.poke_type_of(int(c.get("index", -1)))
            if idx is not None and len(idx):
                self.pokes.append({"idx": idx, "until": self.brain.t_ms + 400.0})
        for fn in self.listeners:
            fn(json.dumps(self.status()).encode(), text=True)

    def _reset(self) -> None:
        self.world.reset()
        self.brain.reset_state()
        self.iface.reset()
        self.pokes.clear()
        self._last_senses = None

    def run(self) -> None:
        brain, world, iface = self.brain, self.world, self.iface
        next_t = time.perf_counter()
        while not self._stop.is_set():
            try:
                while True:
                    self._handle(self.cmds.get_nowait())
            except queue.Empty:
                pass
            if self.clients == 0:
                time.sleep(0.05)
                next_t = time.perf_counter()
                continue

            t0 = time.perf_counter()
            if self.running or self._last_senses is None:
                senses = world.sense(self.frame_s)
                self._last_senses = senses
            else:  # paused: the world is frozen, but the brain keeps running for pokes
                senses = dict(self._last_senses, loom=[], loom_l=0.0, loom_r=0.0, touch=0.0, sugar=0.0)
            self.pokes = [p for p in self.pokes if p["until"] > brain.t_ms]
            brain.set_rates(iface.encode(senses, self.settings, self.pokes))
            counts = brain.run(BLOCKS_PER_FRAME).cpu().numpy()
            spk_idx = np.flatnonzero(counts)
            spk_idx_np = spk_idx.astype(np.uint32)
            act, motor, key_spikes, class_counts = iface.decode(counts, spk_idx, self.frame_s)
            brain_ms = (time.perf_counter() - t0) * 1e3
            if self.running:
                world.step(self.frame_s, {**act, "assist": self.settings["assist"]})

            n_spk = len(spk_idx_np)
            self._brain_ms += (brain_ms - self._brain_ms) * 0.1
            self._sps += (n_spk / self.frame_s - self._sps) * 0.1
            frame = {
                "nSpk": n_spk,
                "world": world.dynamic_state(),
                "senses": {k: round(float(senses[k]), 3) for k in ("target_l", "target_r", "loom_l", "loom_r", "sugar")},
                "eye": np.round(senses["eye"] * 255).astype(int).ravel().tolist(),
                "motor": motor,
                "act": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in act.items()},
                "keySpikes": key_spikes,
                "classCounts": class_counts,
                "perf": {"rtf": self._rtf, "brainMs": self._brain_ms, "spikesPerSec": self._sps},
                "tMs": round(brain.t_ms, 1),
            }
            payload = json.dumps(frame, separators=(",", ":")).encode()
            pad = (-(4 + len(payload))) % 4
            msg = struct.pack("<I", len(payload)) + payload + b" " * pad + spk_idx_np.tobytes()
            for fn in self.listeners:
                fn(msg, text=False)

            # Pace to the requested speed (0 = as fast as possible).
            speed = self.settings["speed"]
            wall = time.perf_counter() - t0
            if speed > 0:
                next_t += self.frame_s / speed
                delay = next_t - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                elif delay < -0.25:
                    next_t = time.perf_counter()
            else:
                next_t = time.perf_counter()
            elapsed = time.perf_counter() - t0
            self._rtf += (self.frame_s / max(elapsed, 1e-6) - self._rtf) * 0.1


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
    try:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                try:
                    hub.sim.submit(json.loads(msg.data))
                except json.JSONDecodeError:
                    pass
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


def serve(host: str = "127.0.0.1", port: int = 8765, device: str | None = None, open_browser: bool = True) -> None:
    if not _port_free(host, port):
        raise SystemExit(f"Port {port} is already in use (is FlyKart already running?). Try:  uv run flykart --port {port + 1}")
    conn = Connectome.load()
    print(f"Loaded {conn.dataset}: {conn.n:,} neurons, {conn.n_connections:,} connections, {conn.n_synapses:,} synapses")
    sim = Sim(conn, device=device)
    print(f"Brain on {sim.device_name} ({'Triton kernel' if sim.brain.use_triton else 'PyTorch ops'})")
    bench = sim.brain.benchmark(200.0)
    sim.brain.reset_state()
    print(f"Idle brain speed: {bench['realtime_factor']:.1f}x real time")
    if bench["realtime_factor"] < 0.9:
        print("  (slower than real time: the kart will move in slow motion. `uv run flykart doctor` explains why.)")
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
