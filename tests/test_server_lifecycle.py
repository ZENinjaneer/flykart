"""Exercise block-boundary controls without loading a connectome or compiling kernels."""

import json
import struct
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import torch

from aiohttp import WSMsgType

from flykart.server import BLOCKS_PER_FRAME, Sim, ws_handler


class FakeBrain:
    D = 18
    p = SimpleNamespace(dt=0.1)
    device = torch.device("cpu")
    use_triton = False
    n = 2

    def __init__(self, *args, **kwargs):
        self.t_ms = 0.0
        self.counts = torch.zeros(self.n, dtype=torch.int32)
        self.calls = []
        self.after_run = lambda: None

    def set_rates(self, rates):
        self.rates = rates

    def run(self, blocks):
        self.calls.append(blocks)
        self.t_ms += blocks * self.D * self.p.dt
        self.counts[:] = torch.tensor([2, 1]) * blocks
        self.after_run()
        return self.counts

    def reset_state(self):
        self.t_ms = 0.0
        self.counts.zero_()


class FakeInterface:
    def __init__(self, *args):
        self.decodes = []
        self.encodes = []

    def encode(self, senses, settings, pokes):
        self.encodes.append((settings.copy(), len(pokes)))
        return torch.ones(2)

    def decode(self, counts, spk_idx, dt_s):
        self.decodes.append((counts.copy(), dt_s))
        return {"throttle": 1.0}, {"gas": int(counts[0])}, counts.tolist(), [int(counts.sum())]

    def reset(self):
        pass

    def poke_group(self, name):
        return np.array([0]) if name == "test" else None

    def poke_type_of(self, index):
        return np.array([index]) if 0 <= index < 2 else None


class FakeWorld:
    def __init__(self, mode):
        self.mode = mode
        self.reset()

    def reset(self):
        self.t = 0.0
        self.steps = []

    def sense(self, dt):
        return dict(target_l=0, target_r=0, loom_l=0, loom_r=0, sugar=0,
                    eye=np.zeros((2, 1, 1)))

    def step(self, dt, act):
        self.steps.append(dt)
        self.t += dt

    def dynamic_state(self):
        return {"t": self.t}


class SimLifecycleTest(unittest.TestCase):
    def setUp(self):
        patches = patch.multiple("flykart.server", Brain=FakeBrain, Interface=FakeInterface, World=FakeWorld)
        patches.start()
        self.addCleanup(patches.stop)
        conn = SimpleNamespace(indptr=None, indices=None, weights=None, n=2)
        self.sim = Sim(conn)
        self.sim.clients = 1
        self.sim.settings["speed"] = 0
        self.messages = []
        self.on_message = lambda frame: None

        def receive(data, text):
            if text:
                frame = json.loads(data)
            else:
                size = struct.unpack_from("<I", data)[0]
                frame = json.loads(data[4:4 + size])
            self.messages.append(frame)
            self.on_message(frame)

        self.sim.listeners.append(receive)

    def frames(self):
        return [f for f in self.messages if "frameMs" in f]

    def test_complete_frame_keeps_eight_block_motor_coupling(self):
        self.sim.submit({"cmd": "run", "on": True})
        self.on_message = lambda f: self.sim.stop() if f.get("frameMs", 0) > 0 else None
        self.sim.run()

        self.assertEqual(self.sim.brain.calls, [1] * BLOCKS_PER_FRAME)
        self.assertEqual(self.sim.world.steps, [self.sim.frame_s])
        self.assertEqual(self.frames()[-1]["keySpikes"], [16, 8])
        self.assertEqual(len(self.sim.iface.encodes), 1)

    def test_gpu_keeps_original_eight_block_compute_batch(self):
        # Only the scheduling decision needs a CUDA device marker; FakeBrain
        # keeps this regression check runnable without NVIDIA hardware.
        self.sim.brain.device = torch.device("cuda")
        self.sim.submit({"cmd": "run", "on": True})
        self.on_message = lambda f: self.sim.stop() if f.get("frameMs", 0) > 0 else None
        self.sim.run()

        self.assertEqual(self.sim.brain.calls, [BLOCKS_PER_FRAME])
        self.assertEqual(self.sim.world.steps, [self.sim.frame_s])
        self.assertEqual(self.frames()[-1]["keySpikes"], [16, 8])
        self.assertEqual(len(self.sim.iface.encodes), 1)

    def test_pause_commits_three_blocks_before_ack_and_freezes_snapshots(self):
        sim = self.sim
        sim.submit({"cmd": "run", "on": True})
        sim.brain.after_run = lambda: sim.submit({"cmd": "run", "on": False}) if len(sim.brain.calls) == 3 else None
        paused_snapshots = 0

        def on_message(frame):
            nonlocal paused_snapshots
            if frame.get("frameMs") == 0 and not frame["running"] and frame["tMs"] > 0:
                paused_snapshots += 1
                if paused_snapshots == 2:
                    sim.stop()
                else:
                    sim.submit({"cmd": "snapshot"})

        self.on_message = on_message
        sim.run()
        self.assertEqual(sim.brain.calls, [1, 1, 1])
        self.assertAlmostEqual(sim.world.t, 0.0054)
        self.assertEqual(len(sim.iface.decodes), 2)  # Initial state, then completed blocks.
        partial = next(f for f in self.frames() if f["frameMs"] > 0)
        self.assertEqual(partial["keySpikes"], [6, 3])
        self.assertEqual(partial["frameMs"], 5.4)
        self.assertTrue(partial["running"])
        paused_ack = next(f for f in self.messages if f.get("type") == "status" and not f["running"])
        self.assertLess(self.messages.index(partial), self.messages.index(paused_ack))
        self.assertEqual(self.frames()[-1]["keySpikes"], [0, 0])
        self.assertEqual(self.frames()[-1]["motor"], partial["motor"])

    def test_reset_discards_partial_counts_and_resumes_from_zero(self):
        sim = self.sim
        sim.submit({"cmd": "run", "on": True})

        def after_run():
            if len(sim.brain.calls) == 3:
                sim.submit({"cmd": "reset"})

        def on_message(frame):
            if frame.get("epoch") == 1 and frame.get("frameMs", 0) > 0:
                sim.stop()

        sim.brain.after_run = after_run
        self.on_message = on_message
        sim.run()
        reset_frames = [f for f in self.frames() if f["epoch"] == 1]
        self.assertEqual(reset_frames[0]["tMs"], 0)
        self.assertEqual(reset_frames[0]["world"]["t"], 0)
        self.assertEqual(reset_frames[0]["frameMs"], 0)
        self.assertEqual(reset_frames[-1]["tMs"], 14.4)
        self.assertEqual(reset_frames[-1]["keySpikes"], [16, 8])
        self.assertEqual(sim.world.steps, [sim.frame_s])
        self.assertEqual(len(sim.brain.calls), 11)

    def test_run_after_paused_poke_does_not_move_world_for_poke_blocks(self):
        sim = self.sim
        sim.submit({"cmd": "poke", "id": "test"})
        sim.brain.after_run = lambda: sim.submit({"cmd": "run", "on": True}) if len(sim.brain.calls) == 3 else None
        self.on_message = lambda f: sim.stop() if f.get("frameMs", 0) > 0 and f["running"] else None
        sim.run()

        advanced = [f for f in self.frames() if f["frameMs"] > 0]
        self.assertEqual([f["frameMs"] for f in advanced], [5.4, 14.4])
        self.assertEqual(advanced[0]["world"]["t"], 0)
        self.assertEqual(sim.world.steps, [sim.frame_s])
        self.assertAlmostEqual(sim.brain.t_ms, 19.8)

    def test_paused_poke_expiry_commits_last_partial_frame_without_world_motion(self):
        sim = self.sim
        sim.submit({"cmd": "poke", "id": "test"})
        self.on_message = lambda f: sim.stop() if f.get("frameMs") == 0 and f["tMs"] > 400 else None
        sim.run()

        self.assertEqual(len(sim.brain.calls), 223)
        self.assertEqual(sim.world.steps, [])
        self.assertEqual(sim.pokes, [])
        advanced = [f for f in self.frames() if f["frameMs"] > 0]
        self.assertEqual(advanced[-1]["frameMs"], 12.6)
        self.assertAlmostEqual(sum(f["frameMs"] for f in advanced), sim.brain.t_ms)
        self.assertEqual(sum(sum(f["keySpikes"]) for f in advanced), 223 * 3)

    def test_drive_change_applies_at_next_block_without_dropping_counts(self):
        sim = self.sim
        sim.submit({"cmd": "run", "on": True})
        sim.brain.after_run = lambda: sim.submit({"cmd": "set", "key": "drive", "value": 0.9}) if len(sim.brain.calls) == 3 else None
        self.on_message = lambda f: sim.stop() if f.get("frameMs") == 14.4 else None
        sim.run()

        self.assertEqual([settings["drive"] for settings, _ in sim.iface.encodes], [0.6, 0.9])
        self.assertEqual([f["keySpikes"] for f in self.frames() if f["frameMs"] > 0], [[6, 3], [16, 8]])
        self.assertAlmostEqual(sim.world.t, 0.0198)

    def test_pause_wakes_slow_speed_pacing_wait(self):
        sim = self.sim
        sim.settings["speed"] = 0.001  # A blocking per-block sleep would last 1.8 seconds.
        completed_block = threading.Event()
        paused = threading.Event()
        sim.brain.after_run = completed_block.set
        self.on_message = lambda f: paused.set() if f.get("type") == "status" and not f["running"] else None
        sim.submit({"cmd": "run", "on": True})
        sim.start()
        try:
            self.assertTrue(completed_block.wait(1))
            started = time.perf_counter()
            sim.submit({"cmd": "run", "on": False})
            self.assertTrue(paused.wait(1), "Pause waited for the speed limiter")
            self.assertLess(time.perf_counter() - started, 1)
            self.assertEqual(len(sim.brain.calls), 1)
            self.assertAlmostEqual(sim.world.t, 0.0018)
        finally:
            sim.stop()
            sim.join(1)
        self.assertFalse(sim.is_alive())

    def test_malformed_neuron_indices_cannot_stop_simulation(self):
        sim = self.sim
        for index in (None, "bad", [], {}, 0.5, float("nan"), float("inf"), True, -1, 1000):
            sim.submit({"cmd": "pokeNeuron", "index": index})
        sim.submit({"cmd": "run", "on": True})
        self.on_message = lambda f: sim.stop() if f.get("frameMs", 0) > 0 else None
        sim.run()
        self.assertEqual(sim.pokes, [])
        self.assertEqual(sim.world.steps, [sim.frame_s])

    def test_snapshot_does_not_encode_or_consume_visual_input_history(self):
        sim = self.sim
        sim.running = True
        sim._snapshot()
        sim._snapshot()
        self.assertEqual(sim.iface.encodes, [])


class WebSocketCommandTest(unittest.IsolatedAsyncioTestCase):
    async def test_non_object_json_is_ignored_and_following_command_is_processed(self):
        commands = []
        sim = SimpleNamespace(init_payload=lambda: {"type": "init"}, submit=commands.append, clients=0)
        hub = SimpleNamespace(sim=sim, clients=set())
        request = SimpleNamespace(app={"hub": hub})
        ws = MagicMock()
        ws.prepare = AsyncMock()
        ws.send_str = AsyncMock()
        payloads = ['null', 'true', '1', '"run"', '[]', '{', '{"cmd":"run","on":true}']
        ws.__aiter__.return_value = [SimpleNamespace(type=WSMsgType.TEXT, data=data) for data in payloads]
        with patch("flykart.server.web.WebSocketResponse", return_value=ws):
            await ws_handler(request)
        self.assertEqual(commands, [
            {"cmd": "snapshot"}, {"cmd": "run", "on": True}, {"cmd": "run", "on": False},
        ])
        self.assertEqual(sim.clients, 0)


if __name__ == "__main__":
    unittest.main()
