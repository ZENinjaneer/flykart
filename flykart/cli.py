"""flykart [serve] | prepare | doctor | bench | drive"""

from __future__ import annotations

import argparse
import sys
import time
import warnings

warnings.filterwarnings("ignore", message=".*Sparse.*")


def cmd_prepare(args) -> None:
    from .connectome import is_prepared, prepare

    if is_prepared() and not args.force:
        print("Connectome already prepared (data/processed). Use --force to rebuild it.")
        return
    prepare()


def cmd_serve(args) -> None:
    from .connectome import is_prepared, prepare
    from .server import serve

    if not is_prepared():
        print("First run: downloading and preparing the connectome (~1.2 GB, a few minutes).\n")
        prepare()
        print()
    serve(host=args.host, port=args.port, device=args.device, open_browser=not args.no_browser,
          cpu_rng=args.cpu_rng)


def cmd_doctor(_args) -> None:
    """Check everything a first-time user needs, with a fix for each problem."""
    import platform
    import shutil

    from .connectome import PROCESSED, ROOT, WEB_DATA, is_prepared

    problems = 0

    def row(name: str, ok: bool | None, detail: str, fix: str = "") -> None:
        nonlocal problems
        mark = {True: "ok ", False: "!! ", None: "-- "}[ok]
        print(f"  {mark} {name:12s} {detail}")
        if ok is False:
            problems += 1
            if fix:
                print(f"      {'':12s} fix: {fix}")

    print("FlyKart doctor\n")
    py = sys.version.split()[0]
    row("Python", sys.version_info >= (3, 11), py, "uv installs a suitable Python for you: run ./setup.sh")
    wsl = "microsoft" in platform.release().lower()
    row("System", None, f"{platform.system()} {platform.machine()}" + (" (WSL2)" if wsl else ""))
    try:
        import torch

        row("PyTorch", True, torch.__version__)
        if torch.cuda.is_available():
            row("GPU", True, f"{torch.cuda.get_device_name(0)} (CUDA {torch.version.cuda})")
        else:
            row("GPU", None, "CUDA unavailable; using the compiled CPU backend")
        try:
            import numba

            row("CPU compiler", True, f"Numba {numba.__version__} (native neuron kernels)")
        except ImportError:
            row("CPU compiler", False, "Numba is not installed",
                "run ./setup.sh --cpu" if torch.version.cuda is None else "run ./setup.sh")
        if torch.cuda.is_available():
            try:
                import triton

                row("Triton", True, f"{triton.__version__} (fast GPU kernels)")
            except ImportError:
                row("Triton", None, "not installed: using PyTorch GPU operations")
    except ImportError:
        row("PyTorch", False, "not installed", "run ./setup.sh (or: uv sync)")
    three = ROOT / "web" / "vendor" / "three"
    ver = (three / "VERSION").read_text().strip() if (three / "VERSION").exists() else "?"
    row("Three.js", (three / "build" / "three.module.js").exists(), f"r{ver.split('.')[1] if '.' in ver else ver} (vendored)",
        "re-clone the repo: web/vendor/three is missing")
    if is_prepared():
        import json

        n = json.loads((WEB_DATA / "neurons.json").read_text())["n"]
        size = sum(f.stat().st_size for f in PROCESSED.glob("*")) / 1e6
        row("Connectome", True, f"{n:,} neurons ready ({size:.0f} MB in data/processed)")
    else:
        row("Connectome", False, "not downloaded yet", "uv run --no-sync flykart prepare  (or just: uv run --no-sync flykart)")
    free = shutil.disk_usage(ROOT).free / 1e9
    row("Disk", free > 3 or is_prepared(), f"{free:.0f} GB free", "the download needs ~1.5 GB free")
    print()
    if problems:
        print(f"{problems} problem(s) above. Fix them, then run:  uv run --no-sync flykart")
    else:
        print("All good. Start FlyKart with:  uv run --no-sync flykart")


def cmd_bench(args) -> None:
    import numpy as np
    import torch

    from .brain import Brain
    from .connectome import Connectome

    conn = Connectome.load()
    b = Brain(conn.indptr, conn.indices, conn.weights, conn.n, device=args.device, cpu_rng=args.cpu_rng)
    name = torch.cuda.get_device_name(b.device) if b.device.type == "cuda" else "CPU"
    backend = "Numba" if b.device.type == "cpu" else ("Triton" if b.use_triton else "PyTorch")
    print(f"{conn.dataset}: {conn.n:,} neurons, {conn.n_connections:,} connections on {name} ({backend})")
    photo = np.flatnonzero(conn.neurons.type.str.match(r"^R[1-8]").to_numpy())
    for label, rate_hz in (("silent", 0.0), ("photoreceptors at 60 Hz", 60.0)):
        r = torch.zeros(conn.n)
        r[torch.as_tensor(photo)] = rate_hz
        b.reset_state()
        b.set_rates(r)
        res = b.benchmark(args.ms)
        print(f"  {label:26s} {res['realtime_factor']:5.2f}x real time "
              f"({res['wall_ms']:.0f} ms for {res['sim_ms']:.0f} ms, {res['spikes']:,} spikes, "
              f"{res['active_neurons']:,} neurons active)")
    if b.device.type == "cuda":
        print(f"  peak GPU memory {torch.cuda.max_memory_allocated() / 1e6:.0f} MB")


def cmd_drive(args) -> None:
    """Run the kart headless and print what happened."""
    import numpy as np

    from .server import BLOCKS_PER_FRAME, Sim
    from .connectome import Connectome

    sim = Sim(Connectome.load(), device=args.device, cpu_rng=args.cpu_rng)
    sim.world.mode = args.mode
    sim._reset()
    sim.settings.update(drive=args.drive, assist=args.assist)
    if args.sigma:
        sim.iface.lc10a_sigma = args.sigma
    if args.peak:
        sim.iface.lc10a_peak = args.peak
    if args.steer_hz:
        sim.iface.steer_hz = args.steer_hz
    brain, world, iface = sim.brain, sim.world, sim.iface
    frames = int(args.seconds / sim.frame_s)
    steer, gaps, speeds, t0 = [], [], [], time.perf_counter()
    for f in range(frames):
        senses = world.sense(sim.frame_s)
        brain.set_rates(iface.encode(senses, sim.settings, []))
        counts = brain.run(BLOCKS_PER_FRAME).cpu().numpy()
        act, motor, _, _ = iface.decode(counts, np.flatnonzero(counts), sim.frame_s)
        world.step(sim.frame_s, {**act, "assist": sim.settings["assist"]})
        steer.append(act["steer"])
        speeds.append(world.kart.speed)
        if world.mode == "chase":
            gaps.append(senses["target"] is not None and abs(senses["target"][0]) < 45)
        if args.verbose and f % int(1 / sim.frame_s) == 0:
            k = world.kart
            print(f"t={world.t:5.1f}s v={k.speed:5.1f} steer={act['steer']:+.2f} thr={act['throttle']:.2f} "
                  f"DNa02 L/R={motor['steer'][0]:5.1f}/{motor['steer'][1]:5.1f} GF={motor['jump']:5.1f} "
                  f"MDN={motor['reverse']:4.1f} MN9={motor['feed']:5.1f} tgt={senses['target']}")
    wall = time.perf_counter() - t0
    e = world.events
    print(f"\n{args.seconds:.0f} s of driving in {wall:.1f} s wall ({args.seconds / wall:.2f}x real time)")
    print(f"laps {e.laps:.2f} | sugar {e.sugar} | jumps {e.jumps} (cleared {e.cleared}) | crashes {e.crashes}")
    print(f"mean speed {np.mean(speeds):.1f} | |steer| mean {np.mean(np.abs(steer)):.2f} | "
          f"on track {100 * world.on_track_frames / max(1, world.frames):.0f}%")
    if gaps:
        print(f"truck within ±45° of heading {100 * np.mean(gaps):.0f}% of the time")


def _serve_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to reach it from other machines")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--device", default=None, help="cuda / cpu (default: cuda if available)")
    p.add_argument("--no-browser", action="store_true", help="don't try to open a browser")
    _cpu_rng_arg(p)


def _cpu_rng_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument("--cpu-rng", choices=("sparse", "full"), default="sparse",
                   help="CPU random input: sparse draws only for stimulated neurons; full preserves the original draw sequence")


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="flykart",
        description="A real fruit-fly connectome drives a kart. Run with no command to start the dashboard.",
    )
    sub = ap.add_subparsers(dest="cmd")
    s = sub.add_parser("serve", help="run the simulation + dashboard (the default)")
    _serve_args(s)
    s.set_defaults(fn=cmd_serve)
    pr = sub.add_parser("prepare", help="download + preprocess the MaleCNS connectome (~1.2 GB)")
    pr.add_argument("--force", action="store_true", help="rebuild even if already prepared")
    pr.set_defaults(fn=cmd_prepare)
    sub.add_parser("doctor", help="check your setup and suggest fixes").set_defaults(fn=cmd_doctor)
    b = sub.add_parser("bench", help="measure brain simulation speed")
    b.add_argument("--ms", type=float, default=2000.0)
    b.add_argument("--device", default=None)
    _cpu_rng_arg(b)
    b.set_defaults(fn=cmd_bench)
    d = sub.add_parser("drive", help="drive headless and report")
    d.add_argument("--seconds", type=float, default=60.0)
    d.add_argument("--mode", default="chase", choices=["chase", "free"])
    d.add_argument("--drive", type=float, default=0.6)
    d.add_argument("--assist", type=float, default=0.0)
    d.add_argument("--device", default=None)
    _cpu_rng_arg(d)
    d.add_argument("--sigma", type=float, default=None, help="LC10a receptive field width (deg)")
    d.add_argument("--peak", type=float, default=None, help="LC10a peak stimulation rate (Hz)")
    d.add_argument("--steer-hz", type=float, default=None, help="DNa02 L-R difference for full lock (Hz)")
    d.add_argument("-v", "--verbose", action="store_true")
    d.set_defaults(fn=cmd_drive)
    argv = sys.argv[1:]
    if not argv or argv[0].startswith("-") and argv[0] not in ("-h", "--help"):
        argv = ["serve", *argv]  # `flykart` alone (or with serve flags) starts the dashboard
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
