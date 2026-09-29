# CPU performance investigation — 2026-09-29

## Implemented CPU backend and measured improvement

The CPU backend now uses Python functions compiled to native machine code by Numba/LLVM. Neuron integration is fused and parallel; spikes use compact bitmasks; synaptic propagation partitions destination neurons between workers so writes do not race and each destination retains its original addition order. This removes the old per-step NumPy array scans, transpose copies, dense spike writes, and expanded edge-index arrays. No CUDA packages or GPU are needed.

The default is **four CPU workers with sparse stimulation draws**. A warmed full-connectome comparison measured **90.0 ms per 14.4 ms simulated frame**, versus **752.6 ms** for the saved NumPy baseline: approximately **8.4× faster**. This is about **0.16× real time**, or 6.3 wall-clock seconds per simulated second, before server and browser overhead. It still does not achieve real-time simulation.

| Implementation and configuration | Median brain time per 14.4 ms simulated frame |
|---|---:|
| Saved NumPy baseline | 752.6 ms |
| Native, full draws, 4 workers | 195.9 ms |
| Native, sparse draws, 1 worker | 187.5 ms |
| Native, sparse draws, 2 workers | 128.4 ms |
| Native, sparse draws, 4 workers — default | 90.0 ms |
| Native, sparse draws, 6 workers | 88.5 ms |

Six workers offered little additional throughput in this sample. Four is the default to leave CPU capacity for the browser and other applications. `FLYKART_CPU_THREADS=6 uv run --no-group cuda --extra cpu flykart` can override it on this machine.

### Live server and behavior validation

After restarting the server with four workers and sparse draws, a WebSocket measurement collected **30 steady-state frames after 25 warm-up frames**. Their 29 arrival intervals averaged **77.8 ms** (median 77.2 ms, range 73.7–96.7 ms), with about **13,362 firing neurons per frame**. That is **0.185× real time**, or about **5.4 wall-clock seconds per simulated second**. The earlier live NumPy sample averaged 604.6 ms, roughly 7.8 times longer; these separate live samples are not a controlled paired benchmark.

Eight Pause commands received acknowledgments in **9.4 ms median**, with a maximum of **13.3 ms** and minimum of 5.9 ms. The measurement ran without a headless test browser; the user's browser could still be connected. These timings include server updates and local WebSocket delivery, but not display refresh or animation interpolation. They are separate from the brain-only benchmark above, with different workload evolution and scheduling.

Full-connectome behavior checks confirmed forward chase movement, a leftward steering response to a target at −75°, a rightward response at +75°, and a giant-fiber stimulus producing a jump. All checked neural state arrays remained finite. Browser validation confirmed rendered kart movement, frozen pose and wheels on Pause, cleared activity on Reset, successful Run after Reset, the raw-flash toggle, and no JavaScript errors. **21 Python tests and 7 Node tests passed**, covering neural equivalence/replay, RNG behavior, server lifecycle controls, and frontend timing. Three additional CUDA regression tests are available and skipped on this CPU-only machine. Both Triton kernels and their launch code match the previous implementation, and a hardware-independent test checks that GPU frames still compute eight blocks together. Actual CUDA execution has not been tested here. These are focused correctness and behavior checks, not a validation of all long-running fly behavior.

### What changes, and what stays reproducible

The network still contains all **165,122 neurons and 25,563,197 connections**. The neuron equations, 0.1 ms integration step, 18-step delay block, and ordinary eight-block/14.4 ms sensory, motor, and physics coupling remain in place. No neurons were removed and no biological time steps were enlarged.

Sparse draws generate independent uniform random input only for neurons with positive external stimulation: 4,314 neurons in this benchmark. This preserves the modeled stimulation distribution, but **changes the seed-to-trajectory mapping** relative to the old full-grid draws. Reset remains deterministic within a chosen mode. Use `uv run --no-group cuda --extra cpu flykart --cpu-rng full` for the original CPU random-draw sequence; full mode still benefits from native kernels and parallel execution.

The comparison checked voltage, synaptic state, refractory state, counts, spike matrices, and synaptic input after three consecutive frames. Full mode matched the saved NumPy implementation bit for bit. Within each RNG mode, 2-, 4-, and 6-worker runs matched the corresponding 1-worker result. These measurements and the automated reference/replay tests support the implementation; they do not establish identical sparse-versus-full trajectories or exhaustive equivalence for every possible parameter setting.

### Controls and display

On the CPU, the server now processes and acknowledges commands between **1.8 ms simulation blocks**, rather than waiting for a whole eight-block frame. Idle and speed-limiter waits can be interrupted immediately. Pause commits any completed partial frame using its actual simulated duration; reset discards partial work together with the brain/world state. Poke and drive changes begin at the next block after closing any partial frame. Paused pokes advance the brain while the world remains frozen. Normal uninterrupted frames retain their established 14.4 ms coupling. GPU execution keeps its original eight-block compute batches to avoid extra kernel launches and host pacing; CPU-only RNG and Numba kernels are not used on CUDA.

The browser uses a smoothed activity glow, with raw batch flashes available as a view option, and interpolates kart poses according to packet timing. Interpolation can add up to roughly one observed packet interval of display delay, capped at one second; it never creates new simulation progress. Pause, reset, and stalled data stop apparent motion. Raster and eye panels redraw when their input changes instead of repeatedly repainting unchanged data.

### Benchmark method and artifacts

The native comparison ran on the same Ryzen 5 220/WSL2 machine as the historical investigation below. It first warmed the full-mode brain through 25 chase frames (360 ms simulated), reaching about 13,067 active neurons per frame. Each configuration then started from a cloned state and advanced three eight-block frames with the same held sensory input. The table reports the median of those three brain-only timings; compilation was excluded. The old live NumPy server remained active during this comparison, creating CPU/cache contention. Sparse and full modes produce different subsequent activity, so their workload is distributionally comparable rather than trajectory-identical. The saved NumPy baseline retains profiling instrumentation, and these short measurements remain sensitive to scheduling and cache effects.

A separate five-frame breakdown of the six-worker sparse configuration measured 26.3 ms in integration including RNG, of which 4.4 ms was RNG, plus 49.7 ms in propagation. These component timings come from a separate pass and should not be added to the table as though they were the same three samples. Synaptic propagation is now the largest measured component.

Artifacts are archived in `$HOME/.local/state/flykart/profiles/2026-09-29/`:

- `flykart_cpu_bench.py`: the implemented-backend comparison, adapted to locate its frozen baseline beside itself.
- `flykart_cpu_bench_parallel.jsonl`: the measured results summarized above, including all thread counts and state comparisons.
- `cpu_numpy_baseline.py`: a frozen copy of the previous instrumented NumPy implementation; the comparison extracts its class without running its historical top-level driver.
- `flykart_live_compiled.py` and `.json`: live WebSocket frame intervals and Pause acknowledgment samples. This script exercises Run, Pause, and Reset on the server at `localhost:8765`.
- `flykart_cpu_behavior.py` and `.json`: independent full-connectome chase, controlled left/right target, and giant-fiber jump checks.

From the repository root, the comparison can be run with `PYTHONPATH=. .venv/bin/python $HOME/.local/state/flykart/profiles/2026-09-29/flykart_cpu_bench.py`. It loads the full dataset and performs substantial computation, so pause the live simulation for a less contended comparison. The script uses backend internals from this implementation and may need adaptation after future backend changes. Brain-only and live measurements are reported separately above.

## Historical investigation: previous NumPy backend

**The remaining sections record the earlier baseline and temporary prototypes, before the native parallel backend above was integrated. Their timings, implementation descriptions, and proposed work are historical, not the current application state.**

The previous CPU implementation spent almost all server update time in neuron integration and synaptic propagation. The game, serialization, and local network were small by comparison. Visual batching added a separate source of perceived delay and pulsing.

### Historical environment and method

- AMD Ryzen 5 220, 6 cores / 12 hardware threads; WSL2; 30 GiB RAM available to WSL. About 26 GiB was available during profiling; no meaningful swapping was observed.
- CPU-only PyTorch 2.14.0, NumPy 2.5.3, Python 3.13.15. The Radeon 740M is not used by this backend.
- These measurements describe the previous NumPy integration and active-outgoing-edge CPU implementation, after the first round of optimizations, not the original dense PyTorch fallback.
- A passive live WebSocket sample observed 26 frames over 15.4 seconds, without sending control commands.
- A separate, instrumented simulation performed 25 warm-up frames followed by 20 measured frames. Each frame advanced 14.4 ms of simulation. The live server remained active, so shared CPU/cache/memory contention and profiling overhead affect absolute times. Component shares are more useful than treating these numbers as an isolated benchmark.
- The instrumented scene averaged 13,386 distinct firing neurons per frame and about 10 million active synaptic edge visits per frame. Benchmarks of a mostly silent freshly initialized brain understate sustained cost.

### Historical observed latency

Live update intervals averaged **604.6 ms** (median 602.0 ms, range 543–683 ms), for approximately **0.024× real time**. One simulated second therefore takes about 42 wall-clock seconds in this sample.

The independent instrumented run averaged **766.6 ms per update**, with the following breakdown:

| Work per 14.4 ms simulated update | Mean wall time | Share |
|---|---:|---:|
| Neuron state arithmetic and temporary arrays | 210.5 ms | 27.5% |
| Finding spikes and writing the spike matrix | 269.5 ms | 35.2% |
| Random-number generation | 74.3 ms | 9.7% |
| Transposing/copying synaptic input | 54.4 ms | 7.1% |
| Accumulating synaptic effects | 88.4 ms | 11.5% |
| Building edge indices, gathering edges, clearing input | 64.4 ms | 8.4% |
| Senses, input encoding, motor decoding, world physics, frame construction, serialization | 2.5 ms | 0.3% |

Rounding and small unclassified dispatch/timer costs explain the remainder. Brain processing totals 764.1 ms, or 99.7% of the instrumented update.

Raw timings and the original profiling script are saved in `$HOME/.local/state/flykart/profiles/2026-09-29/`. The historical `pipeline.py` driver assumes the old brain internals and should not be run unchanged against the current backend. The newer comparison above imports its frozen NumPy class with the required compatibility setup. Historical output files were written to `/tmp/flykart-profile/` and then archived.

### Why the previous CPU path was expensive

`Brain._block_cpu` advances 165,122 neurons through 18 steps per block. A displayed update contains eight blocks: **23,777,568 neuron updates** to produce only 14.4 ms of simulated behavior. Real-time execution would require 1.651 billion neuron updates per second, each involving several arithmetic, threshold, refractory, random-input, and memory operations.

The NumPy implementation performs many separate array operations per step. Each allocates or traverses large arrays. The `(neurons, 18)` spike layout also makes per-time-step column writes and scans strided, increasing memory traffic. The synaptic input is transposed and copied every block. Both that input and the random array are 11.34 MiB per block, or 90.7 MiB each per displayed update, before intermediate arrays and edge traversal.

These timings establish substantial array/layout costs; they are not hardware-counter proof of a particular memory-bandwidth or cache limit.

### Historical network, control latency, and visual behavior

- Live messages are roughly 54–56 KB, around 91 KB/s at the observed cadence. Only about 3.16 KB is JSON; most bytes are firing-neuron indices. Python JSON parsing measured 0.10–0.24 ms. A local HTTP request took about 2.5 ms, though HTTP timing is not a measurement of one-way WebSocket latency.
- The server's reported brain time approximately matches the packet interval. The client mainly waits for the next computation batch; it is not processing a large backlog of network data.
- Control commands are consumed at the start of an eight-block frame. A click can wait almost one compute batch before being handled, and its visible result can wait through another batch.
- Every neuron that fired anywhere in a 14.4 ms simulated window receives the same browser arrival timestamp. Its glow decays with a 220 ms real-time constant. At a 605 ms packet interval, brightness falls to about 6% before the next batch: visible flashing is expected.
- The kart eases toward its most recent position with a 45 ms constant, then waits for the next update. This makes movement appear to start and stop. Camera tracking further reduces visible movement of the kart itself.
- The speed selector only imposes a maximum rate; choosing `max` cannot remove compute latency.

An isolated browser replay of a representative packet measured a **0.93 ms average packet handler** (0.60 ms median), and **10.25 ms average JavaScript animation callback** including render submission and panel updates. The full 400-column raster and eye panels cost about **6.26 ms per redraw** and are redrawn every other animation frame even when data has not changed. Updating those panels only on new data is worthwhile.

The test browser used SwiftShader software rendering. Its 4 FPS is **not a measurement of the user's AMD-accelerated browser**. A requested 16 ms replay timer was throttled, so that run does not establish 60 Hz throughput. GPU execution can be asynchronous, so JavaScript render submission time is not full GPU frame latency. No JavaScript errors occurred. Browser measurements are archived as `browser.json` with the other profiling artifacts.

### Original recommended order of work

The native implementation above now covers recommendations 1–4; the list below preserves the reasoning behind that work.

1. **Compile/fuse CPU integration and propagation.** Keep neuron state local through all 18 steps, walk outgoing edges directly, and avoid temporary arrays and expanded edge-index lists. Preserve float32 operation order, the original random draws, and synaptic accumulation order; validate against the existing exact-state reference tests.
2. **Choose array layout with the compiled loops and reuse storage.** Eliminate the transpose and strided spike scans/writes rather than merely moving the same copies elsewhere. Parallelize independent neuron ranges after the serial fused implementation is validated. Do not parallelize conflicting synaptic writes without a deterministic reduction strategy.
3. **Reduce interaction delay separately from throughput.** Poll/acknowledge commands between 1.8 ms simulation blocks. Decouple intermediate visual messages from the established motor/physics cadence. Simply reducing `BLOCKS_PER_FRAME` also changes sensory/motor coupling frequency and should not be assumed behavior-neutral.
4. **Improve presentation.** Use arrival-aware interpolation and a smoothed firing-rate display, retaining raw spikes as an optional view. This improves continuity without increasing biological simulation throughput. A small display buffer can smooth timing at the cost of additional display latency.
5. **Consider alternative compute or model scope only if required.** The existing CUDA/Triton backend can use a compatible NVIDIA GPU elsewhere. Using this AMD GPU requires a different backend and compatibility work. Fewer neurons or larger integration steps can lower cost but change the model or its numerical behavior and need separate validation.

Disk cleanup, JSON compression, more WebSocket bandwidth, increasing the speed setting, and adding PyTorch threads alone do not address the measured dominant costs. Changing the random generator or generating only for stimulated neurons may help later, but changes seeded replay and should be treated separately from exact optimizations.

### Historical compiled CPU prototype

A temporary Numba prototype combined a strict float32 fused neuron loop with a direct serial outgoing-edge loop. It used `fastmath=False` and retained the original full PyTorch random draws. It was installed in an ephemeral environment for the experiment, not added to the repository dependencies or deployed to the live server.

After 25 simulated chase frames (360 ms; about 13,067 active neurons/frame), three complete eight-block comparisons measured:

| Trial | Previous NumPy implementation | Compiled prototype |
|---|---:|---:|
| 1 | 712 ms | 396 ms |
| 2 | 695 ms | 426 ms |
| 3 | 662 ms | 437 ms |

Aggregate speedup was **1.64×**, with identical voltage, synaptic state, refractory state, spikes, counts, and input arrays. Three individual blocks and three consecutive eight-block frames were checked. This remains roughly 0.033–0.036× real time; it is a measured improvement, not evidence that compilation alone achieves real time.

The first cold compilation took about 1.25 seconds for neuron integration and 0.40 seconds for propagation. Subsequent cached loading took about 0.13 and 0.003 seconds. Median microbenchmarks measured fused neuron arithmetic at 20.8 ms/block plus 8.3 ms/block for random draws, versus 58.8 ms/block for the current complete NumPy integration; serial propagation was 24.3 versus 34.9 ms/block. The temporary source and JSONL results are archived with the other profiling artifacts.

#### Historical improved traversal: 2.11× measured speedup

A second prototype changed propagation to scan contiguous spike rows in presynaptic-neuron order, while preserving the order of additions to each output. Native propagation fell from 17–21 ms/block to 11.3–11.5 ms/block in paired microbenchmarks.

| Trial | Previous NumPy implementation | Fused integration + contiguous propagation |
|---|---:|---:|
| 1 | 683 ms | 313 ms |
| 2 | 632 ms | 312 ms |

This is an aggregate **2.11× speedup**, or **0.046× real time** (about 22× slower than real time), on the same warmed full-connectome workload. All six state arrays matched bit for bit in both tested frames. The added kernel compiled in about 0.20 seconds. These short tests demonstrate a viable optimization, not an exhaustive guarantee for every parameter configuration or long trajectory. The improved implementation is a temporary prototype, not deployed application code.

Sources and measurements are archived as `flykart_numba_probe.py`, `flykart_numba_prem_probe.py`, and their corresponding `*_results.jsonl` files in the profiling directory. The prototypes use an ephemeral Numba environment; project dependencies and the live server were not changed by these experiments.

At the prototype stage, remaining opportunities included scanning spike rows contiguously, representing spikes as bitmasks rather than a dense float matrix, and parallelizing independent neuron updates. These are implemented in the current backend. The default 22-step refractory interval exceeds the 18-step block, so at most one spike per neuron occurs per default block; an event-time representation would require enforcing that parameter constraint. The implemented bitmask supports blocks that allow multiple spikes.

The historical full-grid RNG cost alone exceeded the real-time budget: 74.3 ms of work to advance 14.4 ms. Even eliminating every other measured cost would have left that serial RNG path around five times slower than real time. Generating random input only for neurons with nonzero stimulation was subsequently implemented as sparse mode, described above. It preserves the intended independent stimulation distribution but changes the seed-to-trajectory mapping; it is not bitwise-equivalent replay of full mode.
