# CLAUDE.md — Drone Collision Simulator

Persistent context for Claude Code sessions in this repo. Read `README.md` for
the full architecture; this file is the working contract for how to build here.

## What this project is

A high-throughput simulation of up to 100,000 autonomous drones in a bounded 3D
world. Collisions are meant to be uncommon during normal operation, with rare
controlled collision scenarios added later. Development order is strict:
**a correct, measured local kernel before any UI or distributed work.**

## Current status: Phase 1 kernel + Phase 2 AI/scenario control complete; a local debug viewer has been added

> **This section is stale relative to the actual repository.** `README.md`
> documents a complete Phase 3A (snapshot layer, `SimulationRuntime`,
> viewport/heatmap/collision-marker queries, a FastAPI app under
> `src/drone_sim/api/`, and a static browser page) that this file's "Do NOT
> build yet" list below still says hasn't started. Treat `README.md` as
> authoritative for current scope until this file is reconciled with it — see
> the "Browser vs. Matplotlib viewer" investigation note near the bottom of
> this file for the one Phase 3A change made *during* a CLAUDE.md-governed
> session (and therefore recorded here per this file's own context note),
> which does not by itself bring the rest of this file up to date.
>
> **Further update:** Phase 3B, Phase 4, and now **Phase 5 (optimization and
> deployment) are also complete** — see README.md's "Phase 5: Optimization
> and deployment" section (authoritative) and this file's own "Phase 5"
> section near the bottom for a summary. This paragraph is intentionally
> layered on top of the stale-notice above rather than rewriting it, matching
> how the Phase 4/Phase 3B sessions before it handled the same situation —
> the header line above is now stale on *two* counts (Phase 3A and Phase 5),
> not reconciled here for the same reason those weren't: full reconciliation
> of this file is a larger, separate effort than any one session's changes.

The local single-process kernel (Phase 1) is complete and unchanged. Phase 2
(batched goal-directed movement, trajectory prediction, local collision
avoidance, controlled rare-collision scenarios, collision-rate validation) is
now also complete. 142 tests pass. `benchmark_simulation.py` (Phase 1 path,
1k/10k/100k, headless) remains unchanged (Phase 2 adds zero overhead to that
path — see "Invariants" below). A **separate** `benchmark_avoidance.py` now
measures the full Phase 2 avoidance tick path at 1k/10k/100k — both complete
successfully at 100,000 drones (goal_directed 232 ms/tick, local_avoidance
404 ms/tick, ~1.7-2.0x slowdown). Do not read `benchmark_simulation.py`'s
~7.3 ticks/second as covering avoidance — see README.md's "Phase 2 avoidance
benchmark" for the real numbers and the dominant bottleneck (candidate-pair
generation, computed twice per tick, ~69% of total tick time at every scale).

On top of the unchanged kernel, a minimal Matplotlib-based **local debug
viewer** has also been added (`src/drone_sim/visualization.py`, launched via
`scripts/run_visualizer.py`). It is a prototype for local debugging only —
**not** the Phase 3 production web dashboard (no React, FastAPI, REST,
WebSocket/SSE, Redis, or GPU code). Do NOT jump ahead to later phases unless
explicitly asked.

## How to run

Run everything from the repo root (the folder containing `pyproject.toml`):

```bash
python -m pytest -q                         # full suite (142 tests)
python benchmarks/benchmark_simulation.py   # Phase 1 path only: 1k/10k/100k, headless
python benchmarks/benchmark_avoidance.py    # Phase 2 avoidance path: 1k/10k/100k, headless
python scripts/run_visualizer.py --drones 10000 --render-every 5   # local debug viewer
uvicorn drone_sim.api.app:app --reload                             # Phase 3A API + browser page
python scripts/run_visualizer.py --remote                          # same viewer, polling that server instead
python benchmarks/benchmark_phase5.py                               # Phase 5: optimization/scaling/parallel-execution benchmark
python scripts/smoke_test.py --base-url http://127.0.0.1:8000       # Phase 5 deployment smoke test (add --base-url to skip Docker)
docker compose up --build                                           # Phase 5 local deployment (not verified in this repo's dev environment -- see README)
```

`pyproject.toml` sets `pythonpath = ["src"]` and `testpaths = ["tests"]`, so the
`drone_sim` package resolves and tests are found automatically — but only when
run from the repo root with the directory structure intact. `tests/` is a
sibling of `src/`, NOT inside it. The viewer needs `matplotlib` (the `viz`
extra in `pyproject.toml`, also listed in `requirements.txt`).

## Layout

```
src/drone_sim/   config, state, movement, trajectory, scenarios, validation,
                 boundaries, spatial_hash, collisions, metrics, simulation,
                 visualization
tests/           test_movement, test_trajectory, test_scenarios,
                 test_validation, test_simulation, test_benchmark_avoidance,
                 test_boundaries, test_spatial_hash, test_collisions,
                 test_visualization
benchmarks/      benchmark_simulation.py (Phase 1 path), benchmark_avoidance.py
                 (Phase 2 avoidance path)
scripts/         run_visualizer.py (launches the local debug viewer)
```

Per-tick pipeline, Phase 1 shape (order matters, unchanged by visualization):
MovementSystem -> BoundaryManager -> SpatialHashGrid -> CollisionDetectionEngine
-> CollisionResolutionEngine -> MetricsCollector.

**Phase 2 only changes this when a context-requiring policy is registered**
(currently only `LocalAvoidanceMovementAlgorithm`, via
`MovementAlgorithm.requires_context = True`). `SimulationEngine` checks this
once at construction; if false (the Phase 1 default: Random/Scripted only),
the tick is byte-for-byte the pipeline above. If true, it becomes:
PRE-MOVEMENT SpatialHashGrid -> TrajectoryPredictionService ->
NeighborFeatureBuilder (MovementContext) -> MovementSystem -> BoundaryManager
-> POST-MOVEMENT SpatialHashGrid (rebuilt, the actual detection authority) ->
CollisionDetectionEngine -> CollisionResolutionEngine -> MetricsCollector.
The pre-movement grid/pairs are never reused for real detection — they only
inform movement policies. See README.md's "Phase 2 tick flow" for the full
walkthrough and the trajectory-prediction math.

## Invariants — do not break these

- **State is structure-of-arrays NumPy** (`positions`/`velocities` float32,
  `active_mask` bool, `movement_policy_ids` int32). Never allocate per-drone
  Python objects at scale. `Drone`/`SpatialHashCell` stay logical concepts.
- **Spatial hash must match brute force exactly.** `CollisionDetectionEngine`
  has both `detect()` (spatial) and `detect_brute_force()` (O(N^2) reference).
  Any change to detection must keep them identical on the correctness tests.
- **Candidate pairs are unique** — each unordered pair at most once per tick
  (self-cell i<j + 13 forward neighbor offsets). Don't reintroduce double-counting.
- **`cell_size >= near_miss_radius`** is enforced in config and is what
  guarantees no interacting pair spans non-adjacent cells. Keep the assertion.
- Everything batched/vectorized; avoid per-drone Python loops in hot paths
  (this includes visualization: density grids use `numpy.histogram2d`, not a
  per-drone loop; and Phase 2's `NeighborFeatureBuilder`, which picks each
  drone's most-urgent candidate pair via a stable-sort group-argmin trick,
  the same pattern `SpatialHashGrid` already used for cell grouping).
- Correctness before optimization; measurements before infrastructure.
- The headless benchmark (`benchmarks/benchmark_simulation.py`) stays fully
  independent of the visualization module — it must run with no display and
  without importing `visualization.py`.
- **`SimulationEngine.step`'s optional `profile: TickProfile | None` param is
  the only stage-timing instrumentation.** Disabled by default (`None`,
  every pre-existing call site) — zero behavior change, negligible overhead.
  When passed a `TickProfile()`, `step()` fills in nanosecond stage timings
  for all 10 pipeline stages and marks `context_stages_skipped=True` when no
  `requires_context` policy is registered (never silently omits those
  stages). The only behavioral difference profiling introduces: it calls
  `grid.candidate_pairs()` once explicitly and passes it to
  `CollisionDetectionEngine.detect(state, grid, pairs=...)` (a new optional
  kwarg, `None` by default and behavior-preserving) so `post_pairs_ns` and
  `detection_ns` can be measured separately — pure overhead confined to the
  profiled run, never changes a detection result. `benchmarks/benchmark_avoidance.py`
  is the only consumer; `simulation.py` has no import of or dependency on
  the benchmark.
- **`DroneState` never invokes or references a `MovementAlgorithm`** — only
  integer `movement_policy_ids`. Policy objects live solely in
  `MovementSystem.policies`. Destinations (`goal_positions`) are assigned
  during scenario generation, never inside `MovementSystem.step()`.
- **Trajectory prediction estimates risk; it is never the collision
  authority.** `TrajectoryPredictionService` and `CollisionDetectionEngine`
  must never call each other. The post-movement spatial-hash rebuild and
  `CollisionDetectionEngine.detect()` remain the sole source of truth for
  whether a collision actually happened.
- **`LocalAvoidanceMovementAlgorithm`'s urgency is distance-gated.**
  `dist_urgency` (from `predicted_separation`) is provably `0` whenever a
  pair isn't `PREDICTED_COLLISION`/`PREDICTED_NEAR_MISS`; `time_urgency` only
  modulates an already-real threat's correction strength, it can never
  manufacture urgency on its own (this was a real bug found and fixed during
  implementation — a zero-relative-velocity pair's guarded `ttca=0` looks
  identical to an imminent collision unless gated this way).
- **Two distinct collision measurements, both from `CollisionEventAccumulator`
  in `validation.py`** (canonical `(min(a,b), max(a,b))` pairs throughout):
  - `collision_pair_ticks` — one unordered pair observed inside
    `collision_radius` during one tick; accumulates every tick a pair is
    colliding, including persistent ones (`+= number_of_current_collision_pairs`
    each tick). Measures total time spent colliding.
  - `unique_collision_events` — deduplicated by state-entry, not by tick. A
    continuously-overlapping pair counts as one event on the tick it enters
    that state (`current_pairs - previous_tick_pairs`); it is not recounted
    while it persists, but counts again if it separates and later re-collides.
  - Derived: `average_collision_pairs_per_tick = collision_pair_ticks / num_ticks`;
    `average_collision_duration_ticks = collision_pair_ticks / unique_collision_events`
    (`0.0`, never `NaN`, when there were no collisions).
  - Near misses are tracked in a **separate** `CollisionEventAccumulator`
    instance and never enter the collision metrics.
  - A fresh accumulator is created per `run_policy()` call — state never
    leaks across policy runs or seeds.

## Decisions made (so a fresh session doesn't relitigate them)

- Boundaries: reflect (clamp position + negate that velocity axis); `CLAMP`
  mode zeros the axis instead. Configurable via `BoundaryMode`.
- Collision resolution: equal-mass elastic response along the line of centers
  (swaps normal velocity component) plus minimal separation. Momentum and KE
  conserved — there are tests asserting this.
- Movement policies: `RandomMovementAlgorithm` (reproducible random walk),
  `ScriptedMovementAlgorithm` (constant velocity) — both Phase 1, unchanged.
  Phase 2 adds `GoalDirectedMovementAlgorithm` (steers to `goal_positions`,
  no avoidance, the no-avoidance comparison baseline) and
  `LocalAvoidanceMovementAlgorithm` (goal-directed + a bounded correction away
  from the single most urgent predicted threat, via `MovementContext`).
  `NeuralAvoidanceMovementAlgorithm` is planned future work, **not
  implemented** — do not add an empty/placeholder class for it.
- Reproducibility: state generation seeded from `config.seed`; movement RNG
  seeded from `config.seed + 1`. Same config -> identical runs. Goal-directed
  and local-avoidance policies use no randomness at all (fully deterministic
  given state + config).
- Benchmark world scales at ~64 cells/drone so collisions stay rare but nonzero.
- **Phase 2 scenarios** (`src/drone_sim/scenarios.py`): seven deterministic,
  seeded factories — `head_on_collision`, `crossing_paths`, `near_miss`,
  `parallel_safe`, `stationary_obstacle`, `converging_group`,
  `rare_collision_background` (many safe background drones + a small known
  number of injected collision courses/near misses, with reflective goals so
  it can also drive policy comparison). Timed scenarios bake `config.dt` into
  their geometry so precomputed ground truth always lands on a real tick.
- **Phase 2 validation** (`src/drone_sim/validation.py`):
  `CollisionRateValidator` runs the same (deep-copied) scenario world under
  Scripted/GoalDirected/LocalAvoidance and reports collision/near-miss rates
  (per 10,000 drone-seconds), avoidance success rate on known injected pairs,
  min separation, destination completion, travel time, speed, and stationary
  percentage. The fair "does avoidance help" comparison is
  `goal_directed_no_avoidance` vs `local_avoidance` (same goal-seeking
  traffic pattern) — `scripted_baseline` never seeks goals, so it sees a
  structurally different (sparser) traffic pattern and isn't an
  apples-to-apples comparator for the aggregate rate, only for the specific
  injected-pair ground truth.
- **Local debug viewer** (`src/drone_sim/visualization.py` +
  `scripts/run_visualizer.py`): Matplotlib-only, top-down (x/y) rendering of
  the real `Simulation`/`DroneState`/`DetectionResult` objects — no duplicate
  simulation logic, no changes to movement/boundaries/spatial hash/collision
  code. Density grid via `numpy.histogram2d`; collision markers at the x/y
  midpoint of `DetectionResult.collision_pairs` for the current render
  interval. Defaults to 10,000 drones (not 100,000) with `--render-every`
  ticks between redraws to stay responsive. Keyboard controls: Space
  (pause/resume), R (reset with same config/seed), Escape or window close
  (quit). Matplotlib is imported lazily inside `SimulationViewer` so the pure
  grid/marker functions stay unit-testable in a headless environment.

## Do NOT build yet (later phases)

> **Update:** this list (and the header above) predates Phase 3A/3B/4, all of
> which are now implemented — see README.md (authoritative) and the "Phase 4:
> Distributed execution" addendum near the bottom of this file.
> `WorkerCoordinator`/distributed workers/`PartitionExchangeService`-style
> functionality is **no longer** out of scope; it now exists as
> `drone_sim.coordinator.DistributedCoordinator` (worker pool + spatial
> partitions), local/in-process only. Still genuinely out of scope: online
> reinforcement learning, `NeuralAvoidanceMovementAlgorithm`, Redis or any DB,
> a real multi-machine cluster/RPC layer, GPU acceleration, and the
> React/Canvas/WebGL production dashboard's remaining unbuilt pieces beyond
> Phase 3B's first bounded version. Deterministic avoidance
> (`LocalAvoidanceMovementAlgorithm`) and its validation harness
> (`CollisionRateValidator`) remain the required prerequisite for neural
> avoidance, unchanged.
>
> **Phase 5 update:** Redis and GPU acceleration have now actually been
> *evaluated* (not just deferred) and explicitly rejected on measured
> evidence — see README.md's Phase 5 section, points 5-6, and
> `benchmarks/phase5_results/gpu_native_evaluation.json`. This is a different
> claim than "still out of scope": it means a real profiling/benchmarking
> pass was done and neither was justified, not that nobody looked. A process
> -backed (not GPU/Redis) executor **was** added — `WorkerPool`/
> `DistributedCoordinator`'s new opt-in `use_processes=True`, still entirely
> local/in-process (no real network/cluster), so the "real multi-machine
> cluster/RPC layer" exclusion above is unchanged. Neural avoidance, online
> RL, a real DB, and cloud/Kubernetes deployment remain untouched and out of
> scope.

## Browser vs. Matplotlib viewer investigation (Phase 3A bug fixes)

A session investigated why an active browser session ran substantially
slower than the Matplotlib viewer at 10,000 drones. Full writeup:
README.md's "Browser vs. Matplotlib viewer: isolated-vs-concurrent
investigation" section. Summary of what changed, so a fresh session doesn't
relitigate it:

- **Root cause (dominant, measured): orphaned runtime threads.** Every
  `POST /simulations` left the previous `SimulationRuntime`'s background
  thread running forever (nothing ever called `shutdown()` on it) — a page
  reload or "Apply / New simulation" click added another live full-speed
  simulation thread rather than replacing one. Fixed with a new
  `DELETE /simulations/{id}` endpoint (`routes.py`) plus `index.html` calling
  it on the previous simulation before creating a new one
  (`sessionStorage`-tracked so a reload can clean up too).
- **Secondary fix: `/frame`'s double JSON serialization + second unmeasured
  lock acquisition.** `get_frame()` serialized its full payload twice per
  request (one discarded) and called `get_status()` as a second, unmeasured
  lock acquisition after `get_snapshot_with_lock_wait()`. Fixed by
  `SimulationRuntime.get_snapshot_and_status_with_lock_wait()` (one lock read)
  and a single `json.dumps()` call with a cheap splice for the
  self-referential timing fields.
- **Checked, not a meaningful factor:** the browser's default UI-driven world
  (500x500x100) is 4.75x *sparser* (not denser) than the Matplotlib viewer's
  `world_side_for()`-scaled world at the same drone count — measured to have
  negligible effect on tick cost since both worlds have far more spatial-hash
  cells than drones.
- `TickProfile` (simulation.py) gained additive, opt-in-only occupancy/pair-
  count fields (`occupied_cells`, `mean_cell_occupancy`, `max_cell_occupancy`,
  `candidate_pair_count`, `collision_pair_count`, `near_miss_pair_count`,
  `active_drone_count`) — zero cost when `profile=None`, same invariant as
  before. `SpatialHashGrid.occupancy_stats()` is the new small accessor they
  read from.
- New `benchmarks/benchmark_viewer_comparison.py` — the controlled comparison
  behind all of the above; also runnable as four standalone `--demo`s.
- 218 tests now (was 205) — 13 new, all under the existing Phase 1/2/3A
  invariants; nothing about movement, collision detection, thresholds,
  spatial hashing, or boundary behavior changed.

## Matplotlib viewer as a second API client (`--remote`)

A later session made the Matplotlib viewer and the browser page able to
display **the same live simulation** at once, on request, rather than each
always owning an independent one. Full writeup: README.md's "Remote mode:
same data as the browser page" (under "Local debug viewer (prototype)").
Summary so a fresh session doesn't relitigate it:

- New `src/drone_sim/api_client.py` — a stdlib-`urllib`-only HTTP client
  (`create_simulation`, `start/pause/resume/reset/delete_simulation`,
  `get_status`, `get_frame`). Deliberately does not import `httpx`/`requests`
  or anything from `drone_sim.api`, so the kernel/viz side of the package
  gains no new hard dependency and stays decoupled from whatever HTTP client
  the API side happens to use — same import-boundary spirit as
  `drone_sim/api/app.py`'s "only place that imports FastAPI" comment, just
  from the other direction.
- `visualization.py` gained `RemoteSimulationViewer`, sharing the existing
  `SimulationViewer`'s Matplotlib scaffold (`_build_figure()`, factored out
  of both) but sourcing its grid from the server's already-binned
  `heatmap.counts` (see `heatmap.py`) instead of running
  `numpy.histogram2d` over local positions — it never touches raw drone
  positions or a local `Simulation` at all. `scripts/run_visualizer.py
  --remote` uses it: with no `--simulation-id` it creates+starts a
  simulation on `--api-url` and prints a `?simulation_id=`-joined browser
  URL; with `--simulation-id` it attaches to one already running (e.g. one
  the browser created) and requires `--x-max`/`--y-max` since there is no
  endpoint to recover a simulation's world bounds from its id alone.
- `static/index.html`'s `init()` now checks `?simulation_id=` before its
  normal `createSimulation()` call — present means *join* (skip creating,
  skip the reload-cleanup delete), absent means the original behavior,
  unchanged.
- Space/R in the remote viewer call the server's pause/resume/reset
  endpoints (shared state, visible to every client polling that id), not a
  client-local toggle. This surfaced a real bug during implementation:
  guessing pause-vs-resume from the viewer's last-polled status and silently
  swallowing a 409 let two Space presses faster than one poll interval eat
  the second keypress (see `RemoteSimulationViewer._toggle_pause_resume`'s
  docstring) — fixed by retrying the other action on a 409 instead of
  swallowing it, since the 409 itself proves which state the server was
  actually in. A viewer that created its own simulation still deletes it
  server-side on window close, mirroring `index.html`'s
  `stopSimulationIfAny` leak-prevention above.
- New `tests/test_api_client.py` spins up a real `uvicorn.Server` in a
  background thread on a free port (FastAPI's `TestClient` never opens a
  real socket, and `api_client` specifically needs one) to exercise the
  client against actual HTTP.
- 222 tests now (was 218) — 4 new; nothing about movement, collision
  detection, thresholds, spatial hashing, boundary behavior, or the existing
  local (non-`--remote`) viewer path changed.

**Follow-up fix, same session, once `--remote` was actually used against a
live shared simulation:** the GUI and browser showed very different
"collision" numbers even when pointed at the identical `simulation_id`, for
two compounding reasons, both now fixed:
- `RemoteSimulationViewer`'s metrics text only ever showed
  `total_collisions` (cumulative-since-start, from `RunningMetrics`) with no
  per-tick count at all -- comparing that against index.html's per-tick
  `collision markers: N` line looks wildly different by construction (a
  measured example: ~350 per tick vs. 146,824 cumulative), not because
  anything was broken. Fixed by adding a `collision markers: {len(markers)}`
  line to `_poll_and_redraw()`'s output, sourced from the same `frame.markers`
  index.html reads, so both clients show the *same-shaped* per-tick number
  next to the *same-shaped* cumulative one.
- `join_url()` returned only `?simulation_id=`, never the viewport it was
  actually polling -- `index.html`'s join path left its x_min/x_max/y_min/
  y_max inputs at their hardcoded 0-500 defaults regardless of what
  simulation was joined, so the two clients could silently query different
  windows of the same world (this happened not to truncate anything for
  every valid `--drones` value at the time, since `world_side_for(100_000)`
  ≈ 370 < 500, but was still a latent correctness gap, not something to rely
  on by coincidence). Fixed by having `join_url()` carry `x_min`/`x_max`/
  `y_min`/`y_max` as query params and `index.html`'s `init()` apply them to
  the input boxes before the first `refresh()`.
- 224 tests now (was 222) — 2 new, in `tests/test_visualization.py`
  (`RemoteSimulationViewer` attach-mode construction needs no network, so
  these run with `matplotlib.use("Agg")` and a monkeypatched
  `api_client.get_frame`, not a live server).

## Phase 4: Distributed execution (local logical workers)

A session implemented Phase 4 exactly as scoped by README.md's roadmap
(worker coordinator/pool, spatial partitions, boundary-drone exchange,
partition rebalancing, worker failure recovery) — **not** the older
GPU/CUDA-flavored "Phase 4" some prior planning conversations may have
implied; GPU work stays under Phase 5 ("Optional native or GPU
acceleration"), untouched. Full writeup: README.md's "Phase 4: Distributed
execution" section. Summary so a fresh session doesn't relitigate it:

- New modules, additive only — `simulation.py`/`Simulation` is completely
  unchanged: `src/drone_sim/partition.py` (`PartitionGrid`, deterministic
  X-axis slab partitioning — 1-D by design, not a full 3-D grid, since it
  makes neighbor/halo queries exact and trivial while satisfying every
  Phase 4 requirement), `src/drone_sim/worker.py` (`Worker`, `WorkerPool`,
  `WorkerLifecycleState`, fault injection), `src/drone_sim/coordinator.py`
  (`DistributedConfig`, `DistributedCoordinator`).
- **Ownership model:** one authoritative `World` lives in the coordinator;
  a drone's partition is derived each tick from its position (never stored),
  so ownership transfer on crossing a boundary is automatic. Which *worker*
  runs a given partition is the only thing rebalancing/failure-recovery
  mutates (`coordinator.partition_worker`); drone-to-partition assignment
  stays purely spatial and is never itself migrated.
- **Cross-partition collision dedup rule:** a pair `(i, j)` is kept only
  from the partition `p == min(owner(i), owner(j))` — lower partition id
  always wins the tie. Applied uniformly to collision/near-miss/candidate
  pairs.
- **Tick-level transactional commit:** movement + detection are always
  computed into fresh staging arrays; `world.state` is mutated exactly once,
  after a fully successful attempt. A `WorkerFailure` mid-attempt leaves
  authoritative state completely untouched. Retries use per-`(seed, tick,
  partition_id)`-derived RNG (never per-worker), so retries/reassignment are
  bit-for-bit deterministic regardless of which healthy worker ends up
  running a partition.
- **Explicitly out of scope, not silently approximated:**
  `LocalAvoidanceMovementAlgorithm` (`requires_context=True`) is rejected by
  `DistributedCoordinator.__init__` with `NotImplementedError` — correct
  cross-partition `MovementContext` exchange needs a second, pre-movement
  ghost round-trip that this phase does not implement. `Simulation` (plain,
  single-process) still supports it fully, unchanged.
- **Known, pre-existing (not new) subtlety:** `CollisionResolutionEngine`
  processes pairs sequentially, so a drone with 2+ simultaneous collisions
  in one tick can resolve in a different order — and therefore produce a
  slightly different result — depending on partition count. This is a
  property of the unmodified single-worker kernel becoming visible under
  partitioning, not a Phase 4 correctness bug; observed directly in
  `benchmark_distributed.py`'s dense 5,000-drone case (374 vs. 371 cumulative
  collisions between `coordinator_1w` and `coordinator_4w`, with every
  individual tick's collision *pairs* still agreeing exactly).
- New `benchmarks/benchmark_distributed.py` — `single_worker` vs.
  `coordinator_1w` vs. `coordinator_Nw`, all `GoalDirectedMovementAlgorithm`
  (deterministic, no RNG, so cross-partition-count agreement is a meaningful
  check) — reports overhead (not speedup: `coordinator_Nw` is measured
  slower than `single_worker` here, since N logical workers still run
  sequentially in one Python process/GIL by default), agreement, determinism,
  and rebalancing behavior.
- 276 tests now (was 241) — 35 new across `tests/test_partition.py`,
  `tests/test_worker.py`, `tests/test_coordinator.py`; nothing about
  movement, collision detection, thresholds, spatial hashing, boundary
  behavior, or the single-process `Simulation` path changed.

## Phase 5: Optimization and deployment

A session implemented Phase 5 as scoped by README.md's roadmap: measure
first, optimize only what measurements identify, evaluate (not
default-add) native/GPU/Redis infrastructure, then monitoring/checkpointing/
deployment. Full writeup: README.md's "Phase 5: Optimization and deployment"
section. Summary so a fresh session doesn't relitigate it:

- **The one real hot-path optimization, measured not assumed:**
  `SpatialHashGrid.build()`/`candidate_pairs()` (`spatial_hash.py`) gained a
  dense cell->unique-cell-index lookup array, used instead of
  `np.searchsorted()` whenever the world is dense enough for it to pay off
  (two cheap per-build guard constants; falls back to the original
  `searchsorted` path otherwise — the same trick regressed to 0.03x-0.6x on
  a sparse world, which is *why* the guards exist, not a hypothetical).
  Measured 1.3x-1.7x on `benchmark_simulation.py`/`benchmark_avoidance.py` at
  every scale from 1,000-100,000 drones, byte-identical collision/near-miss
  counts before and after. `build()` also stopped calling `np.unique()` on an
  array that `argsort()` had just fully sorted (a redundant O(n log n) now
  O(n)) — a smaller, free win folded into the same change.
- **`WorkerPool`/`DistributedCoordinator` gained a third, opt-in execution
  backend: `use_processes=True`** (mutually exclusive with the existing
  `use_threads`), a persistent `ProcessPoolExecutor` (not per-tick — that
  would pay spawn cost every tick). Added *because* measurement justified it:
  threading measured **no benefit** (GIL not released long enough by these
  NumPy calls), while a real process pool measured a genuine 1.2x-1.8x
  speedup at 4-8 workers / 20,000-100,000 drones — but is *worse* than
  sequential at 1 worker, which is why it's opt-in, not the new default.
  Fault injection is checked in the parent process before any job crosses the
  process boundary, so it works identically across all three modes.
  `WorkerPool.shutdown()`/`DistributedCoordinator.shutdown()` release the
  pool — always call it when done with `use_processes=True`, or worker
  processes leak.
- **GPU and native (Numba/Cython/Rust) acceleration were evaluated and
  explicitly rejected**, not silently skipped — see
  `benchmarks/phase5_results/gpu_native_evaluation.json` for the evidence
  (no CUDA-capable stack installed on the dev machine; the hot path is an
  irregular gather/scatter workload, not GPU-shaped; post-optimization
  profiling shows no remaining single Python-loop bottleneck to justify a
  new compiled dependency).
- **Redis was evaluated and explicitly rejected.** The existing SSE stream
  (`GET .../stream`, Phase 3B) already has no queue anywhere (latest-state
  semantics) — Phase 5 added direct tests for multi-consumer isolation
  (`test_stream_multiple_concurrent_consumers_each_get_advancing_frames`) and
  slow-consumer isolation
  (`test_stream_slow_consumer_does_not_block_simulation_or_other_consumers`),
  both passing, confirming the existing design already satisfies what Redis
  would otherwise be brought in to solve.
- **New `drone_sim.api.monitoring` module**: `GET /health` (liveness, always
  ok), `GET /ready` (readiness, gated on a FastAPI `lifespan` startup flag —
  a real, if currently trivial, distinction from `/health`), `GET /metrics`
  (per-simulation tick/status/drone-count/tick-timing/collision figures from
  the existing `SimulationSnapshot`/`RunningMetrics`, plus process RSS via
  new stdlib-only `drone_sim.process_metrics`, plus API request count/latency
  via a small timing middleware, plus streaming counters). Distributed
  -execution metrics (`DistributedCoordinator.metrics_summary()`) were **not**
  a live endpoint when Phase 5 first shipped — deliberate scope, not an
  oversight, since the API only ever ran a plain `Simulation` via
  `SimulationRuntime` at that point. **A follow-up session closed this**: see
  the "Distributed execution via the API" note below — `GET /metrics` now
  includes a per-simulation `"distributed"` key when applicable.
- **New `drone_sim.checkpoint` module**: versioned (`.npz`, not pickle),
  atomic (`os.replace()` after a fully-written temp file), deterministic
  -resume simulation checkpointing for a plain `Simulation`. Captures the
  movement RNG's exact bit-generator state (new
  `SimulationEngine.get_rng_state()`/`set_rng_state()`), not just
  `config.seed` — required for `RandomMovementAlgorithm` to resume its exact
  draw sequence. Deliberately does NOT persist locks/threads/sockets/worker
  handles, the full per-tick metrics history (diagnostic, unbounded — a
  resumed `Simulation` starts with an empty one, same as a brand-new one), or
  movement-policy objects (constructor arguments to `load_checkpoint()`, same
  as `Simulation` already takes). `load_checkpoint()` never starts background
  execution.
- **New deployment artifacts**: root `Dockerfile` (backend, multi-stage,
  non-root, `HOST`/`PORT` env vars), `frontend/Dockerfile` +
  `frontend/nginx.conf` (multi-stage, `VITE_API_BASE_URL` as a **build arg**
  since Vite bakes `import.meta.env.*` in at build time), `docker-compose.yml`
  (health-gated startup, bounded default resource limits), and
  `scripts/smoke_test.py` (build -> up -> wait for `/ready` -> create sim ->
  status -> step + `/frame` -> pause -> down; `--base-url` skips the
  docker-compose-managed steps to test the HTTP sequence alone). **Docker
  was not available on the machine this was originally implemented on** --
  it was subsequently installed (WSL2 + Docker Desktop) in the same session
  and a real `docker compose up --build` was run, which caught a genuine bug
  a non-containerized smoke test never could: `pyproject.toml` had no
  `[tool.setuptools.package-data]` entry, so a real (non-editable) `pip
  install ".[api]"` silently dropped `api/static/index.html` (every local
  dev workflow uses `pip install -e`, which never hits this). Fixed by
  adding `"drone_sim.api" = ["static/*", "static/**/*"]` to
  `pyproject.toml`. After the fix, `docker compose up --build` starts both
  containers healthy and `scripts/smoke_test.py` passes against them for
  real — see README.md's Phase 5 "Deployment" subsection.
- New `benchmarks/benchmark_phase5.py` — the single bounded, parametrized
  entry point (drone count, tick count, policy, local/distributed, worker
  count, executor choice, checkpoint benchmarking, profiling on/off,
  JSON/CSV output) the Phase 5 spec asked for, reusing
  `benchmark_simulation.py`'s world-scaling helper and
  `benchmark_avoidance.py`'s memory-measurement helper rather than
  duplicating them.
- **319 tests now (was 276)** — 43 new: 17 in `tests/test_checkpoint.py`, 8
  in `tests/test_monitoring.py`, 5 process-executor tests in
  `tests/test_worker.py`, 3 in `tests/test_coordinator.py`, 3 dense-lookup
  -equivalence tests in `tests/test_spatial_hash.py`, 2 event-transport
  tests in `tests/test_stream.py`. Nothing about movement, collision
  detection, thresholds, spatial hashing, boundary behavior, or any
  pre-existing endpoint's observable behavior changed. Frontend unchanged
  (still 39 Vitest tests, production build still succeeds) — Phase 5 made no
  frontend code changes.

## Distributed execution via the API (follow-up session, after Phase 5)

Closed a gap Phase 5 explicitly left open (see `DistributedCoordinator
.metrics_summary()`'s original docstring, and CLAUDE.md's Phase 5 summary
above): the FastAPI layer previously only ever drove a plain `Simulation` via
`SimulationRuntime`, never a `DistributedCoordinator`. Full writeup:
README.md's "Distributed mode via the API" subsection. Summary so a fresh
session doesn't relitigate it:

- **New `drone_sim.distributed_runtime.DistributedSimulationRuntime`** —
  deliberately duplicates `SimulationRuntime`'s lock/thread/pause-event/
  snapshot-publishing scaffolding rather than branching distributed-mode
  logic into `runtime.py` itself (~20+ tests, exercised transitively by
  nearly every API/stream test — too high-blast-radius for an opt-in
  feature). Drives `DistributedCoordinator.step()` instead of
  `Simulation.step()`; `build_snapshot()`/`RunningMetrics` work against it
  unmodified via duck typing (same `.world`/`.clock`/`.metrics` shapes).
- **`POST /simulations` gained `distributed`/`num_workers`/`num_partitions`/
  `executor` fields** (`"sequential"|"threads"|"processes"` — one enum, not
  two raw booleans, so the invalid "threads AND processes" combination is
  unrepresentable at the API boundary). `distributed=true` +
  `policy=local_avoidance` returns `400` (DistributedCoordinator's existing
  `NotImplementedError` for `requires_context` policies, caught and
  translated) — leak-safe, since that rejection happens before any worker
  pool is created.
- **Every other route handler needed zero changes** — `/frame`, `/stream`,
  `/viewport`, `/heatmap`, `/collisions`, pause/resume/step/reset/delete all
  already called only the method surface both runtime classes implement
  identically.
- **`GET /metrics` now includes a per-simulation `"distributed"` key**
  (absence, not null, distinguishes plain from distributed sims) sourced
  from `DistributedCoordinator.metrics_summary()` — the method Phase 5 added
  but never wired up.
- `reset()` on the new wrapper shuts down the *old* coordinator's worker pool
  before building a new one (required for `executor="processes"` to never
  accumulate orphaned pools across resets) — safe under the wrapper's lock
  because `reset()` already requires `status != RUNNING`, the same guarantee
  `SimulationRuntime.reset()` relies on.
- Verified against a real `uvicorn` process AND the real Docker container
  (`docker compose up --build`, then a manual `distributed=true,
  executor="processes"` create/step/`/metrics`/delete sequence against the
  container, confirmed no leaked worker processes afterward).
- **340 tests now (was 319)** — 21 new: 16 in `tests/test_distributed_runtime.py`
  (mirrors `tests/test_runtime.py`'s full lifecycle suite, plus
  process-pool leak-safety cases), 5 in `tests/test_api.py`. Zero changes to
  `tests/test_runtime.py`, `tests/test_runtime_timing.py`,
  `tests/test_coordinator.py`, `tests/test_worker.py`, `tests/test_snapshot.py`,
  or any pre-existing `tests/test_api.py`/`tests/test_stream.py`/
  `tests/test_monitoring.py` case — confirmed via `git diff` before and after.

## Context note

This file was seeded from a planning/implementation conversation held in the
claude.ai chat interface (not Claude Code). That chat does not transfer
automatically; this CLAUDE.md is the durable handoff. Update it when decisions
change so future sessions stay in sync.
