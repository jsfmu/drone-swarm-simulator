# CLAUDE.md — Drone Collision Simulator

Persistent context for Claude Code sessions in this repo. Read `README.md` for
the full architecture; this file is the working contract for how to build here.

## What this project is

A high-throughput simulation of up to 100,000 autonomous drones in a bounded 3D
world. Collisions are meant to be uncommon during normal operation, with rare
controlled collision scenarios added later. Development order is strict:
**a correct, measured local kernel before any UI or distributed work.**

## Current status: Phase 1 kernel + Phase 2 AI/scenario control complete; a local debug viewer has been added

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

`NeuralAvoidanceMovementAlgorithm` and any online reinforcement learning,
REST/WebSocket/SSE APIs, RealtimeGateway, Redis or any DB, WorkerCoordinator /
distributed workers, PartitionExchangeService, GPU acceleration, and the
React/Canvas/WebGL production dashboard. The Matplotlib local debug viewer
above is explicitly in scope as a Phase 1 debugging aid; it is not a step
toward the production dashboard's tech stack. Deterministic avoidance
(`LocalAvoidanceMovementAlgorithm`) and its validation harness
(`CollisionRateValidator`) are the required prerequisite for neural
avoidance — they exist so a future learned policy has something trustworthy
to be measured against, not so it can be skipped. Keep scope to the local
kernel + Phase 2 AI/scenario control (+ the debug viewer) until asked to
advance.

## Context note

This file was seeded from a planning/implementation conversation held in the
claude.ai chat interface (not Claude Code). That chat does not transfer
automatically; this CLAUDE.md is the durable handoff. Update it when decisions
change so future sessions stay in sync.
