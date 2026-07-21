# CLAUDE.md — Drone Collision Simulator

Persistent context for Claude Code sessions in this repo. Read `README.md` for
the full architecture; this file is the working contract for how to build here.

## What this project is

A high-throughput simulation of up to 100,000 autonomous drones in a bounded 3D
world. Collisions are meant to be uncommon during normal operation, with rare
controlled collision scenarios added later. Development order is strict:
**a correct, measured local kernel before any UI or distributed work.**

## Current status: Phase 1 kernel complete; a local debug viewer has been added

The local single-process kernel is complete. 73 tests pass (67 kernel tests +
6 visualization-calculation tests). Benchmarks run at 1k / 10k / 100k drones,
headless and unchanged.

On top of the unchanged kernel, a minimal Matplotlib-based **local debug
viewer** has been added (`src/drone_sim/visualization.py`, launched via
`scripts/run_visualizer.py`). It is a prototype for local debugging only —
**not** the Phase 3 production web dashboard (no React, FastAPI, REST,
WebSocket/SSE, Redis, or GPU code). Do NOT jump ahead to later phases unless
explicitly asked.

## How to run

Run everything from the repo root (the folder containing `pyproject.toml`):

```bash
python -m pytest -q                         # full suite (73 tests)
python benchmarks/benchmark_simulation.py   # 1k / 10k / 100k benchmark, headless
python scripts/run_visualizer.py --drones 10000 --render-every 5   # local debug viewer
```

`pyproject.toml` sets `pythonpath = ["src"]` and `testpaths = ["tests"]`, so the
`drone_sim` package resolves and tests are found automatically — but only when
run from the repo root with the directory structure intact. `tests/` is a
sibling of `src/`, NOT inside it. The viewer needs `matplotlib` (the `viz`
extra in `pyproject.toml`, also listed in `requirements.txt`).

## Layout

```
src/drone_sim/   config, state, movement, boundaries, spatial_hash,
                 collisions, metrics, simulation, visualization
tests/           test_movement, test_boundaries, test_spatial_hash,
                 test_collisions, test_visualization
benchmarks/      benchmark_simulation.py
scripts/         run_visualizer.py (launches the local debug viewer)
```

Per-tick pipeline (order matters, unchanged by visualization):
MovementSystem -> BoundaryManager -> SpatialHashGrid -> CollisionDetectionEngine
-> CollisionResolutionEngine -> MetricsCollector.

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
  per-drone loop).
- Correctness before optimization; measurements before infrastructure.
- The headless benchmark (`benchmarks/benchmark_simulation.py`) stays fully
  independent of the visualization module — it must run with no display and
  without importing `visualization.py`.

## Decisions made (so a fresh session doesn't relitigate them)

- Boundaries: reflect (clamp position + negate that velocity axis); `CLAMP`
  mode zeros the axis instead. Configurable via `BoundaryMode`.
- Collision resolution: equal-mass elastic response along the line of centers
  (swaps normal velocity component) plus minimal separation. Momentum and KE
  conserved — there are tests asserting this.
- Movement policies: `RandomMovementAlgorithm` (reproducible random walk) and
  `ScriptedMovementAlgorithm` (constant velocity). AI/avoidance is Phase 2.
- Reproducibility: state generation seeded from `config.seed`; movement RNG
  seeded from `config.seed + 1`. Same config -> identical runs.
- Benchmark world scales at ~64 cells/drone so collisions stay rare but nonzero.
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

REST/WebSocket/SSE APIs, RealtimeGateway, Redis or any DB, WorkerCoordinator /
distributed workers, PartitionExchangeService, neural nets, GPU acceleration,
and the React/Canvas/WebGL production dashboard. The Matplotlib local debug
viewer above is explicitly in scope as a Phase 1 debugging aid; it is not a
step toward the production dashboard's tech stack. Keep scope to the local
kernel (+ this debug viewer) until asked to advance.

## Context note

This file was seeded from a planning/implementation conversation held in the
claude.ai chat interface (not Claude Code). That chat does not transfer
automatically; this CLAUDE.md is the durable handoff. Update it when decisions
change so future sessions stay in sync.
