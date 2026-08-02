"""Pydantic request/response models for the Phase 3A local API.

Kept separate from the simulation kernel: nothing in ``drone_sim``'s core
package (``state.py``, ``simulation.py``, ...) imports from here, so the
kernel has no FastAPI/Pydantic dependency.
"""

from __future__ import annotations

from typing import List, Literal, Optional, Tuple

from pydantic import BaseModel, Field


class CreateSimulationRequest(BaseModel):
    """Minimal subset of ``SimulationConfig`` exposed for Phase 3A simulation creation.

    ``policy``/``scenario`` are Phase 3B additions -- narrow API-layer selection
    of an *existing* movement policy / scenario factory, never a new one. Both
    default to ``None``, reproducing the exact Phase 3A creation path
    (``RandomMovementAlgorithm``, ``DroneState.generate()``) when omitted.
    """

    num_drones: int = Field(..., gt=0, le=100_000)
    bounds_max: Tuple[float, float, float] = (1000.0, 1000.0, 1000.0)
    seed: int = 0
    dt: float = Field(1.0, gt=0)
    max_speed: float = Field(5.0, gt=0)
    collision_radius: float = Field(1.0, gt=0)
    near_miss_radius: float = Field(2.0, gt=0)
    policy: Optional[Literal["goal_directed", "local_avoidance"]] = None
    scenario: Optional[
        Literal[
            "head_on_collision", "crossing_paths", "near_miss", "parallel_safe",
            "stationary_obstacle", "converging_group", "rare_collision_background",
        ]
    ] = None
    #: Phase 5 additions -- consulted only when distributed=True; harmless
    #: (silently ignored) otherwise. See DistributedSimulationRuntime /
    #: DistributedCoordinator. num_workers capped at 32 as an informative
    #: bound, not a load-bearing one -- see README.md's Phase 5 "Docker"
    #: notes on matching it to the deployment's actual CPU limit.
    distributed: bool = False
    num_workers: int = Field(1, ge=1, le=32)
    num_partitions: Optional[int] = Field(None, ge=1)
    executor: Literal["sequential", "threads", "processes"] = "sequential"


class SimulationStatusResponse(BaseModel):
    simulation_id: str
    status: str
    tick: int
    num_drones: int
    execution_mode: Literal["single_process", "distributed"] = "single_process"
    num_workers: Optional[int] = None


class DronePosition(BaseModel):
    drone_id: int
    x: float
    y: float
    z: float
    vx: float
    vy: float
    vz: float


class ViewportResponse(BaseModel):
    simulation_id: str
    tick: int
    total_visible: int
    returned: int
    truncated: bool
    drones: List[DronePosition]


class HeatmapResponse(BaseModel):
    simulation_id: str
    tick: int
    x_bins: int
    y_bins: int
    x_edges: List[float]
    y_edges: List[float]
    counts: List[List[int]]
    max_density: int
    num_drones_included: int


class CollisionMarkerResponse(BaseModel):
    drone_a: int
    drone_b: int
    tick: int
    x: float
    y: float
    z: float
    distance: float
    relative_speed: float


class CollisionsResponse(BaseModel):
    simulation_id: str
    tick: int
    markers: List[CollisionMarkerResponse]


class MetricsResponse(BaseModel):
    simulation_id: str
    tick: int
    metrics: dict
    #: Phase 5: populated only for distributed=True simulations -- see
    #: DistributedCoordinator.metrics_summary().
    distributed_metrics: Optional[dict] = None


class CheckpointSaveRequest(BaseModel):
    """``name`` is a bare identifier, never a path -- the server always
    resolves it to ``<checkpoint dir>/<name>.npz`` (see routes.py's
    ``_resolve_checkpoint_path``), so this charset is restrictive by design
    (no ``/``, ``\\``, or ``.``) to make path traversal structurally
    impossible rather than merely checked-for."""

    name: str = Field(..., pattern=r"^[A-Za-z0-9_-]{1,64}$")


class CheckpointSaveResponse(BaseModel):
    simulation_id: str
    name: str
    tick: int
    num_drones: int
    size_bytes: int
    saved_at: str


class CheckpointLoadRequest(BaseModel):
    name: str = Field(..., pattern=r"^[A-Za-z0-9_-]{1,64}$")


class CheckpointLoadResponse(BaseModel):
    simulation_id: str
    name: str
    tick: int
    num_drones: int
    status: str


class CheckpointInfo(BaseModel):
    name: str
    tick: int
    num_drones: int
    size_bytes: int
    modified_at: str


class CheckpointListResponse(BaseModel):
    checkpoints: List[CheckpointInfo]
